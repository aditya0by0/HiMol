import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import rdkit
import sys
import logging, time
from tqdm import tqdm
import numpy as np
import wandb
from gnn_model import GNN
from decoder import Model_decoder

sys.path.append('./util/')

from data_utils import *



lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def identity_collate(batch):
    return batch


def group_node_rep(node_rep, batch_size, num_part):
    group = []
    super_group = []
    # print('num_part', num_part)
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count:count + num_atom])
        super_group.append(node_rep[count + num_all -1])
        count += num_all
    return group, super_group


def train(model_list, loader, optimizer_list, device):
    model, model_decoder = model_list

    model.train()
    model_decoder.train()
    # epoch-level accumulators for wandb logging
    epoch_totals = {'loss': 0.0, 'bond_if_auc': 0.0, 'bond_if_ap': 0.0,
                    'bond_type_acc': 0.0, 'atom_type_acc': 0.0,
                    'atom_num_rmse': 0.0, 'bond_num_rmse': 0.0}
    num_steps = 0
    if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        #batch内的每个item是MolTree类型
        batch_size = len(batch)

        graph_batch = molgraph_to_graph_data(batch)
        graph_batch = graph_batch.to(device)
        node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
        num_part = graph_batch.num_part
        node_rep, super_node_rep = group_node_rep(node_rep, batch_size, num_part)

        loss, bond_if_auc, bond_if_ap, bond_type_acc, atom_type_acc, atom_num_rmse, bond_num_rmse = model_decoder(batch, node_rep, super_node_rep)

        optimizer_list.zero_grad()

        loss.backward()

        optimizer_list.step()

        epoch_totals['loss'] += loss.item()
        epoch_totals['bond_if_auc'] += float(bond_if_auc)
        epoch_totals['bond_if_ap'] += float(bond_if_ap)
        epoch_totals['bond_type_acc'] += float(bond_type_acc)
        epoch_totals['atom_type_acc'] += float(atom_type_acc)
        epoch_totals['atom_num_rmse'] += float(atom_num_rmse)
        epoch_totals['bond_num_rmse'] += float(bond_num_rmse)
        num_steps += 1

        if_auc += bond_if_auc
        if_ap += bond_if_ap
        type_acc += bond_type_acc
        a_type_acc += atom_type_acc
        a_num_rmse += atom_num_rmse
        b_num_rmse += bond_num_rmse

        if (step+1) % 20 == 0:
            if_auc = if_auc / 20
            if_ap = if_ap / 20
            type_acc = type_acc / 20
            a_type_acc = a_type_acc / 20
            a_num_rmse = a_num_rmse / 20
            b_num_rmse = b_num_rmse / 20

            print('Batch:',step,'loss:',loss.item())
            if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0

    num_steps = max(num_steps, 1)
    return {k: v / num_steps for k, v in epoch_totals.items()}


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='./data/zinc/all.txt',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--output_model_file', type=str, default='./saved_model/pretrain.pth',
                        help='filename to output the pre-trained model')
    # The preprocessing cache makes __getitem__ a trivial lookup, so extra
    # DataLoader workers only duplicate the cached graphs across processes.
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument("--hidden_size", type=int, default=512, help='hidden size')
    parser.add_argument('--wandb_project', type=str, default='himol-pretrain',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name (default: auto)')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode (use "disabled" to turn off logging)')
    args = parser.parse_args()

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                     mode=args.wandb_mode, config=vars(args))

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = MoleculeDataset(args.dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=identity_collate, drop_last=True)

    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    model_decoder = Model_decoder(args.hidden_size, device).to(device)

    model_list = [model, model_decoder]
    optimizer = optim.Adam([{"params":model.parameters()},{"params":model_decoder.parameters()}], lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print('====epoch',epoch)
        metrics = train(model_list, loader, optimizer, device)
        print('====epoch', epoch, 'train metrics:', metrics)
        run.log({'epoch': epoch, **{'train/' + k: v for k, v in metrics.items()}})

        if not args.output_model_file == "":
            torch.save(model.state_dict(), args.output_model_file)

    run.finish()


if __name__ == "__main__":
    main()
