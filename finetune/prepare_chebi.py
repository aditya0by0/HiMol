"""One-time converter: python-chebai ChEBI `data.pkl` -> HiMol raw files.

`data.pkl` is written with numpy>=2 and holds pickled rdkit Mol objects, so it
cannot be read from HiMol's numpy<2 venv. Run THIS script with a numpy-2
interpreter (e.g. the anaconda base env or the python-chebai venv), NOT the
HiMol .venv. It emits plain-text + int8 .npy artifacts that HiMol's loader can
read regardless of numpy version.

Outputs (into <out_dir>, default finetune/dataset/chebi/raw/):
    smiles.txt   canonical SMILES, one per line, aligned to data.pkl row order
    ids.txt      chebi_id per line (join key for the split CSV)
    labels.npy   int8 [N, num_classes], values 0/1
    classes.txt  the class ids in label-column order

Usage:
    python finetune/prepare_chebi.py \
        --data_pkl <path/to/ChEBI50/processed/data.pkl> \
        --out_dir  finetune/dataset/chebi/raw
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem


DEFAULT_DATA_PKL = (
    r"C:\Users\sifluegel\PycharmProjects\python-chebai"
    r"\data\chebi_v251\ChEBI50\processed\data.pkl"
)
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "chebi", "raw")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_pkl", default=DEFAULT_DATA_PKL,
                        help="path to python-chebai processed data.pkl")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="output raw directory for HiMol")
    args = parser.parse_args()

    print("Reading %s ..." % args.data_pkl)
    df = pd.read_pickle(args.data_pkl)

    # Columns: chebi_id, mol, then one boolean column per class.
    class_cols = [c for c in df.columns if c not in ("chebi_id", "mol")]
    print("molecules: %d | classes: %d" % (len(df), len(class_cols)))

    os.makedirs(args.out_dir, exist_ok=True)

    smiles_list = []
    ids_list = []
    none_ct = 0
    for mol, cid in zip(df["mol"], df["chebi_id"]):
        if mol is None:
            # keep row alignment with an empty SMILES so labels stay aligned;
            # the HiMol loader will fail to parse it and skip it.
            smiles_list.append("")
            none_ct += 1
        else:
            smiles_list.append(Chem.MolToSmiles(mol))
        ids_list.append(str(cid))
    if none_ct:
        print("warning: %d rows had a None mol (written as empty SMILES)" % none_ct)

    labels = df[class_cols].to_numpy(dtype=np.int8)  # 0/1

    with open(os.path.join(args.out_dir, "smiles.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(smiles_list) + "\n")
    with open(os.path.join(args.out_dir, "ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids_list) + "\n")
    with open(os.path.join(args.out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(str(c) for c in class_cols) + "\n")
    np.save(os.path.join(args.out_dir, "labels.npy"), labels)

    print("wrote smiles.txt, ids.txt, classes.txt, labels.npy to %s" % args.out_dir)
    print("labels shape:", labels.shape, "dtype:", labels.dtype)


if __name__ == "__main__":
    main()
