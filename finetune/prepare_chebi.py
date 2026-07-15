"""One-time converter: python-chebai ChEBI `data.pkl` -> HiMol raw files.

`data.pkl` is written with numpy>=2 and holds pickled rdkit Mol objects, so it
cannot be read from HiMol's numpy<2 venv. Run THIS script with a numpy-2
interpreter (e.g. the anaconda base env or the python-chebai venv), NOT the
HiMol .venv. It emits plain-text + int8 .npy artifacts that HiMol's loader can
read regardless of numpy version.

Handles both python-chebai schema generations:
    newer (e.g. v251): chebi_id (str), mol (rdkit Mol),  then one bool col per class
    older (e.g. v241): id (int), name (str), SMILES (str), then one bool col per class
When there is no `mol` column the SMILES column is parsed with rdkit instead.
Either way the output is canonical SMILES, so downstream files are identical in
shape and the HiMol loader needs no changes.

Outputs (into <out_dir>, default finetune/dataset/chebi/raw/):
    smiles.txt   canonical SMILES, one per line, aligned to data.pkl row order
    ids.txt      chebi_id per line (join key for the split CSV)
    labels.npy   int8 [N, num_classes], values 0/1
    classes.txt  the class ids in label-column order

Rows whose structure is missing or unparseable are written as an EMPTY SMILES
line rather than dropped: the loader keys labels/ids by line number, so row
alignment must be preserved. It skips the empty ones when building graphs.

Usage:
    python finetune/prepare_chebi.py \
        --data_pkl <path/to/ChEBI50/processed/data.pkl> \
        --out_dir  finetune/dataset/chebi/raw
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


DEFAULT_DATA_PKL = (
    r"C:\Users\sifluegel\PycharmProjects\python-chebai"
    r"\data\chebi_v251\ChEBI50\processed\data.pkl"
)
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "chebi", "raw")

# Column-name variants seen across python-chebai versions. Anything not matched
# here is treated as a class column, so every metadata name must be listed.
ID_COLS = ("chebi_id", "id")
MOL_COLS = ("mol",)
SMILES_COLS = ("SMILES", "smiles")
OTHER_META_COLS = ("name",)


def _pick(df, candidates):
    """Return the first of `candidates` present in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_columns(df):
    """Locate the id column and the structure column, whichever schema this is.

    :return: (id_col, mol_col, smiles_col) with exactly one of mol_col /
        smiles_col non-None (mol wins if both are present).
    """
    id_col = _pick(df, ID_COLS)
    if id_col is None:
        raise SystemExit(
            "no id column found (looked for %s); columns start with: %s"
            % (list(ID_COLS), list(df.columns[:8])))

    mol_col = _pick(df, MOL_COLS)
    smiles_col = _pick(df, SMILES_COLS)
    if mol_col is None and smiles_col is None:
        raise SystemExit(
            "no structure column found (looked for %s or %s); columns start "
            "with: %s" % (list(MOL_COLS), list(SMILES_COLS),
                          list(df.columns[:8])))
    if mol_col is not None:
        smiles_col = None  # prefer the pre-built Mol objects
    return id_col, mol_col, smiles_col


def build_smiles(df, mol_col, smiles_col):
    """Canonical SMILES per row, '' where the structure is missing/unparseable.

    :return: (smiles_list, n_failed)
    """
    smiles_list, failed = [], 0
    if mol_col is not None:
        for mol in df[mol_col]:
            if mol is None:
                smiles_list.append("")
                failed += 1
            else:
                smiles_list.append(Chem.MolToSmiles(mol))
    else:
        # Older dumps store SMILES text; parse it back through rdkit so the
        # output is canonicalised the same way the `mol` path is.
        for s in df[smiles_col]:
            mol = None
            if isinstance(s, str) and s.strip():
                mol = Chem.MolFromSmiles(s)
            if mol is None:
                smiles_list.append("")
                failed += 1
            else:
                smiles_list.append(Chem.MolToSmiles(mol))
    return smiles_list, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_pkl", default=DEFAULT_DATA_PKL,
                        help="path to python-chebai processed data.pkl")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="output raw directory for HiMol")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")  # unparseable rows are counted, not logged

    print("Reading %s ..." % args.data_pkl)
    df = pd.read_pickle(args.data_pkl)

    id_col, mol_col, smiles_col = resolve_columns(df)
    src_col = mol_col or smiles_col
    print("schema: id=%r structure=%r (%s)"
          % (id_col, src_col,
             "rdkit Mol objects" if mol_col else "SMILES text, parsed by rdkit"))

    # Everything that isn't known metadata is a class column.
    meta_cols = {id_col, src_col}
    meta_cols.update(c for c in OTHER_META_COLS if c in df.columns)
    class_cols = [c for c in df.columns if c not in meta_cols]
    print("molecules: %d | classes: %d | ignored metadata: %s"
          % (len(df), len(class_cols), sorted(meta_cols, key=str)))

    os.makedirs(args.out_dir, exist_ok=True)

    smiles_list, failed = build_smiles(df, mol_col, smiles_col)
    ids_list = [str(cid) for cid in df[id_col]]
    if failed:
        print("warning: %d/%d rows had a missing/unparseable structure "
              "(written as empty SMILES; the loader skips them)"
              % (failed, len(df)))

    labels = df[class_cols].to_numpy(dtype=np.int8)  # bool -> 0/1

    # loader.py stores the id as torch.tensor([int(id)]), and the files are
    # read back line-by-line, so ids must be numeric and newline-free.
    bad = [i for i in ids_list if not i.strip().lstrip("-").isdigit()]
    if bad:
        raise SystemExit("ids must be integers for loader.py's int(); %d are "
                         "not, e.g. %s" % (len(bad), bad[:5]))

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
