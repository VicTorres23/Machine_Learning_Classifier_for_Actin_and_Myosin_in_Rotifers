#!/usr/bin/env python3

import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Filter only confirmed predictions where all models agree and prediction is not Not_Target.")
    parser.add_argument("--input_csv", required=True, help="Input prediction CSV")
    parser.add_argument("--input_fasta", required=True, help="Original FASTA file")
    parser.add_argument("--output_actin_csv", required=True, help="Output CSV for predicted Actins")
    parser.add_argument("--output_myosin_csv", required=True, help="Output CSV for predicted Myosins")
    parser.add_argument("--output_myosin_fasta", required=True, help="Output FASTA for predicted Myosins (used for PFAM search)")
    return parser.parse_args()

def read_fasta(fasta_file):
    sequences = {}
    header = None
    seq_lines = []

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences[header] = "".join(seq_lines)

                header = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            sequences[header] = "".join(seq_lines)
    return sequences

def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"Total rows in input CSV: {len(df)}")

    consensus = df[
        (df["Logistic_Regression"] != "Not_Target") &
        (df["Logistic_Regression"] == df["MLP"]) &
        (df["MLP"] == df["XGBoost"])
    ].copy()
    print(f"Consensus rows: {len(consensus)}")

    actins = consensus[consensus["XGBoost"] == "Actin"].copy()
    myosins = consensus[consensus["XGBoost"].astype(str).str.startswith("Myosin_")].copy()

    print(f"Actin rows: {len(actins)}")
    print(f"Myosin rows: {len(myosins)}")

    fasta_sequences = read_fasta(args.input_fasta)
    print(f"Sequences loaded from FASTA: {len(fasta_sequences)}")

    myosins["Sequence"] = myosins["Title"].astype(str).str.strip().map(lambda x: fasta_sequences.get(x, "").replace("*", ""))
    actins["Sequence"] = actins["Title"].astype(str).str.strip().map(lambda x: fasta_sequences.get(x, "").replace("*", ""))

    actins.to_csv(args.output_actin_csv, index=False)
    myosins.to_csv(args.output_myosin_csv, index=False)

    written = 0
    missing = 0

    with open(args.output_myosin_fasta, "w") as fasta:
        for _, row in myosins.iterrows():
            title = str(row["Title"]).strip()
            if title in fasta_sequences:
                seq = fasta_sequences[title]
                fasta.write(f">{title}\n{seq}\n")
                written += 1
            else:
                missing += 1
    print(f"Sequences written: {written}")
    print(f"Sequences written: {missing}")
if __name__ == "__main__":
    main()
