#!/usr/bin/env python3

import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Extract predicted Actins and Myosins from Monte Carlo results using minimum ocurrence thresholds.")
    parser.add_argument("--input_csv", required=True, help="Input prediction CSV")
    parser.add_argument("--input_fasta", required=True, help="Original FASTA file")
    parser.add_argument("--output_actin_csv", required=True, help="Output CSV for predicted Actins")
    parser.add_argument("--output_myosin_csv", required=True, help="Output CSV for predicted Myosins")
    parser.add_argument("--output_myosin_fasta", required=True, help="Output FASTA for predicted Myosins (used for PFAM search)")
    parser.add_argument("--class_threshold", type=float, default=80.0, help="Minimum percentage required to keep a Actin prediction (default: 80.0)")
    parser.add_argument("--myosin_threshold", type=float, default=80.0, help="Minimum percentage required to keep a Myosin prediction (default: 80.0")
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

def SelectBest(df):
    if df.empty:
        return df.copy()

    return (df.sort_values(["Title", "Percentage"], ascending=[True, False]).drop_duplicates(subset=["Title"], keep="first").copy())

def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"Total rows in input CSV: {len(df)}")

    required_cols = {"Title", "Model", "Class", "Percentage"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")

    actins = df[(df["Class"] == "Actin") & (df["Percentage"] >= args.class_threshold)].copy()
    actins = SelectBest(actins)

    myosins = df[df["Class"].astype(str).str.startswith("Myosin_") & (df["Percentage"] >= args.myosin_threshold)].copy()
    myosins = SelectBest(myosins)

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
    print(f"Sequences missing: {missing}")
if __name__ == "__main__":
    main()
