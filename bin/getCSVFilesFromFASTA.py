#!/usr/bin/env python3

from Bio import SeqIO
from propy.PyPro import GetProDes
import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate descriptor CSV for one FASTA file.")
    parser.add_argument("--input_fasta", required=True, help="Input FASTA chunk")
    parser.add_argument("--output_csv", required=True, help="Output CSV file")
    args = parser.parse_args()
    output_folder = os.path.dirname(args.output_csv)
    os.makedirs(output_folder, exist_ok=True)

    titles = []
    descriptors = []

    for rec in SeqIO.parse(args.input_fasta, "fasta"):
        sequence = str(rec.seq).upper().rstrip("*")
        title = rec.description
        print(title)
        if any(x in sequence for x in ['X', 'Z', 'B', 'U', '*']) or len(sequence) < 31:
            continue

        titles.append(title)
        descriptor = GetProDes(sequence)
        Features = descriptor.GetALL()
        print(Features)
        descriptors.append(Features)

    for i, desc in enumerate(descriptors):
            desc["Title"] = titles[i]

    if descriptors:
        df = pd.DataFrame(descriptors)
        df = df[['Title'] + [col for col in df.columns if col != 'Title']]
        df.to_csv(args.output_csv, index=False)
        print(f"Saved: {args.output_csv}")
    else:
        print(f"No valid sequences in: {args.fasta_file}")
        pd.DataFrame(columns=["Title"]).to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
