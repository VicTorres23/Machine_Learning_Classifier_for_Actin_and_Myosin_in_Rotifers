#!/usr/bin/env python3

from Bio import SeqIO
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Split a FASTA file into chunked FASTA files.")
    parser.add_argument("--input_fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output_folder", required=True, help="Output folder for chunk FASTA files.")
    parser.add_argument("--sequences_per_file", type=int, default=1000, help="Sequences per chunk")
    parser.add_argument("--prefix", default="chunk", help="Prefix for chunk file names")
    parser.add_argument("--suffix", default="", help="Suffix to append to chunk names before .fasta")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    records = list(SeqIO.parse(args.input_fasta, "fasta"))
    total = len(records)

    for i in range(0, total, args.sequences_per_file):
        chunk = records[i:i+args.sequences_per_file]
        chunk_index = i // args.sequences_per_file + 1
        suffix = f"_{args.suffix}" if args.suffix else ""
        chunk_file = os.path.join(args.output_folder, f"{args.prefix}_{chunk_index}{suffix}.fasta")
        SeqIO.write(chunk, chunk_file, "fasta")
        print(f"Wrote {chunk_file} with {len(chunk)} sequences")

if __name__ == "__main__":
    main()
