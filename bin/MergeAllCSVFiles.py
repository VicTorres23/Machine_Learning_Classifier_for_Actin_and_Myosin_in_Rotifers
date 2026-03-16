#!/usr/bin/env python3

import os
import glob
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into one.")
    parser.add_argument("--input_folder", required=True, help="Folder containing CSV files")
    parser.add_argument("--output_file", required=True, help="Merged output CSV")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.input_folder, "*.csv")))

    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(args.output_file, index=False)

    print(f"Merged {len(csv_files)} CSV files → {args.output_file}")

if __name__ == "__main__":
    main()
