import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Build feedback dataset using PFAM validation of predicted myosins.")
    parser.add_argument("--prediction_csv", required=True)
    parser.add_argument("--myosin_csv", required=True)
    parser.add_argument("--hmmer_domtblout", required=True)
    parser.add_argument("--output_feedback_csv", required=True)
    parser.add_argument("--output_confirmed_csv", required=True)
    parser.add_argument("--output_hard_negative_csv", required=True)
    parser.add_argument("--output_uncertain_csv", required=True)
    parser.add_argument("--min_fragment_length", type=int, default=500)
    parser.add_argument("--hard_negative_score_cutoff", type=int, default=4)

    return parser.parse_args()

def Parse_domtblout(domtblout_file):
    """
    Extract proten IDs with PF00063 hits from HMMER domtblout.
    """

    pfam_hits = set()

    with open(domtblout_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            protein_id = parts[0]
            pfam_hits.add(protein_id)
    return pfam_hits

def compute_score(length, has_domain):
    """
    Simple scoring system for evaluating predicted myosins.
    :param length:
    :param has_domain:
    :return:
    """

    score = 0

    if has_domain:
        score += 5

    if length >= 500:
        score += 1

    if length >= 700:
        score += 1

    if length >= 900:
        score += 1

    return score

def main():
    args = parse_args()
    myosin_df = pd.read_csv(args.myosin_csv)
    pfam_hits = Parse_domtblout(args.hmmer_domtblout)
    myosin_df["Protein_ID"] = myosin_df["Title"].astype(str).str.replace(" ", "_")
    myosin_df["Sequence_Length"] = myosin_df["Sequence"].str.len()
    myosin_df["PF00063"] = myosin_df["Protein_ID"].isin(pfam_hits)
    myosin_df["Score"] = (myosin_df["PF00063"].astype(int) * 5 + (myosin_df["Sequence_Length"] >= 500).astype(int) + (myosin_df["Sequence_Length"] >= 700).astype(int) + (myosin_df["Sequence_Length"] >= 900).astype(int))

    def classify(row):
        if row["PF00063"]:
            return "domain_confirmed_myosin"
        elif row["Score"] >= args.hard_negative_score_cutoff:
            return "hard_negative_candidate"
        else:
            return "uncertain_fragment"

    myosin_df["Status"] = myosin_df.apply(classify, axis=1)
    myosin_df.to_csv(args.output_feedback_csv, index=False)
    confirmed = myosin_df[myosin_df["Status"] == "domain_confirmed_myosin"]
    hard_neg = myosin_df[myosin_df["Status"] == "hard_negative_candidate"]
    uncertain = myosin_df[myosin_df["Status"] == "uncertain_fragment"]

    confirmed.to_csv(args.output_confirmed_csv, index=False)
    hard_neg.to_csv(args.output_hard_negative_csv, index=False)
    uncertain.to_csv(args.output_uncertain_csv, index=False)

if __name__ == "__main__":
    main()