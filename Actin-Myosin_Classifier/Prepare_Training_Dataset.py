from propy.PyPro import GetProDes
import pandas as pd
import regex as re
from Bio import SeqIO

#Actin_FASTA = "/Users/victor_torres/PycharmProjects/Actin_Myosin/positive_fasta_actin.fasta"
#Myosin_FASTA = "/Users/victor_torres/PycharmProjects/Actin_Myosin/positive_fasta_myosin.fasta"
#Negative_FASTA = "/Users/victor_torres/PycharmProjects/Actin_Myosin/uniprotkb_ACVR1_AND_reviewed_true_2025_04_08.fasta"
#More_Negatives = "/Users/victor_torres/PycharmProjects/Actin_Myosin/uniprotkb_RPL3_AND_reviewed_true_2025_04_09.fasta"
#Even_More_Negatives = "/Users/victor_torres/PycharmProjects/Actin_Myosin/uniprotkb_40S_Ribosomal_Protein_S3_AND_2025_04_09.fasta"
#filtered_rotifer_actin = "/Users/victor_torres/PycharmProjects/Actin_Myosin/filtered_rotifer_actin.fasta"
#filtered_rotifer_myosin = "/Users/victor_torres/PycharmProjects/Actin_Myosin/filtered_rotifer_myosin.fasta"
#GPCR_fasta = "/Users/victor_torres/PycharmProjects/Actin_Myosin/GPCR.fasta"
#sequences_10 = "/Users/victor_torres/PycharmProjects/Actin_Myosin/first_10_sequences.fasta"
#myosin_classified = "/Users/victor_torres/PycharmProjects/Actin_Myosin/myosin3_sequences.fasta"
#rotifer_myosin_test = "/Users/victor_torres/PycharmProjects/Actin_Myosin/Classes_and_Myosins.fasta"
#Myosin_2 = "/Users/victor_torres/Downloads/cymobase_myo-2.fasta"
Walsh_Rotifers = "/Users/victor_torres/Downloads/20250408_rotifer_proteins.fasta"

titles = []
descriptors = []
for rec in SeqIO.parse(Walsh_Rotifers, "fasta"):
    sequence = str(rec.seq)
    title = str(rec.description)
    if 'X' in sequence.upper() or 'Z' in sequence.upper() or 'B' in sequence.upper() or len(sequence) < 31 or 'U' in sequence.upper() or '*' in sequence.upper():
        continue

    #print(title)

    titles.append(title)
    print(title)
    print(sequence)

    descriptor = GetProDes(sequence.upper())
    Features = descriptor.GetALL()
    print(Features)
    descriptors.append(Features)

for i, desc in enumerate(descriptors):
    desc["Title"] = titles[i]
    #desc["Classification"] = "Myosin"
training_dataframe = pd.DataFrame(descriptors)
training_dataframe = training_dataframe[['Title'] + [col for col in training_dataframe.columns if col != 'Title']]
training_dataframe.to_csv("New_20250408_rotifer_proteins.csv", index=False)
