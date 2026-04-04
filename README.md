### Machine Learning Classifier for Actin and Myosin in Rotifers

This project implements a machine learning-based pipeline to detect and classify Actin and Myosin proteins in Rotifer Genomes.
The workflow integrates feature extraction, model training, protein classification and PFAM domain validation using a reproducible NextFlow pipeline.

## Features/Highlights:
- Machine learning classification of Actin and Myosin proteins.
- Myosin subclass prediction (e.g. Myosin I, II, V, etc.)
- PFAM Domain Validation (Myosin_head PF00063)
- Automated dataset construction
- Phylogenetic tree validation (Maximum Likelihood)
- Fully reproducible NextFlow pipeline.
- Docker support for portability.

## Workflow Overview:

The project consists of two main Nextflow workflows:
1. **Build_Testing_Dataset.nf**
- Splits FASTA proteome into smaller chunks.
- Converts each chunk into descriptor-based CSV files.
- Merges all CSV files into a single testing dataset.

2. **Predict_Actin-Myosin.nf**
- Trains and applies the Actin/Myosin classifier.
- Performs 5-fold cross-validation.
- Extracts predicted actin and myosin sequences.
- Validates predicted Myosins using HMMER against PF00063.
- Builds a feedback table containing confirmed Myosins, uncertain fragmens, and hard negative candidates.

## Repository Structure:
```text
├── Actin/Myosin Classifier/              # ML Classifier for Actin/Myosin detection.
├── Myosin Classifier/                    # Myosin subclass classifier.
├── FASTA_Files/                          # Input genome FASTA sequences.
├── Maximum_Likelihood_Phylogenetic_Tree/ # Phylogenetic result
├── Plots_And_Diagrams/                   # Confusion matrices and plots
├── Results/                              # Prediction outputs
├── pfam/                                 # PFAM domain files
├── bin/                                  # Python Scripts
├── Build_Testing_Dataset.nf              # NextFlow Testing Dataset Builder
├── Predict_Actin-Myosin.nf               # Main prediction pipeline
├── nextflow.contig                       # NextFlow configuration
└── Dockerfile                            # Docker container definition
```
## Prerequisites:

- Python 3.10+
- NextFlow
- Docker
- HMMER
- PFAM database
- scikit-learn
- pandas
- numpy
- xgboost
- propy3
- matplotlib

## Installation:

Introduce the following commands into a Bash terminal:
```
git clone https://github.com/VicTorres23/Machine_Learning_Classifier_for_Actin_and_Myosin_in_Rotifers.git
cd Machine_Learning_Classifier_for_Actin_and_Myosin_in_Rotifers
docker pull ghcr.io/victorres23/ml_actin-myosin:v1.0
```

## How to run the Pipeline:

**Dataset Preparation:**
```
nextflow run Build_Testing_Dataset.nf \
  --input_fasta <fasta_file.fasta> \ # Path to the fasta file containing all sequences.
  --outdir <output_directory> \      # Output directory where the dataset will be stored.
  --sequences_per_file 1000          # Number of sequences per chunk.
```
**Sequence Prediction:**
```
nextflow run Predict_Actin-Myosin.nf \
  --training_dataset <training_dataset_to_be_used.csv> \ # Path to the training dataset CSV file used to train the model.
  --test_dataset <test_dataset_to_be_used.csv> \         # Path to the test dataset CSV file containing ProPy Descriptors.
  --test_fasta <fasta_file.fasta> \                      # Path to the fasta file containing all sequences used in the CSV file.
  --pf00063_hmm <PF00063.hmm> \                          # Path to the PFAM HMM file of PF00063 (myosin_head).
  --outdir <Results> \                                   # Path to the directory were all results of the pipeline will be stored.
  --output_name <output_name.csv> \                      # Name of the main prediction csv file.
  --hmm_evalue 1e-5 \                                    # E-value threshold for hmmsearch domain detection.
  --min_fragment_length 500 \                            # Minimum sequence length to consider full-length myosin.
  --hard_negative_score_cutoff 4                         # Score cutoff to label hard negative candidates
```
## Output Files:

Prediction and Performance Results:
```
prediction_results/
└── prediction_outputs/
    ├── Actin_Myosin_Predictions.csv             # All Classification Results.
    ├── Confusion_Matrix_Logistic_Regression.png # Confusion Matrix Results for Logistic Regression.
    ├── Confusion_Matrix_MLP.png                 # Confusion Matrix Results for Multi-Layer Perceptron.
    ├── Confusion_Matrix_XGBoost.png             # Confusion Matrix Results for eXtreme Gradient Boosting.
    ├── Logistic_Regression_Report.csv           # Logistic Regression Performance Metrics.
    ├── MLP_Report.csv                           # Multi-Layer Perceptron Performance Metrics.
    ├── XGBoost_Report.csv                       # eXtreme Gradient Boosting Performance Metrics.
    ├── PCA_CumulativeVariance.png               # Cumulative Variance Plot of the Principal Component Analysis.
    ├── PCA_PC1_PC2_AllPoints.png                # PCA Scatter Plot (PC1 vs PC2).
    ├── PCA_ScreePlot.png                        # PCA Scree Plot.
    ├── PCA_TopLoadings_PC1.png                  # Top feature loadings for PC1.
    ├── PCA_TopLoadings_PC2.png                  # Top feature loadings for PC2.
    ├── PCA_TopLoadings_PC3.png                  # Top feature loadings for PC3.
    ├── label_encoder.pkl                        # Saved label encoder.
    ├── scaler.pkl                               # Saved feature scaler.
    ├── pca_transformer.pkl                      # Saved PCA transformer.
    └── xgboost_model.pkl                        # Trained XGBoost Model.
```
5-Fold Cross Validation Results: 
```
cross_validation_results/
└── crossval_outputs/
    ├── Cross_Validation_F1_PerFold.csv       # F1 Score for each Fold.
    └── Model_Comparison_F1_Macro_5fold.png   # Macro F1 comparison across models.
```
Confirmed Predictions:
```
confirmed_predictions/
└── confirmed_outputs/
    ├── Predicted_Actins.csv    # Predicted Actin Sequences.
    ├── Predicted_Myosins.csv   # Predicted Myosin Sequences.
    └── Predicted_Myosin.fasta  # FASTA file of predicted Myosins.
```
PFAM Results:
```
pfam_validation/
└── pfam_outputs/
    ├── pf00063_hits.tbl              # Sequence-level PF00063 Hits.
    ├── pf00063_domtblout.tbl         # Domain level PF00063 Matches.
    └── pf00063_hmmsearch_stdout.txt  # Raw hmmsearch output.
```
Feedback Builder Results:
```
feedback_builder/
└── feedback_outputs/
    ├── Myosin_Feedback_Table.csv     # Summary feedback table.
    ├── Domain_Confirmed_Myosins.csv  # PFAM-confirmed Myosins.
    ├── Hard_Negative_Candidates.csv  # Likely false positives.
    └── Uncertain_Fragments.csv       # Partial/ambiguous sequences.
```
If you use this pipeline in your research, please cite:
Rodriguez-Torres V. E., Walsh E. J., Mohl J. E. Integrative Machine Learning Framework for Detecting Cytoskeletal-Related Proteins in Rotifers: A Case Study of Actin and Myosin Including Two Rotifer Genomes. <Add more information after publication>

Authors
Victor E. Rodriguez-Torres
Elizabeth J. Walsh
Jonathon E. Mohl
The University of Texas at El Paso

For questions about the timeline, please contact:
Victor E. Rodriguez-Torres
Computational Science PhD Program
The University of Texas at El Paso
vrodrigueztor@miners.utep.edu
