Machine Learning Classifier for Actin and Myosin in Rotifers

This project implements a machine learning-based pipeline to detect and classify Actin and Myosin proteins in Rotifer Genomes.
The workflow integrates feature extraction, model training, protein classification and PFAM domain validation using a reproducible NextFlow pipeline.

Features/Highlights:
- Machine learning classification of Actin and Myosin proteins.
- Myosin subclass prediction (e.g. Myosin I, II, V, etc.)
- PFAM Domain Validation (Myosin_head PF00063)
- Automated dataset construction
- Phylogenetic tree validation (Maximum Likelihood)
- Fully reproducible NextFlow pipeline.
- Docker support for portability.

Repository Structure:

Required dependencies:
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
