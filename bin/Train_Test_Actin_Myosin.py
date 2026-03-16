#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
from xgboost import XGBClassifier
import os
import numpy as np
import argparse

#OUTPUT_DIR = "/Users/victor_torres/Documents/Ph_D_in_Computational Sciences/Spring_Semester_2026/Research/Project_Files/Machine_Learning_Classifier_for_Actin_and_Myosin_in_Rotifers/Plots_and_Diagrams"

#training_dataset = pd.read_csv("/Users/victor_torres/PycharmProjects/Actin_Myosin/Myosin_Classifier_Training_Dataset.csv")
#training_dataset = pd.read_csv("/Users/victor_torres/PycharmProjects/Actin_Myosin/Training_Dataset_V3.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Actin/Myosin classifies and predict unlabeled sequences.")
    parser.add_argument("--training_dataset", required=True, help="CSV file for training")
    parser.add_argument("--test_dataset", required=True, help="CSV file for unlabeled/test prediction")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--output_name", default="Actin_Myosin_Predictions.csv", help="Prediction output file name")
    return parser.parse_args()

def PlotTopLoadings(pca_df, output_dir, pc="PC1", top_n=15):
    s = pca_df[pc].sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)
    plt.figure(figsize=(8, 6))

    colors = plt.cm.Greys(np.linspace(0.3, 0.8, len(s)))

    plt.barh(s.index, s.values, color=colors)
    plt.gca().invert_yaxis()

    plt.xlabel("Loading (weight)")
    plt.title(f"Top {top_n} Feature Loadings for {pc}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"PCA_TopLoadings_{pc}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def getConfusionMat(y_true, y_pred, labels, title, output_dir):
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(cmap="Greys", ax=ax, colorbar=False)

    ax.set_title("Confusion Matrix " + title, fontsize=26)
    ax.set_xlabel("Predicted Label ", fontsize=26)
    ax.set_ylabel("True Label ", fontsize=26)
    ax.tick_params(axis="both", labelsize=22)

    for text in display.text_.ravel():
        text.set_fontsize(26)

    save_path = os.path.join(output_dir, f"Confusion_Matrix_{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def evaluate_model(model, name, X_train_pca, y_train, X_test_pca, Y_test, output_dir):
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    report_dict = classification_report(Y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(output_dir, f"{name}_Report.csv"))
    print(f"{name} report saved")
    return model

def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    training_dataset = pd.read_csv(args.training_dataset)

    le = LabelEncoder()
    training_dataset['Classification'] = le.fit_transform(training_dataset['Classification'])
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

    X = training_dataset.drop(columns=["Title", "Classification"])
    Y = training_dataset["Classification"]

    X_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    joblib.dump(pca, os.path.join(output_dir, "pca_transformer.pkl"))

    feature_names = X.columns
    loadings = pca.components_
    pca_df = pd.DataFrame(
        loadings.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=feature_names
    )

    pca_df.to_csv("PCA_Feature_Contributions.csv")
    print("Saved PCA feature contributions to 'PCA_Feature_Contributions.csv'")

    expl_var = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(expl_var) + 1), expl_var, marker="o", color="black", markerfacecolor="gray")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PCA_ScreePlot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    Cumulative_Var = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(Cumulative_Var) + 1), Cumulative_Var, marker="o", color="black", markerfacecolor="gray")
    plt.axhline(0.95, linestyle="--", color="darkgray")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PCA_CumulativeVariance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if "PC1" in pca_df.columns:
        PlotTopLoadings(pca_df, output_dir, "PC1", top_n=15)
    if "PC2" in pca_df.columns:
        PlotTopLoadings(pca_df, output_dir, "PC2", top_n=15)
    if "PC3" in pca_df.columns:
        PlotTopLoadings(pca_df, output_dir, "PC3", top_n=15)

    X_all_scaled = scaler.transform(X)
    X_all_PCA = pca.transform(X_all_scaled)

    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100

    plt.figure(figsize=(8, 6))
    plt.scatter(X_all_PCA[:, 0], X_all_PCA[:, 1], s=12, alpha=0.35, color="gray", edgecolors="none", rasterized=True)
    plt.xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    plt.ylabel(f"PC2 ({pc2_var:.1f}% variance)")
    plt.grid(alpha=0.2)
    plt.title("PCA Projection (PC1 vs PC2)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PCA_PC1_PC2_AllPoints.png"), dpi=300, bbox_inches="tight")
    plt.close()

    for pc in pca_df.columns[:5]:
        top_features = pca_df[pc].abs().sort_values(ascending=False).head(5)
        print(top_features)

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca_transformer.pkl")

#X_train, X_test, y_train, Y_test = train_test_split(X_pca, Y, test_size=0.3, random_state=42, stratify=Y)

    lr = evaluate_model(LogisticRegression(max_iter=1000), "Logistic_Regression", X_train_pca, y_train, X_test_pca, Y_test, output_dir)
    joblib.dump(lr, "logistic_regression_model.pkl")
    getConfusionMat(Y_test, lr.predict(X_test_pca), le.classes_, "Logistic Regression", output_dir)

    mlp = evaluate_model(MLPClassifier(max_iter=1000, random_state=42), "MLP", X_train_pca, y_train, X_test_pca, Y_test, output_dir)
    joblib.dump(mlp, "mlp_model.pkl")
    getConfusionMat(Y_test, mlp.predict(X_test_pca), le.classes_, "MLP", output_dir)

    xgb = evaluate_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "XGBoost", X_train_pca, y_train, X_test_pca, Y_test, output_dir)
    joblib.dump(xgb, os.path.join(output_dir, "xgboost_model.pkl"))
    getConfusionMat(Y_test, xgb.predict(X_test_pca), le.classes_, "XGBoost", output_dir)

    test_data = pd.read_csv(args.test_dataset)
    titles = test_data["Title"]
    X_unlabeled = test_data.drop(columns=["Title"])
    X_unlabeled = X_unlabeled[X.columns]

    X_unlabeled_scaled = scaler.transform(X_unlabeled)
    X_unlabeled_pca = pca.transform(X_unlabeled_scaled)

    lr_preds = le.inverse_transform(lr.predict(X_unlabeled_pca))
    mlp_preds = le.inverse_transform(mlp.predict(X_unlabeled_pca))
    xgb_preds = le.inverse_transform(xgb.predict(X_unlabeled_pca))

    results_df = pd.DataFrame({
        "Title": titles,
        "Logistic_Regression": lr_preds,
        "MLP": mlp_preds,
        "XGBoost": xgb_preds
    })
    results_df.to_csv(os.path.join(output_dir, args.output_name), index=False)

if __name__ == "__main__":
    main()
