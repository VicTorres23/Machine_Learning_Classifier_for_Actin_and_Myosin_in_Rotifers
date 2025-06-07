import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

training_dataset = pd.read_csv("/Users/victor_torres/PycharmProjects/Actin_Myosin/Training_Dataset_V3.csv")

le = LabelEncoder()
training_dataset['Classification'] = le.fit_transform(training_dataset['Classification'])

X = training_dataset.drop(columns=["Title", "Classification"])
y = training_dataset["Classification"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP": MLPClassifier(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

for name, model in models.items():
    scores = cross_val_score(model, X_pca, y, cv=cv, scoring='f1_macro')
    print(f"{name} - Mean F1-Score: {scores.mean():.4f} ± {scores.std():.4f}")