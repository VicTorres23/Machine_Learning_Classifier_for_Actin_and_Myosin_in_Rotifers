import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
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

training_dataset = pd.read_csv("/Users/victor_torres/PycharmProjects/Actin_Myosin/Training_Dataset_V3.csv")

le = LabelEncoder()
training_dataset['Classification'] = le.fit_transform(training_dataset['Classification'])
joblib.dump(le, "label_encoder.pkl")

X = training_dataset.drop(columns=["Title", "Classification"])
Y = training_dataset["Classification"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

feature_names = X.columns
loadings = pca.components_
pca_df = pd.DataFrame(
    loadings.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=feature_names
)
pca_df.to_csv("PCA_Feature_Contributions.csv")
print("Saved PCA feature contributions to 'PCA_Feature_Contributions.csv'")

for pc in pca_df.columns[:5]:
    top_features = pca_df[pc].abs().sort_values(ascending=False).head(5)
    print(top_features)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca_transformer.pkl")

X_train, X_test, y_train, Y_test = train_test_split(X_pca, Y, test_size=0.3, random_state=42, stratify=Y)

def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report_dict = classification_report(Y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f"{name}_Report.csv")
    print(f"{name} report saved as {name}_Report.csv")
    return model

def getConfusionMat(y_true, y_pred, title):
    poster_colors = LinearSegmentedColormap.from_list("tealish", ["#e0f2f1", "#80cbc4", "#004d40"])
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=le.classes_)
    display.plot(cmap=poster_colors)
    plt.title("Confusion Matrix "+title)
    plt.show()

lr = evaluate_model(LogisticRegression(max_iter=1000), "Logistic_Regression")
joblib.dump(lr, "logistic_regression_model.pkl")
getConfusionMat(Y_test, lr.predict(X_test), "Logistic Regression")

mlp = evaluate_model(MLPClassifier(max_iter=1000, random_state=42), "MLP")
joblib.dump(mlp, "mlp_model.pkl")
getConfusionMat(Y_test, mlp.predict(X_test), "MLP")

xgb = evaluate_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "XGBoost")
joblib.dump(xgb, "xgboost_model.pkl")
getConfusionMat(Y_test, xgb.predict(X_test), "XGBoost")

test_data = pd.read_csv("/Users/victor_torres/PycharmProjects/Actin_Myosin/Brian_Sequences.csv")
titles = test_data["Title"]
X_unlabeled = test_data.drop(columns=["Title"])

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

results_df.to_csv("Sequences_Predicted.csv", index=False)
print("Predictions saved to GPCR_Predictions_Combined.csv")

