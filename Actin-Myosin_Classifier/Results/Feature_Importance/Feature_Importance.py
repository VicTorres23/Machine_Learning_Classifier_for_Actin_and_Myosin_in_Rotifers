import pandas as pd
import matplotlib.pyplot as plt
import joblib
from xgboost import plot_importance

xgb = joblib.load("xgboost_model.pkl")

df = pd.read_csv("/Users/victor_torres/PycharmProjects/Actin_Myosin/Training_Dataset_V3.csv")
X = df.drop(columns=["Title", "Classification"])

plt.figure(figsize=(10, 8))
plot_importance(xgb, max_num_features=20, importance_type='gain')  # Use 'gain', 'weight', or 'cover'
plt.title("Top 20 Feature Importances - XGBoost")
plt.tight_layout()
feature_names = X.columns  # Assuming X is your original DataFrame
features = []
#plt.show()
for i in [88, 2, 0, 15, 129, 94, 63, 98, 93, 41, 29, 159, 80, 114, 32, 67, 25, 5, 115, 44]:
    features.append(feature_names[i])
values = [16.62, 12.32, 10.72, 8.08, 3.44, 2.777, 2.774, 2.54, 2.12, 2.09, 1.98, 1.76, 1.75, 1.45, 1.23, 1.16, 1.02, 0.94, 0.89, 0.87]
plt.figure(figsize=(8, 5))
bars = plt.bar(features, values)
plt.xlabel('Features')
plt.ylabel('F Score')
plt.title('Top 20 Feature Importances - XGBoost')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()



