import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pca_df = pd.read_csv("PCA_Feature_Contributions.csv", index_col=0)

for pc in pca_df.columns[:5]:
    top_features = pca_df[pc].abs().sort_values(ascending=False).head(5)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title(f"Top 5 Contributing Features to {pc}")
    plt.xlabel("Absolute Loading Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()