import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_values(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Valeurs manquantes dans le dataset")
    return plt
