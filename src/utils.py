import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_distribution(df, save_path="images/churn_distribution.png"):
    """Plot the distribution of churn in the dataset."""
    os.makedirs("images", exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    churn_counts = df["Churn"].value_counts()
    colors = ["#4CAF50", "#FF5722"]
    plt.bar(["No Churn", "Churn"], churn_counts.values, color=colors)
    plt.title("Churn Distribution")
    plt.ylabel("Count")
    
    # Add value labels
    for i, v in enumerate(churn_counts.values):
        plt.text(i, v + 50, str(v), ha="center", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Churn distribution saved to {save_path}")

def plot_categorical_distribution(df, column, save_path=None):
    """Plot distribution of a categorical column against churn."""
    os.makedirs("images", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    crosstab = pd.crosstab(df[column], df["Churn"], normalize="index") * 100
    crosstab.plot(kind="bar", stacked=True, ax=plt.gca(), color=["#4CAF50", "#FF5722"])
    plt.title(f"Churn Rate by {column}")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.legend(title="Churn")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        save_path = f"images/{column.lower()}_churn.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")

def plot_numerical_distribution(df, column, save_path=None):
    """Plot distribution of a numerical column against churn."""
    os.makedirs("images", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    churned = df[df["Churn"] == "Yes"][column]
    not_churned = df[df["Churn"] == "No"][column]
    
    plt.hist([not_churned, churned], bins=30, alpha=0.7, 
             label=["No Churn", "Churn"], color=["#4CAF50", "#FF5722"])
    plt.title(f"{column} Distribution by Churn")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        save_path = f"images/{column.lower()}_distribution.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")

