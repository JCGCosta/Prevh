import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluation:
    def __init__(self,
                 metrics: pd.DataFrame,
                 confusion_matrix: list,
                 labels: list):
        self.metrics = metrics
        self.confusion_matrix = confusion_matrix
        self.labels = labels

    def plot_confusion_matrices(self):
        for f, fold in enumerate(self.confusion_matrix):
            for i, matrix in enumerate(fold):
                plt.figure(figsize=(6, 4))
                sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Not " + self.labels[i], self.labels[i]],
                            yticklabels=["Not " + self.labels[i], self.labels[i]])
                plt.title(f"[Fold-{f}] Confusion Matrix for Class: {self.labels[i]}")
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.show()

    def get_metrics(self):
        if self.metrics.shape[0] == 1: return self.metrics
        mean_df = pd.DataFrame(self.metrics.mean()).T
        mean_df.index = ['Mean']
        return pd.concat([self.metrics, mean_df])