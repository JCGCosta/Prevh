import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class Evaluation:
    def __init__(self,
                 metrics: pd.DataFrame,
                 confusion_matrix: list,
                 labels: list):
        self.metrics = metrics
        self.confusion_matrix = confusion_matrix
        self.labels = labels

    def plot_confusion_matrices(self):
        num_folds = len(self.confusion_matrix)
        num_classes = len(self.labels)
        total_plots = num_folds * num_classes

        cols = min(num_classes, 10)
        rows = math.ceil(total_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes = axes.flatten()

        plot_index = 0
        for f, fold in enumerate(self.confusion_matrix):
            for i, matrix in enumerate(fold):
                ax = axes[plot_index]
                sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Not " + self.labels[i], self.labels[i]],
                            yticklabels=["Not " + self.labels[i], self.labels[i]], ax=ax)
                ax.set_title(f"[Fold-{f}] Confusion Matrix for Class: {self.labels[i]}")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                plot_index += 1

        plt.tight_layout()
        plt.show()

    def get_metrics(self):
        if self.metrics.shape[0] == 1: return self.metrics
        mean_df = pd.DataFrame(self.metrics.mean()).T
        mean_df.index = ['Mean']
        return pd.concat([self.metrics, mean_df])

    def get_accuracy_score(self):
        return self.metrics["accuracy"]

    def get_precision_score(self):
        return self.metrics["precision"]

    def get_recall_score(self):
        return self.metrics["recall"]

    def get_f1_score(self):
        return self.metrics["f1-score"]

    def get_r2_score(self):
        return self.metrics["r2-score"]

    def get_mse_score(self):
        return self.metrics["MSE"]

    def get_mae_score(self):
        return self.metrics["MAE"]