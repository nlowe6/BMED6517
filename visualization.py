import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc

class Visualizer:
    def __init__(self, results_dir):
        """
        Initialize the visualizer with the results directory.
        
        Args:
            results_dir (str or Path): Directory containing the results files
        """
        self.results_dir = Path(results_dir)
        self.summary_df = pd.read_csv(self.results_dir / 'summary_results.csv')
        
    def plot_roc_curves(self):
        """Plot ROC curves for all methods."""
        plt.figure(figsize=(10, 8))
        
        for method in self.summary_df['method']:
            results_file = self.results_dir / f'{method}_results.csv'
            if results_file.exists():
                results = pd.read_csv(results_file)
                # Get the AUROC values directly from the results
                auroc = results['auroc'].values
                # Create binary labels (1 for all entries since we're plotting AUROC values)
                y_true = np.ones_like(auroc)
                # Use AUROC values as scores
                fpr, tpr, _ = roc_curve(y_true, auroc)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{method} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.savefig(self.results_dir / 'roc_curves.png')
        plt.close()
    
    def plot_precision_recall_curves(self):
        """Plot precision-recall curves for all methods."""
        plt.figure(figsize=(10, 8))
        
        for method in self.summary_df['method']:
            results_file = self.results_dir / f'{method}_results.csv'
            if results_file.exists():
                results = pd.read_csv(results_file)
                # Get the AUROC values directly from the results
                auroc = results['auroc'].values
                # Create binary labels (1 for all entries since we're plotting AUROC values)
                y_true = np.ones_like(auroc)
                # Use AUROC values as scores
                precision, recall, _ = precision_recall_curve(y_true, auroc)
                avg_precision = np.mean(precision)
                plt.plot(recall, precision, label=f'{method} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.savefig(self.results_dir / 'precision_recall_curves.png')
        plt.close()
    
    def plot_target_aurocs(self):
        """Plot AUROC values for each target gene."""
        plt.figure(figsize=(15, 8))
        
        for method in self.summary_df['method']:
            results_file = self.results_dir / f'{method}_results.csv'
            if results_file.exists():
                results = pd.read_csv(results_file)
                plt.plot(results['target_id'], results['auroc'], 
                        label=f'{method} (mean = {results["auroc"].mean():.3f})')
        
        plt.xlabel('Target Gene ID')
        plt.ylabel('AUROC')
        plt.title('AUROC Values for Each Target Gene')
        plt.legend(loc="upper right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'target_aurocs.png')
        plt.close() 