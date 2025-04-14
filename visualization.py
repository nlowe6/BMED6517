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
                fpr, tpr, _ = roc_curve(results['target_id'], results['auroc'])
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
                precision, recall, _ = precision_recall_curve(results['target_id'], results['auroc'])
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
        """Plot target-specific AUROCs for all methods."""
        plt.figure(figsize=(12, 6))
        
        # Get all target IDs
        target_ids = set()
        for method in self.summary_df['method']:
            results_file = self.results_dir / f'{method}_results.csv'
            if results_file.exists():
                results = pd.read_csv(results_file)
                target_ids.update(results['target_id'])
        
        target_ids = sorted(list(target_ids))
        x = np.arange(len(target_ids))
        width = 0.8 / len(self.summary_df['method'])
        
        for i, method in enumerate(self.summary_df['method']):
            results_file = self.results_dir / f'{method}_results.csv'
            if results_file.exists():
                results = pd.read_csv(results_file)
                aurocs = results.set_index('target_id')['auroc'].reindex(target_ids).fillna(0)
                plt.bar(x + i*width, aurocs, width, label=method)
        
        plt.xlabel('Target Gene')
        plt.ylabel('AUROC')
        plt.title('Target-Specific AUROCs')
        plt.xticks(x + width*len(self.summary_df['method'])/2, target_ids, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / 'target_aurocs.png')
        plt.close() 