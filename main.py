import os
import numpy as np
import pandas as pd
from pathlib import Path
from data_loader import GRNDataLoader
from grn_methods import GRNReconstructor
from evaluation import GRNEvaluator
from scipy import stats
import time
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_roc_prc_curves(ground_truth, prediction, output_dir='plots', filename_prefix='grn'):
    """
    Plot and save ROC and Precision-Recall curves as PNG files.
    
    Args:
        ground_truth (np.ndarray): Binary ground truth matrix (n_tfs, n_targets)
        prediction (np.ndarray): Predicted regulatory scores (same shape)
        output_dir (str): Directory to save plots
        filename_prefix (str): Prefix for saved plot filenames
    """
    y_true = ground_truth.flatten()
    y_score = prediction.flatten()
    
    # Compute ROC and PRC data
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ROC Plot
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'{filename_prefix}_roc_curve.png')
    plt.close()

    # PRC Plot
    plt.figure()
    plt.plot(recall, precision, label=f'AUPR = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'{filename_prefix}_prc_curve.png')
    plt.close()


def compare_methods(results_dir):
    """
    Compare the performance of all methods against correlation.
    
    Args:
        results_dir (str): Directory containing results from all datasets
    """
    # Initialize lists to store metrics for each method
    method_metrics = {
        'lasso': {'aurocs': [], 'aps': []},
        'tigress': {'aurocs': [], 'aps': []},
        'ensemble': {'aurocs': [], 'aps': []}
    }
    
    # Process each dataset
    datasets = ["5_mr_50_cond", "40_mr_50_cond", "100_mr_100_cond"]
    for dataset in datasets:
        dataset_dir = Path(results_dir) / dataset
        if not dataset_dir.exists():
            continue
            
        # Read summary results
        summary_file = dataset_dir / 'summary_results.csv'
        if summary_file.exists():
            try:
                df = pd.read_csv(summary_file)
                
                # Get metrics for all methods
                for method in method_metrics.keys():
                    method_data = df[df['method'] == method]
                    if not method_data.empty:
                        metrics = method_data.iloc[0]
                        method_metrics[method]['aurocs'].append(metrics['roc_auc'])
                        method_metrics[method]['aps'].append(metrics['avg_precision'])
            except Exception as e:
                print(f"Error processing {summary_file}: {str(e)}")
                continue
    
    # Print comparison results
    print("\nComparison of All Methods:")
    print("=" * 60)
    
    # Print metrics for each method
    for method in method_metrics.keys():
        if not method_metrics[method]['aurocs']:
            print(f"\nNo data available for {method.upper()}")
            continue
            
        print(f"\n{method.upper()}:")
        print("-" * 40)
        print(f"Mean ROC AUC: {np.mean(method_metrics[method]['aurocs']):.4f}")
        print(f"Mean Average Precision: {np.mean(method_metrics[method]['aps']):.4f}")
        
    print("\n" + "=" * 60)

def run_analysis(dataset_path, output_dir):
    """
    Run GRN reconstruction analysis on a dataset.
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_dir (str): Path to save results
    """
    try:
        print(f"\nProcessing dataset: {dataset_path}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loader
        print("Initializing data loader...")
        loader = GRNDataLoader(dataset_path)
        
        # Load data
        print("Loading expression matrix...")
        expression_matrix, gene_ids = loader.load_expression_matrix()
        print(f"Loaded expression matrix with shape: {expression_matrix.shape}")
        print(f"Number of genes: {len(gene_ids)}")
        
        print("Loading ground truth matrix...")
        ground_truth_matrix = loader.get_ground_truth_matrix()
        
        # Split into TF and target expressions
        print("Splitting into TF and target expressions...")
        tf_expression, target_expression, tf_ids, target_ids = loader.get_tf_target_split(
            expression_matrix, gene_ids)
        
        # Initialize evaluator
        print("Initializing evaluator...")
        evaluator = GRNEvaluator(ground_truth_matrix)
        
        # Run different methods
        methods = [
            'ensemble'
        ]
        results = {}
        
        for method in methods:
            print(f"\nRunning {method} method...")
            
            # Initialize reconstructor
            reconstructor = GRNReconstructor(method=method)
            
            # Get predictions
            if method == 'ensemble':
                predicted_matrix = reconstructor._ensemble_method(
                tf_expression, target_expression, ground_truth=ground_truth_matrix
                )
            else:
                predicted_matrix = reconstructor.fit(tf_expression, target_expression)
            
            # Evaluate predictions
            metrics = evaluator.evaluate(predicted_matrix)
            results[method] = metrics
            
            # Save results
            results_df = pd.DataFrame({
                'target_id': target_ids,
                'auroc': metrics['target_aurocs']
            })
            results_df.to_csv(output_dir / f'{method}_results.csv', index=False)
            
            # Get top regulators
            top_regulators = evaluator.get_top_k_regulators(predicted_matrix, k=5)
            
            # Save top regulators
            with open(output_dir / f'{method}_top_regulators.txt', 'w') as f:
                for target_idx, regulators in top_regulators:
                    f.write(f"Target {target_ids[target_idx]}:\n")
                    for tf_idx, score in regulators:
                        f.write(f"  TF {tf_ids[tf_idx]}: {score:.4f}\n")
                    f.write("\n")
            
            print(f"Method: {method}")
            print(f"Overall ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"Average Precision: {metrics['avg_precision']:.4f}")
            print(f"Mean Target AUROC: {metrics['mean_target_auroc']:.4f}")
        
        # Save summary results
        summary_df = pd.DataFrame({
            'method': methods,
            'roc_auc': [results[m]['roc_auc'] for m in methods],
            'avg_precision': [results[m]['avg_precision'] for m in methods],
            'mean_target_auroc': [results[m]['mean_target_auroc'] for m in methods]
        })
        summary_df.to_csv(output_dir / 'summary_results.csv', index=False)

        plot_roc_prc_curves(
            ground_truth_matrix, 
            predicted_matrix, 
            output_dir=output_dir, 
            filename_prefix=method
        )
        
    except Exception as e:
        print(f"Error processing dataset {dataset_path}: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    datasets = [
        "5_mr_50_cond",
        "40_mr_50_cond",
        "100_mr_100_cond"
    ]
    
    for dataset in datasets:
        try:
            print(f"\nProcessing dataset: {dataset}")
            run_analysis(
                dataset_path=f"Project1_updated_021125/{dataset}",
                output_dir=f"results/{dataset}"
            )
        except Exception as e:
            print(f"Failed to process dataset {dataset}: {str(e)}")
            continue
    
    # Compare methods after processing all datasets
    compare_methods("results") 