import os
import numpy as np
import pandas as pd
from pathlib import Path
from data_loader import GRNDataLoader
from grn_methods import GRNReconstructor
from evaluation import GRNEvaluator
from visualization import Visualizer

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
        print(f"Ground truth matrix shape: {ground_truth_matrix.shape}")
        
        # Split into TF and target expressions
        print("Splitting into TF and target expressions...")
        tf_expression, target_expression, tf_ids, target_ids = loader.get_tf_target_split(
            expression_matrix, gene_ids)
        print(f"TF expression shape: {tf_expression.shape}")
        print(f"Target expression shape: {target_expression.shape}")
        
        # Initialize evaluator
        print("Initializing evaluator...")
        evaluator = GRNEvaluator(ground_truth_matrix)
        
        # Run different methods
        methods = [
            'correlation', 
            'mutual_info', 
            'lasso', 
            'genie3', 
            'aracne', 
            'clr', 
            'tigress', 
            'wgcna',
            'dnn'
        ]
        results = {}
        
        for method in methods:
            print(f"\nRunning {method} method...")
            
            # Initialize reconstructor
            reconstructor = GRNReconstructor(method=method)
            
            # Get predictions
            predicted_matrix = reconstructor.fit(tf_expression, target_expression)
            print(f"Predicted matrix shape: {predicted_matrix.shape}")
            
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
        
        # Visualize results
        visualizer = Visualizer(output_dir)
        visualizer.plot_roc_curves()
        visualizer.plot_precision_recall_curves()
        visualizer.plot_target_aurocs()
        
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