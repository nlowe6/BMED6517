import os
import numpy as np
import pandas as pd
from pathlib import Path
from data_loader import GRNDataLoader
from grn_methods import GRNReconstructor
from evaluation import GRNEvaluator
from visualization import Visualizer
from scipy import stats
import time

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
        print(f"Ground truth matrix shape: {ground_truth_matrix.shape}")
        print(f"Ground truth matrix type: {ground_truth_matrix.dtype}")
        print(f"Ground truth unique values: {np.unique(ground_truth_matrix)}")
        
        # Split into TF and target expressions
        print("Splitting into TF and target expressions...")
        tf_expression, target_expression, tf_ids, target_ids = loader.get_tf_target_split(
            expression_matrix, gene_ids)
        print(f"TF expression shape: {tf_expression.shape}")
        print(f"Target expression shape: {target_expression.shape}")
        print(f"Number of TFs: {len(tf_ids)}")
        print(f"Number of targets: {len(target_ids)}")
        
        # Initialize evaluator
        print("Initializing evaluator...")
        evaluator = GRNEvaluator(ground_truth_matrix)
        
        # Run different methods
        methods = [
            'lasso', 
            'tigress', 
            'ensemble'
        ]
        results = {}
        
        for method in methods:
            print(f"\nRunning {method} method...")
            
            # Initialize reconstructor
            reconstructor = GRNReconstructor(method=method)
            
            # Get predictions
            predicted_matrix = reconstructor.fit(tf_expression, target_expression)
            print(f"Predicted matrix shape: {predicted_matrix.shape}")
            print(f"Predicted matrix type: {predicted_matrix.dtype}")
            print(f"Predicted matrix min: {predicted_matrix.min()}")
            print(f"Predicted matrix max: {predicted_matrix.max()}")
            print(f"Predicted matrix mean: {predicted_matrix.mean()}")
            
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

def main():
    try:
        # Load the data
        data = np.load('5_mr_50_cond.npz')
        
        # Check data format
        print("Available arrays in the file:", list(data.keys()))
        
        # Get expression data
        tf_expression = data['tf_expression']
        target_expression = data['target_expression']
        
        # Ensure data is in correct format (2D arrays)
        if len(tf_expression.shape) != 2 or len(target_expression.shape) != 2:
            raise ValueError("Expression data must be 2D arrays")
            
        # Print data shapes and types
        print(f"\nTF expression shape: {tf_expression.shape}")
        print(f"Target expression shape: {target_expression.shape}")
        print(f"TF expression type: {tf_expression.dtype}")
        print(f"Target expression type: {target_expression.dtype}")
        
        # Initialize reconstructor with TIGRESS method
        reconstructor = GRNReconstructor()
        
        # Run TIGRESS method with timing
        print("\nRunning TIGRESS method...")
        start_time = time.time()
        regulatory_matrix = reconstructor._tigress_method(tf_expression, target_expression)
        end_time = time.time()
        
        # Print results
        print("\nResults:")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Regulatory matrix shape: {regulatory_matrix.shape}")
        print(f"Min value: {regulatory_matrix.min():.4f}")
        print(f"Max value: {regulatory_matrix.max():.4f}")
        print(f"Mean value: {regulatory_matrix.mean():.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
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