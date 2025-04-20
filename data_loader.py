import pandas as pd
import numpy as np
from pathlib import Path

class GRNDataLoader:
    def __init__(self, dataset_path):
        """
        Initialize the data loader with the path to the dataset.
        
        Args:
            dataset_path (str): Path to the dataset directory
        """
        self.dataset_path = Path(dataset_path)
        
    def load_expression_matrix(self):
        """
        Load the expression matrix from the simulated_noNoise.txt file.
        
        Returns:
            tuple: (expression_matrix, gene_ids)
            - expression_matrix: numpy array of shape (n_genes, n_conditions)
            - gene_ids: list of gene IDs
        """
        expr_file = self.dataset_path / "simulated_noNoise.txt"
        print(f"\nLoading expression matrix from: {expr_file}")
        
        df = pd.read_csv(expr_file, sep='\t', header=0)
        print(f"Raw expression matrix shape: {df.shape}")
        
        # Convert to numpy array and get gene IDs
        expression_matrix = df.values.T  # Transpose to get (n_genes, n_conditions)
        gene_ids = [str(i) for i in range(len(expression_matrix))]
        
        print(f"Processed expression matrix shape: {expression_matrix.shape}")
        print(f"Number of genes: {len(gene_ids)}")
        print(f"Sample gene IDs: {gene_ids[:5]}")
        
        return expression_matrix, gene_ids
    
    def load_ground_truth(self):
        """
        Load the ground truth GRN from bipartite_GRN.csv.
        
        Returns:
            tuple: (tf_ids, target_ids)
            - tf_ids: list of transcription factor IDs
            - target_ids: list of target gene IDs
        """
        grn_file = self.dataset_path / "bipartite_GRN.csv"
        print(f"\nLoading ground truth from: {grn_file}")
        
        # First try to read the first line to check for headers
        with open(grn_file, 'r') as f:
            first_line = f.readline().strip()
        
        # Check if the first line contains headers
        if first_line.lower() in ['tf_id,target_gene_id', 'tf_id,target_id']:
            # File has headers
            df = pd.read_csv(grn_file)
            if 'TF_id' in df.columns and 'Target_gene_id' in df.columns:
                tf_col = 'TF_id'
                target_col = 'Target_gene_id'
            else:
                tf_col = df.columns[0]
                target_col = df.columns[1]
        else:
            # File has no headers
            df = pd.read_csv(grn_file, header=None, names=['tf_id', 'target_id'])
            tf_col = 'tf_id'
            target_col = 'target_id'
        
        print(f"Ground truth data shape: {df.shape}")
        print(f"Sample ground truth entries:")
        print(df.head())
        
        # Get unique TF and target IDs
        tf_ids = sorted(df[tf_col].unique())
        target_ids = sorted(df[target_col].unique())
        
        print(f"Number of unique TFs: {len(tf_ids)}")
        print(f"Number of unique targets: {len(target_ids)}")
        print(f"Sample TF IDs: {tf_ids[:5]}")
        print(f"Sample target IDs: {target_ids[:5]}")
        
        return tf_ids, target_ids
    
    def get_ground_truth_matrix(self):
        """
        Create a binary matrix representing the ground truth GRN.
        
        Returns:
            numpy.ndarray: Binary matrix of shape (n_tfs, n_targets)
            where 1 indicates a regulatory relationship
        """
        print("\nCreating ground truth matrix...")
        tf_ids, target_ids = self.load_ground_truth()
        
        # Convert IDs to integers and get unique sorted lists
        tf_ids = sorted([int(tf) for tf in tf_ids])
        target_ids = sorted([int(tgt) for tgt in target_ids])
        
        print(f"Number of TFs in matrix: {len(tf_ids)}")
        print(f"Number of targets in matrix: {len(target_ids)}")
        
        # Create binary matrix with correct dimensions
        ground_truth = np.zeros((len(tf_ids), len(target_ids)), dtype=int)
        
        # Create mapping from IDs to indices
        tf_idx_map = {tf: idx for idx, tf in enumerate(tf_ids)}
        target_idx_map = {tgt: idx for idx, tgt in enumerate(target_ids)}
        
        # Fill the matrix
        grn_file = self.dataset_path / "bipartite_GRN.csv"
        df = pd.read_csv(grn_file)
        
        # Determine column names
        if 'TF_id' in df.columns and 'Target_gene_id' in df.columns:
            tf_col = 'TF_id'
            target_col = 'Target_gene_id'
        else:
            tf_col = df.columns[0]
            target_col = df.columns[1]
        
        n_relationships = 0
        for _, row in df.iterrows():
            try:
                tf_idx = tf_idx_map[int(row[tf_col])]
                target_idx = target_idx_map[int(row[target_col])]
                ground_truth[tf_idx, target_idx] = 1
                n_relationships += 1
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid entry: {row[tf_col]}, {row[target_col]}")
                continue
        
        print(f"Number of regulatory relationships: {n_relationships}")
        print(f"Ground truth matrix shape: {ground_truth.shape}")
        print(f"Number of positive entries: {np.sum(ground_truth)}")
        print(f"Number of negative entries: {np.sum(1 - ground_truth)}")
        
        return ground_truth
    
    def get_tf_target_split(self, expression_matrix, gene_ids):
        """
        Split the expression matrix into TF and target gene expressions.
        
        Args:
            expression_matrix (numpy.ndarray): Expression matrix
            gene_ids (list): List of gene IDs
            
        Returns:
            tuple: (tf_expression, target_expression, tf_ids, target_ids)
        """
        # Get TF and target IDs from ground truth
        tf_ids, target_ids = self.load_ground_truth()
        
        # Convert all IDs to integers for consistent comparison
        gene_ids = [int(g) for g in gene_ids]
        tf_ids = [int(t) for t in tf_ids]
        target_ids = [int(t) for t in target_ids]
        
        # Create mapping from gene IDs to indices
        gene_idx_map = {gid: idx for idx, gid in enumerate(gene_ids)}
        
        # Get indices for TFs and targets
        tf_indices = []
        target_indices = []
        
        # Get TF indices
        for tf in tf_ids:
            if tf in gene_idx_map:
                tf_indices.append(gene_idx_map[tf])
        
        # Get target indices
        for tgt in target_ids:
            if tgt in gene_idx_map:
                target_indices.append(gene_idx_map[tgt])
        
        if not tf_indices or not target_indices:
            raise ValueError("No valid TF or target indices found. Check if gene IDs match between expression matrix and ground truth.")
        
        # Sort indices to maintain order
        tf_indices.sort()
        target_indices.sort()
        
        # Split expression matrix
        tf_expression = expression_matrix[tf_indices, :]
        target_expression = expression_matrix[target_indices, :]
        
        # Get corresponding gene IDs
        tf_gene_ids = [gene_ids[i] for i in tf_indices]
        target_gene_ids = [gene_ids[i] for i in target_indices]
        
        print(f"Number of TFs: {len(tf_indices)}")
        print(f"Number of targets: {len(target_indices)}")
        print(f"TF expression shape: {tf_expression.shape}")
        print(f"Target expression shape: {target_expression.shape}")
        
        return tf_expression, target_expression, tf_gene_ids, target_gene_ids 