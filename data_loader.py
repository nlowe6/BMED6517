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
        df = pd.read_csv(expr_file, sep='\t', header=0)
        
        # Convert to numpy array and get gene IDs
        expression_matrix = df.values.T  # Transpose to get (n_genes, n_conditions)
        gene_ids = [str(i) for i in range(len(expression_matrix))]  # Generate gene IDs
        
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
        
        # Get unique TF and target IDs
        tf_ids = sorted(df[tf_col].unique())
        target_ids = sorted(df[target_col].unique())
        
        return tf_ids, target_ids
    
    def get_ground_truth_matrix(self):
        """
        Create a binary matrix representing the ground truth GRN.
        
        Returns:
            numpy.ndarray: Binary matrix of shape (n_tfs, n_targets)
            where 1 indicates a regulatory relationship
        """
        tf_ids, target_ids = self.load_ground_truth()
        grn_file = self.dataset_path / "bipartite_GRN.csv"
        
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
        
        # Create binary matrix
        ground_truth = np.zeros((len(tf_ids), len(target_ids)))
        for _, row in df.iterrows():
            tf_idx = tf_ids.index(row[tf_col])
            target_idx = target_ids.index(row[target_col])
            ground_truth[tf_idx, target_idx] = 1
            
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
        
        # Convert gene IDs to strings for comparison
        gene_ids = [str(g) for g in gene_ids]
        tf_ids = [str(t) for t in tf_ids]
        target_ids = [str(t) for t in target_ids]
        
        # Get indices for TFs and targets
        tf_indices = [gene_ids.index(tf) for tf in tf_ids if tf in gene_ids]
        target_indices = [gene_ids.index(tgt) for tgt in target_ids if tgt in gene_ids]
        
        # Split expression matrix
        tf_expression = expression_matrix[tf_indices, :]  # Changed to use correct indexing
        target_expression = expression_matrix[target_indices, :]  # Changed to use correct indexing
        
        # Get corresponding gene IDs
        tf_gene_ids = [gene_ids[i] for i in tf_indices]
        target_gene_ids = [gene_ids[i] for i in target_indices]
        
        return tf_expression, target_expression, tf_gene_ids, target_gene_ids 