import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class GRNEvaluator:
    def __init__(self, ground_truth_matrix):
        """
        Initialize the evaluator with ground truth matrix.
        
        Args:
            ground_truth_matrix (numpy.ndarray): Binary matrix of true regulatory relationships
        """
        self.ground_truth = ground_truth_matrix
        
    def evaluate(self, predicted_matrix):
        """
        Evaluate the predicted GRN against ground truth.
        
        Args:
            predicted_matrix (numpy.ndarray): Matrix of predicted regulatory relationships
            
        Returns:
            dict: Dictionary containing various evaluation metrics
        """
        # Flatten matrices for evaluation
        y_true = self.ground_truth.flatten()
        y_score = predicted_matrix.flatten()
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Calculate PR curve and average precision
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        
        # Calculate per-target gene AUROC
        target_aurocs = self._calculate_per_target_aurocs(predicted_matrix)
        
        return {
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'target_aurocs': target_aurocs,
            'mean_target_auroc': np.mean(target_aurocs)
        }
    
    def _calculate_per_target_aurocs(self, predicted_matrix):
        """
        Calculate AUROC for each target gene separately.
        
        Args:
            predicted_matrix (numpy.ndarray): Matrix of predicted regulatory relationships
            
        Returns:
            numpy.ndarray: AUROC values for each target gene
        """
        n_targets = predicted_matrix.shape[1]
        target_aurocs = np.zeros(n_targets)
        
        for j in range(n_targets):
            y_true = self.ground_truth[:, j]
            y_score = predicted_matrix[:, j]
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            target_aurocs[j] = auc(fpr, tpr)
        
        return target_aurocs
    
    def get_top_k_regulators(self, predicted_matrix, k=5):
        """
        Get top k regulators for each target gene.
        
        Args:
            predicted_matrix (numpy.ndarray): Matrix of predicted regulatory relationships
            k (int): Number of top regulators to return
            
        Returns:
            list: List of tuples (target_idx, [(tf_idx, score), ...])
        """
        n_targets = predicted_matrix.shape[1]
        top_regulators = []
        
        for j in range(n_targets):
            # Get scores for this target
            scores = predicted_matrix[:, j]
            
            # Get indices of top k regulators
            top_k_indices = np.argsort(scores)[-k:][::-1]
            top_k_scores = scores[top_k_indices]
            
            # Store as tuples of (tf_idx, score)
            regulators = list(zip(top_k_indices, top_k_scores))
            top_regulators.append((j, regulators))
        
        return top_regulators 