import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class GRNEvaluator:
    def __init__(self, ground_truth_matrix):
        """
        Initialize the evaluator with ground truth matrix.
        
        Args:
            ground_truth_matrix (numpy.ndarray): Binary matrix of true regulatory relationships
        """
        # Ensure ground truth is binary (0 or 1)
        self.ground_truth = (ground_truth_matrix > 0).astype(int)
        
        # Print ground truth statistics
        print(f"Ground truth shape: {self.ground_truth.shape}")
        print(f"Number of positive samples: {np.sum(self.ground_truth)}")
        print(f"Number of negative samples: {np.sum(1 - self.ground_truth)}")
        print(f"Ground truth unique values: {np.unique(self.ground_truth)}")
        
    def evaluate(self, predicted_matrix):
        """
        Evaluate the predicted GRN against ground truth.
        
        Args:
            predicted_matrix (numpy.ndarray): Matrix of predicted regulatory relationships
            
        Returns:
            dict: Dictionary containing various evaluation metrics
        """
        try:
            # Check matrix shapes
            if predicted_matrix.shape != self.ground_truth.shape:
                raise ValueError(f"Shape mismatch: predicted_matrix {predicted_matrix.shape} != ground_truth {self.ground_truth.shape}")
            
            # Flatten matrices for evaluation
            y_true = self.ground_truth.flatten()
            y_score = predicted_matrix.flatten()
            
            # Check if we have both positive and negative samples
            n_pos = np.sum(y_true)
            n_neg = len(y_true) - n_pos
            
            if n_pos == 0 or n_neg == 0:
                print(f"Warning: Ground truth contains only one class (pos: {n_pos}, neg: {n_neg})")
                return {
                    'roc_auc': 0.5,
                    'avg_precision': 0.0,
                    'target_aurocs': np.zeros(predicted_matrix.shape[1]),
                    'mean_target_auroc': 0.5
                }
            
            # Calculate overall ROC AUC
            try:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
            except Exception as e:
                print(f"Error calculating overall ROC AUC: {str(e)}")
                roc_auc = 0.5
            
            # Calculate average precision
            try:
                avg_precision = average_precision_score(y_true, y_score)
            except Exception as e:
                print(f"Error calculating average precision: {str(e)}")
                avg_precision = 0.0
            
            # Calculate per-target AUROCs
            target_aurocs = self._calculate_per_target_aurocs(predicted_matrix)
            mean_target_auroc = np.mean(target_aurocs)
            
            return {
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'target_aurocs': target_aurocs,
                'mean_target_auroc': mean_target_auroc
            }
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return {
                'roc_auc': 0.5,
                'avg_precision': 0.0,
                'target_aurocs': np.zeros(predicted_matrix.shape[1]),
                'mean_target_auroc': 0.5
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
            try:
                y_true = self.ground_truth[:, j]
                y_score = predicted_matrix[:, j]
                
                # Skip if all samples are of the same class
                n_pos = np.sum(y_true)
                n_neg = len(y_true) - n_pos
                
                if n_pos == 0 or n_neg == 0:
                    print(f"Warning: Target {j} has only one class (pos: {n_pos}, neg: {n_neg})")
                    target_aurocs[j] = 0.5
                    continue
                
                # Ensure y_true is binary
                y_true = (y_true > 0).astype(int)
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_score)
                target_aurocs[j] = auc(fpr, tpr)
                
            except Exception as e:
                print(f"Error calculating AUROC for target {j}: {str(e)}")
                print(f"y_true unique values: {np.unique(y_true)}")
                print(f"y_score min/max: {np.min(y_score)}, {np.max(y_score)}")
                target_aurocs[j] = 0.5
        
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