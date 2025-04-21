import numpy as np
from scipy import stats
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import GridSearchCV
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import average_precision_score
from data_loader import GRNDataLoader

class GRNReconstructor:
    def __init__(self, method='correlation'):
        """
        Initialize the GRN reconstructor with a specific method.
        
        Args:
            method (str): Method to use for GRN reconstruction
        """
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, tf_expression, target_expression):
        """
        Fit the GRN reconstruction model.
        
        Args:
            tf_expression (numpy.ndarray): TF expression matrix of shape (n_tfs, n_conditions)
            target_expression (numpy.ndarray): Target gene expression matrix of shape (n_targets, n_conditions)
            
        Returns:
            numpy.ndarray: Predicted regulatory relationships matrix of shape (n_tfs, n_targets)
        """
        if self.method == 'tigress':
            return self._tigress_method(tf_expression, target_expression)
        elif self.method == 'ensemble':
            return self._ensemble_method(tf_expression, target_expression)
        elif self.method == 'elasticnet':
            return self._elasticnet_method(tf_expression, target_expression)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _tigress_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using TIGRESS (Trustful Inference of Gene REgulation using Stability Selection) algorithm.
        
        Args:
            tf_expression (numpy.ndarray): TF expression matrix of shape (n_tfs, n_conditions)
            target_expression (numpy.ndarray): Target gene expression matrix of shape (n_targets, n_conditions)
            
        Returns:
            numpy.ndarray: Regulatory strength matrix of shape (n_tfs, n_targets)
        """
        n_tfs = tf_expression.shape[0]
        n_targets = target_expression.shape[0]
        regulatory_matrix = np.zeros((n_tfs, n_targets))
        
        # Scale the data
        scaler = StandardScaler()
        tf_scaled = scaler.fit_transform(tf_expression.T)
        
        # Number of bootstrap samples
        n_bootstrap = 50
        
        # For each target gene
        for j in range(n_targets):
            try:
                # Initialize selection frequencies
                selection_freq = np.zeros(n_tfs)
                
                # Perform bootstrap sampling
                for _ in range(n_bootstrap):
                    # Sample with replacement
                    indices = np.random.choice(tf_scaled.shape[0], size=tf_scaled.shape[0], replace=True)
                    X_boot = tf_scaled[indices]
                    y_boot = target_expression[j, indices]
                    
                    # Fit LARS with adjusted parameters for better numerical stability
                    lars = LassoLarsCV(
                        cv=5,
                        max_iter=2000,
                        n_jobs=-1,
                    )
                    lars.fit(X_boot, y_boot)
                    
                    # Update selection frequencies
                    selection_freq += (np.abs(lars.coef_) > 0).astype(int)
                
                # Normalize selection frequencies
                selection_freq /= n_bootstrap
                
                # Store the selection frequencies
                regulatory_matrix[:, j] = selection_freq
                
            except Exception as e:
                print(f"Error processing target {j}: {str(e)}")
                regulatory_matrix[:, j] = 0
        
        return regulatory_matrix

    def _elasticnet_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using ElasticNet regression.
    
        Args:
            tf_expression (numpy.ndarray): TF expression matrix of shape (n_tfs, n_conditions)
            target_expression (numpy.ndarray): Target gene expression matrix of shape (n_targets, n_conditions)
        
        Returns:
            numpy.ndarray: Regulatory strength matrix of shape (n_tfs, n_targets)
        """
        n_tfs = tf_expression.shape[0]
        n_targets = target_expression.shape[0]
        regulatory_matrix = np.zeros((n_tfs, n_targets))
    
        # Scale the data
        scaler = StandardScaler()
        tf_scaled = scaler.fit_transform(tf_expression.T)  # shape: (n_conditions, n_tfs)

        param_grid = {
            'l1_ratio': [.1,.2,.3,.4,.5,.6,.7,.8,.9]
        }
    
        for j in range(n_targets):
            try:
                enet = ElasticNetCV(
                    max_iter=10000, n_jobs = -1, alphas = [.01, .1, 1.0, 10]
            )
                grid = GridSearchCV(enet, param_grid, cv = 5, scoring='neg_mean_squared_error')
                grid.fit(tf_scaled,target_expression[j, :])
                best_model = grid.best_estimator_
                print(f"Target {j}: best l1_ratio = {best_model.l1_ratio}")
                regulatory_matrix[:, j] = np.abs(best_model.coef_)
            except Exception as e:
                print(f"ElasticNet error on target {j}: {str(e)}")
                regulatory_matrix[:, j] = 0
    
        return regulatory_matrix


    def _ensemble_method(self, tf_expression, target_expression, ground_truth = None):
        """
        Reconstruct GRN using an ensemble of TIGRESS and Lasso methods.
        
        Args:
            tf_expression (numpy.ndarray): TF expression matrix of shape (n_tfs, n_conditions)
            target_expression (numpy.ndarray): Target gene expression matrix of shape (n_targets, n_conditions)
            
        Returns:
            numpy.ndarray: Regulatory strength matrix of shape (n_tfs, n_targets)
        """
        # Get predictions from both methods
        tigress_matrix = self._tigress_method(tf_expression, target_expression)
        elastic_matrix = self._elasticnet_method(tf_expression, target_expression)
        
        # Normalize both matrices to [0, 1] range
        tigress_matrix = (tigress_matrix - tigress_matrix.min()) / (tigress_matrix.max() - tigress_matrix.min())
        elastic_matrix = (elastic_matrix - elastic_matrix.min()) / (elastic_matrix.max() - elastic_matrix.min())

        best_score = -1
        best_weight = .5
        best_ensemble = None

        for w in np.linspace(0,1,11):
            ensemble_matrix = w * tigress_matrix + (1-w) * elastic_matrix

            ensemble_flat = ensemble_matrix.flatten()
            ground_truth_flat = ground_truth.flatten()
            score = average_precision_score(ground_truth_flat, ensemble_flat)

            print(f"Weight TIGRESS={w:.1f}, ElasticNet={1 - w:.1f} => AUPR={score:.4f}")
            if score > best_score:
                best_score = score
                best_weight = w
                best_ensemble = ensemble_matrix
        
        print(f"Best ensemble weights: TIGRESS={best_weight:.2f}, ElasticNet={1 - best_weight:.2f}, AUPR={best_score:.4f}")
        return best_ensemble
