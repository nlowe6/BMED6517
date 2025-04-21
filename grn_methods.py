import numpy as np
from scipy import stats
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data, Batch
from sklearn.linear_model import ElasticNet

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
        if self.method == 'correlation':
            return self._correlation_method(tf_expression, target_expression)
        elif self.method == 'lasso':
            return self._lasso_method(tf_expression, target_expression)
        elif self.method == 'tigress':
            return self._tigress_method(tf_expression, target_expression)
        elif self.method == 'ensemble':
            return self._ensemble_method(tf_expression, target_expression)
        elif self.method == 'elasticnet':
            return self._elasticnet_method(tf_expression, target_expression)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _correlation_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using Pearson correlation.
        
        Args:
            tf_expression (numpy.ndarray): TF expression matrix of shape (n_tfs, n_conditions)
            target_expression (numpy.ndarray): Target gene expression matrix of shape (n_targets, n_conditions)
            
        Returns:
            numpy.ndarray: Correlation matrix of shape (n_tfs, n_targets)
        """
        n_tfs = tf_expression.shape[0]
        n_targets = target_expression.shape[0]
        correlation_matrix = np.zeros((n_tfs, n_targets))
        
        # Calculate correlation matrix using numpy's corrcoef
        for i in range(n_tfs):
            for j in range(n_targets):
                try:
                    corr = np.corrcoef(tf_expression[i, :], target_expression[j, :])[0, 1]
                    correlation_matrix[i, j] = np.abs(corr) if not np.isnan(corr) else 0
                except:
                    correlation_matrix[i, j] = 0
        
        return correlation_matrix
    
    def _lasso_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using Lasso regression.
        
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
        tf_scaled = scaler.fit_transform(tf_expression.T)  # Transpose to get (n_conditions, n_tfs)
        
        # For each target gene
        for j in range(n_targets):
            try:
                # Fit Lasso regression with increased iterations and adjusted parameters
                lasso = LassoCV(
                    cv=10,              # More folds for better generalization
                    max_iter=10000,     # Allow convergence for harder problems
                    tol=1e-5,           # Higher precision
                    n_alphas=200,       # More fine-grained alpha grid
                    alphas=np.logspace(-4, 0.5, 200),  # Custom alpha range,
                    n_jobs=-1           # Use all available cores
                )

                lasso.fit(tf_scaled, target_expression[j, :])
                
                # Store coefficients
                regulatory_matrix[:, j] = np.abs(lasso.coef_)
            except Exception as e:
                print(f"Error processing target {j}: {str(e)}")
                regulatory_matrix[:, j] = 0
        
        return regulatory_matrix
    
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
    
        for j in range(n_targets):
            try:
                enet = ElasticNet(
                    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],  # mix of Ridge and Lasso
                    max_iter=10000,
            )
                enet.fit(tf_scaled, target_expression[j, :])
                regulatory_matrix[:, j] = np.abs(enet.coef_)
            except Exception as e:
                print(f"ElasticNet error on target {j}: {str(e)}")
                regulatory_matrix[:, j] = 0
    
        return regulatory_matrix


    def _ensemble_method(self, tf_expression, target_expression):
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
        
        # Combine predictions with equal weights
        ensemble_matrix = 0.5 * tigress_matrix + 0.5 * elastic_matrix
        
        return ensemble_matrix 