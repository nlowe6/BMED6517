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

class DNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super(DNNRegressor, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class GRNReconstructor:
    def __init__(self, method='correlation'):
        """
        Initialize the GRN reconstructor with a specific method.
        
        Args:
            method (str): Method to use for GRN reconstruction
                         Options: 'correlation', 'mutual_info', 'lasso', 'genie3', 
                                 'aracne', 'clr', 'tigress', 'wgcna', 'dnn'
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
        elif self.method == 'mutual_info':
            return self._mutual_information_method(tf_expression, target_expression)
        elif self.method == 'lasso':
            return self._lasso_method(tf_expression, target_expression)
        elif self.method == 'genie3':
            return self._genie3_method(tf_expression, target_expression)
        elif self.method == 'aracne':
            return self._aracne_method(tf_expression, target_expression)
        elif self.method == 'clr':
            return self._clr_method(tf_expression, target_expression)
        elif self.method == 'tigress':
            return self._tigress_method(tf_expression, target_expression)
        elif self.method == 'wgcna':
            return self._wgcna_method(tf_expression, target_expression)
        elif self.method == 'dnn':
            return self._dnn_method(tf_expression, target_expression)
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
        n_tfs = tf_expression.shape[0]  # Number of TFs
        n_targets = target_expression.shape[0]  # Number of target genes
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
    
    def _mutual_information_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using mutual information.
        
        Args:
            tf_expression (numpy.ndarray): TF expression matrix of shape (n_tfs, n_conditions)
            target_expression (numpy.ndarray): Target gene expression matrix of shape (n_targets, n_conditions)
            
        Returns:
            numpy.ndarray: Mutual information matrix of shape (n_tfs, n_targets)
        """
        n_tfs = tf_expression.shape[0]  # Number of TFs
        n_targets = target_expression.shape[0]  # Number of target genes
        mi_matrix = np.zeros((n_tfs, n_targets))
        
        # Calculate mutual information for each TF-target pair
        for i in range(n_tfs):
            for j in range(n_targets):
                try:
                    mi_matrix[i, j] = self._calculate_mutual_information(
                        tf_expression[i, :], target_expression[j, :])
                except:
                    mi_matrix[i, j] = 0
        
        return mi_matrix
    
    def _calculate_mutual_information(self, x, y, bins=10):
        """
        Calculate mutual information between two variables.
        
        Args:
            x (numpy.ndarray): First variable
            y (numpy.ndarray): Second variable
            bins (int): Number of bins for discretization
            
        Returns:
            float: Mutual information value
        """
        # Remove any NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        if len(x) == 0 or len(y) == 0:
            return 0
        
        # Calculate joint probability
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = hist_2d / float(len(x))
        
        # Calculate marginal probabilities
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Calculate mutual information
        mi = 0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
        
        return mi
    
    def _lasso_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using Lasso regression.
        
        Args:
            tf_expression (numpy.ndarray): TF expression matrix of shape (n_tfs, n_conditions)
            target_expression (numpy.ndarray): Target gene expression matrix of shape (n_targets, n_conditions)
            
        Returns:
            numpy.ndarray: Regulatory strength matrix of shape (n_tfs, n_targets)
        """
        n_tfs = tf_expression.shape[0]  # Number of TFs
        n_targets = target_expression.shape[0]  # Number of target genes
        regulatory_matrix = np.zeros((n_tfs, n_targets))
        
        # Scale the data
        scaler = StandardScaler()
        tf_scaled = scaler.fit_transform(tf_expression.T)  # Transpose to get (n_conditions, n_tfs)
        
        for j in range(n_targets):
            try:
                # Fit Lasso regression for each target gene
                lasso = LassoCV(cv=5, random_state=42)
                lasso.fit(tf_scaled, target_expression[j, :])  # Use j-th target gene expression
                
                # Store coefficients
                regulatory_matrix[:, j] = np.abs(lasso.coef_)
            except:
                regulatory_matrix[:, j] = 0
        
        return regulatory_matrix
    
    def _genie3_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using GENIE3 algorithm.
        
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
                # Train a Random Forest model
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42
                )
                
                # Fit the model
                rf.fit(tf_scaled, target_expression[j, :])
                
                # Get feature importances
                importances = rf.feature_importances_
                
                # Store the importances
                regulatory_matrix[:, j] = importances
                
            except Exception as e:
                print(f"Error processing target {j}: {str(e)}")
                regulatory_matrix[:, j] = 0
        
        return regulatory_matrix
    
    def _aracne_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using ARACNE algorithm.
        
        Args:
            tf_expression (numpy.ndarray): TF expression matrix of shape (n_tfs, n_conditions)
            target_expression (numpy.ndarray): Target gene expression matrix of shape (n_targets, n_conditions)
            
        Returns:
            numpy.ndarray: Regulatory strength matrix of shape (n_tfs, n_targets)
        """
        n_tfs = tf_expression.shape[0]
        n_targets = target_expression.shape[0]
        regulatory_matrix = np.zeros((n_tfs, n_targets))
        
        # Calculate mutual information matrix
        mi_matrix = np.zeros((n_tfs + n_targets, n_tfs + n_targets))
        all_expression = np.vstack((tf_expression, target_expression))
        
        # Calculate pairwise mutual information
        for i in range(n_tfs + n_targets):
            for j in range(i+1, n_tfs + n_targets):
                mi = self._calculate_mutual_information(all_expression[i, :], all_expression[j, :])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        # Apply Data Processing Inequality (DPI)
        for i in range(n_tfs):
            for j in range(n_tfs, n_tfs + n_targets):
                for k in range(n_tfs + n_targets):
                    if k != i and k != j:
                        # If I(i,j) < min(I(i,k), I(k,j)), remove edge (i,j)
                        if mi_matrix[i, j] < min(mi_matrix[i, k], mi_matrix[k, j]):
                            mi_matrix[i, j] = 0
                            mi_matrix[j, i] = 0
        
        # Extract TF-target relationships
        regulatory_matrix = mi_matrix[:n_tfs, n_tfs:]
        
        return regulatory_matrix
    
    def _clr_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using CLR (Context Likelihood of Relatedness) algorithm.
        
        Args:
            tf_expression (numpy.ndarray): TF expression matrix of shape (n_tfs, n_conditions)
            target_expression (numpy.ndarray): Target gene expression matrix of shape (n_targets, n_conditions)
            
        Returns:
            numpy.ndarray: Regulatory strength matrix of shape (n_tfs, n_targets)
        """
        n_tfs = tf_expression.shape[0]
        n_targets = target_expression.shape[0]
        regulatory_matrix = np.zeros((n_tfs, n_targets))
        
        # Calculate mutual information matrix
        mi_matrix = np.zeros((n_tfs + n_targets, n_tfs + n_targets))
        all_expression = np.vstack((tf_expression, target_expression))
        
        # Calculate pairwise mutual information
        for i in range(n_tfs + n_targets):
            for j in range(i+1, n_tfs + n_targets):
                mi = self._calculate_mutual_information(all_expression[i, :], all_expression[j, :])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        # Calculate z-scores for each gene
        z_scores = np.zeros_like(mi_matrix)
        for i in range(n_tfs + n_targets):
            mi_values = mi_matrix[i, :]
            mean_mi = np.mean(mi_values)
            std_mi = np.std(mi_values)
            if std_mi > 0:
                z_scores[i, :] = (mi_values - mean_mi) / std_mi
        
        # Calculate CLR scores
        clr_matrix = np.zeros_like(mi_matrix)
        for i in range(n_tfs + n_targets):
            for j in range(n_tfs + n_targets):
                if i != j:
                    clr_matrix[i, j] = np.sqrt(z_scores[i, j]**2 + z_scores[j, i]**2)
        
        # Extract TF-target relationships
        regulatory_matrix = clr_matrix[:n_tfs, n_tfs:]
        
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
        n_bootstrap = 100
        
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
                    
                    # Fit LARS
                    lars = LassoLarsCV(cv=5)
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
    
    def _wgcna_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using WGCNA (Weighted Gene Co-expression Network Analysis) algorithm.
        
        Args:
            tf_expression (numpy.ndarray): TF expression matrix of shape (n_tfs, n_conditions)
            target_expression (numpy.ndarray): Target gene expression matrix of shape (n_targets, n_conditions)
            
        Returns:
            numpy.ndarray: Regulatory strength matrix of shape (n_tfs, n_targets)
        """
        n_tfs = tf_expression.shape[0]
        n_targets = target_expression.shape[0]
        regulatory_matrix = np.zeros((n_tfs, n_targets))
        
        # Combine TF and target expression
        all_expression = np.vstack((tf_expression, target_expression))
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(all_expression)
        
        # Apply soft thresholding (power transformation)
        power = 6  # Soft thresholding power
        adj_matrix = np.abs(corr_matrix) ** power
        
        # Set diagonal to 0
        np.fill_diagonal(adj_matrix, 0)
        
        # Calculate topological overlap
        topo_overlap = np.zeros_like(adj_matrix)
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if i != j:
                    # Calculate topological overlap
                    a_ij = adj_matrix[i, j]
                    a_ik = adj_matrix[i, :]
                    a_kj = adj_matrix[:, j]
                    l_ij = np.sum(np.minimum(a_ik, a_kj)) - a_ij
                    k_i = np.sum(a_ik) - a_ij
                    k_j = np.sum(a_kj) - a_ij
                    w_ij = min(k_i, k_j) + 1 - a_ij
                    topo_overlap[i, j] = (l_ij + a_ij) / w_ij
        
        # Extract TF-target relationships
        regulatory_matrix = topo_overlap[:n_tfs, n_tfs:]
        
        return regulatory_matrix
    
    def _dnn_method(self, tf_expression, target_expression):
        """
        Reconstruct GRN using Deep Neural Network with GPU acceleration.
        
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
        
        # Convert to PyTorch tensors and move to GPU
        X = torch.FloatTensor(tf_scaled).to(self.device)
        
        # For each target gene
        for j in range(n_targets):
            try:
                # Prepare target data and move to GPU
                y = torch.FloatTensor(target_expression[j, :]).to(self.device)
                
                # Create dataset and dataloader with pin_memory for faster GPU transfer
                dataset = TensorDataset(X, y)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=64,  # Increased batch size for better GPU utilization
                    shuffle=True,
                    pin_memory=True,  # Faster data transfer to GPU
                    num_workers=4  # Parallel data loading
                )
                
                # Initialize model and move to GPU
                model = DNNRegressor(input_dim=n_tfs).to(self.device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Enable cuDNN benchmarking for faster training
                torch.backends.cudnn.benchmark = True
                
                # Train model
                model.train()
                for epoch in range(100):
                    for batch_X, batch_y in dataloader:
                        optimizer.zero_grad()
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                # Get feature importance using integrated gradients
                model.eval()
                with torch.no_grad():  # Disable gradient computation for evaluation
                    baseline = torch.zeros_like(X[0]).to(self.device)
                    integrated_grads = torch.zeros_like(X[0]).to(self.device)
                    
                    # Process TFs in batches for better GPU utilization
                    batch_size = 32
                    for i in range(0, n_tfs, batch_size):
                        end_idx = min(i + batch_size, n_tfs)
                        batch_inputs = torch.zeros((end_idx - i, X.shape[1]), device=self.device)
                        batch_inputs[:, i:end_idx] = X[0, i:end_idx]
                        
                        # Calculate integrated gradients for batch
                        alpha = torch.linspace(0, 1, 50, device=self.device)
                        for a in alpha:
                            interpolated = baseline + a * (batch_inputs - baseline)
                            interpolated.requires_grad_(True)
                            output = model(interpolated)
                            output.backward(torch.ones_like(output))
                            integrated_grads[i:end_idx] += interpolated.grad[:, i:end_idx].sum(dim=0)
                        
                        integrated_grads[i:end_idx] *= (batch_inputs[:, i:end_idx] - baseline[i:end_idx]) / len(alpha)
                
                # Store the importance scores
                regulatory_matrix[:, j] = integrated_grads.cpu().numpy()
                
            except Exception as e:
                print(f"Error processing target {j}: {str(e)}")
                regulatory_matrix[:, j] = 0
        
        return regulatory_matrix 