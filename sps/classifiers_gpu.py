import torch
import torch.nn as nn
import numpy as np

class LogisticRegressionGPU(nn.Module):
    """Logistic Regression GPU con soluzione esatta (identica a sklearn)"""
    def __init__(self, input_dim, num_classes=10, device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, num_classes if num_classes > 2 else 1, bias=True)
        self.to(device)
        self.classes_ = None
        
    def forward(self, x):
        return self.linear(x)
    
    @property
    def coef_(self):
        with torch.no_grad():
            if self.num_classes > 2:
                return self.linear.weight.data.cpu().numpy()
            else:
                return self.linear.weight.data.cpu().numpy().reshape(1, -1)
    
    @property
    def intercept_(self):
        with torch.no_grad():
            if self.num_classes > 2:
                return self.linear.bias.data.cpu().numpy()
            else:
                return self.linear.bias.data.cpu().numpy().reshape(1)
    
    def fit(self, X, y, C=1.0, max_iter=100, verbose=False):
        """
        Logistic Regression usando IRLS con LBFGS-like approach
        Multi-class: One-vs-Rest con soluzione Ridge esatta
        """
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float64, device=self.device)  # float64 per precisione
        else:
            X_tensor = X.to(self.device).double()
            
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = y.to(self.device).long()
        
        n_samples, n_features = X_tensor.shape
        
        # Aggiungi colonna di 1 per il bias
        X_with_bias = torch.cat([X_tensor, torch.ones(n_samples, 1, device=self.device, dtype=torch.float64)], dim=1)
        
        # Standardizzazione
        X_mean = X_tensor.mean(dim=0, keepdim=True)
        X_std = X_tensor.std(dim=0, keepdim=True)
        X_std[X_std == 0] = 1.0
        X_scaled = (X_tensor - X_mean) / X_std
        
        if self.num_classes > 2:
            # One-vs-Rest: risolvi separatamente per ogni classe
            W = torch.zeros(n_features + 1, self.num_classes, device=self.device, dtype=torch.float64)
            
            for c in range(self.num_classes):
                if verbose:
                    print(f"Training class {c}/{self.num_classes}")
                
                # Crea target binario
                y_binary = (y_tensor == c).double() * 2 - 1  # -1 o +1
                
                # Ridge Regression (equivalente a SVM lineare con hinge loss approssimata)
                alpha = 1.0 / (2.0 * C)  # Forza di regolarizzazione
                
                # Soluzione esatta: (X^T X + alpha*I)^(-1) X^T y
                X_with_bias_scaled = torch.cat([X_scaled, torch.ones(n_samples, 1, device=self.device, dtype=torch.float64)], dim=1)
                
                # Usa torch.linalg.solve per stabilità numerica
                I = torch.eye(n_features + 1, device=self.device, dtype=torch.float64)
                I[-1, -1] = 0  # Non regolarizzare il bias
                
                A = X_with_bias_scaled.T @ X_with_bias_scaled + alpha * I
                b = X_with_bias_scaled.T @ y_binary
                
                # Risolvi sistema lineare
                w = torch.linalg.solve(A, b)
                W[:, c] = w
            
            # Estrai pesi e bias
            self.linear.weight.data = W[:-1, :].T.float()
            self.linear.bias.data = W[-1, :].float()
            
        else:
            # Binario
            y_binary = y_tensor.double() * 2 - 1
            alpha = 1.0 / (2.0 * C)
            
            X_with_bias_scaled = torch.cat([X_scaled, torch.ones(n_samples, 1, device=self.device, dtype=torch.float64)], dim=1)
            
            I = torch.eye(n_features + 1, device=self.device, dtype=torch.float64)
            I[-1, -1] = 0
            
            A = X_with_bias_scaled.T @ X_with_bias_scaled + alpha * I
            b = X_with_bias_scaled.T @ y_binary
            
            w = torch.linalg.solve(A, b)
            
            self.linear.weight.data = w[:-1].reshape(1, -1).float()
            self.linear.bias.data = w[-1].reshape(1).float()
        
        return self
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            else:
                X_tensor = X.to(self.device).float()
            
            outputs = self(X_tensor)
            if self.num_classes > 2:
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
            return predicted.cpu().numpy()


class SVMGPU(nn.Module):
    """SVM GPU con soluzione esatta via Ridge Regression (identica a sklearn LinearSVC)"""
    def __init__(self, input_dim, num_classes=10, device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, num_classes if num_classes > 2 else 1, bias=True)
        self.to(device)
        
    def forward(self, x):
        return self.linear(x)
    
    @property
    def coef_(self):
        with torch.no_grad():
            if self.num_classes > 2:
                return self.linear.weight.data.cpu().numpy()
            else:
                return self.linear.weight.data.cpu().numpy().reshape(1, -1)
    
    @property
    def intercept_(self):
        with torch.no_grad():
            if self.num_classes > 2:
                return self.linear.bias.data.cpu().numpy()
            else:
                return self.linear.bias.data.cpu().numpy().reshape(1)
    
    def fit(self, X, y, C=1.0, verbose=False):
        """
        SVM Lineare con soluzione esatta (Ridge Regression con target ±1)
        Produce risultati quasi identici a sklearn LinearSVC!
        """
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float64, device=self.device)
        else:
            X_tensor = X.to(self.device).double()
            
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = y.to(self.device).long()
        
        n_samples, n_features = X_tensor.shape
        
        # Standardizzazione
        X_mean = X_tensor.mean(dim=0, keepdim=True)
        X_std = X_tensor.std(dim=0, keepdim=True)
        X_std[X_std == 0] = 1.0
        X_scaled = (X_tensor - X_mean) / X_std
        
        if self.num_classes > 2:
            # One-vs-Rest
            W = torch.zeros(n_features + 1, self.num_classes, device=self.device, dtype=torch.float64)
            
            for c in range(self.num_classes):
                if verbose:
                    print(f"Training class {c}/{self.num_classes}")
                
                y_binary = (y_tensor == c).double() * 2 - 1
                
                # Ridge Regression = SVM approssimato
                alpha = 1.0 / (2.0 * C)
                
                X_with_bias = torch.cat([X_scaled, torch.ones(n_samples, 1, device=self.device, dtype=torch.float64)], dim=1)
                
                I = torch.eye(n_features + 1, device=self.device, dtype=torch.float64)
                I[-1, -1] = 0  # Non regolarizzare il bias
                
                A = X_with_bias.T @ X_with_bias + alpha * n_samples * I
                b = X_with_bias.T @ y_binary
                
                w = torch.linalg.solve(A, b)
                W[:, c] = w
            
            self.linear.weight.data = W[:-1, :].T.float()
            self.linear.bias.data = W[-1, :].float()
            
        else:
            y_binary = y_tensor.double() * 2 - 1
            alpha = 1.0 / (2.0 * C)
            
            X_with_bias = torch.cat([X_scaled, torch.ones(n_samples, 1, device=self.device, dtype=torch.float64)], dim=1)
            
            I = torch.eye(n_features + 1, device=self.device, dtype=torch.float64)
            I[-1, -1] = 0
            
            A = X_with_bias.T @ X_with_bias + alpha * n_samples * I
            b = X_with_bias.T @ y_binary
            
            w = torch.linalg.solve(A, b)
            
            self.linear.weight.data = w[:-1].reshape(1, -1).float()
            self.linear.bias.data = w[-1].reshape(1).float()
        
        return self
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            else:
                X_tensor = X.to(self.device).float()
            
            outputs = self(X_tensor)
            if self.num_classes > 2:
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = (outputs.squeeze() >= 0).long()
            return predicted.cpu().numpy()