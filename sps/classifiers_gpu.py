import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogisticRegressionGPU(nn.Module):
    """Logistic Regression ottimizzata per GPU"""
    def __init__(self, input_dim, num_classes=10, device='cuda'):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(input_dim, num_classes if num_classes > 2 else 1)
        self.to(device)
        self.classes_ = None
        
    def forward(self, x):
        return self.linear(x)
    
    @property
    def coef_(self):
        """Compatibilità con scikit-learn: restituisce i coefficienti come numpy array"""
        with torch.no_grad():
            if self.linear.out_features > 1:
                return self.linear.weight.data.cpu().numpy()
            else:
                return self.linear.weight.data.cpu().numpy().reshape(1, -1)
    
    def fit(self, X, y, epochs=100, lr=0.01, weight_decay=0.001, verbose=False):
        """Fit del modello"""
        # Assicurati che i dati siano tensori PyTorch sul device corretto
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device)
            
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = y.to(self.device)
        
        # Determina il tipo di classificazione
        num_classes = len(y_tensor.unique())
        
        if num_classes > 2:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
            y_tensor = y_tensor.float()
            
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % 20 == 0:
                print(f'LogReg Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')
        
        return self
    
    def predict(self, X):
        """Predizione"""
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            else:
                X_tensor = X.to(self.device)
            
            outputs = self(X_tensor)
            
            if outputs.shape[-1] > 1:
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze()
            
            return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """Probabilità"""
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            else:
                X_tensor = X.to(self.device)
            
            outputs = self(X_tensor)
            
            if outputs.shape[-1] > 1:
                probs = torch.softmax(outputs, dim=-1)
            else:
                probs = torch.sigmoid(outputs)
            
            return probs.cpu().numpy()


class SVMGPU(nn.Module):
    """SVM Lineare ottimizzata per GPU"""
    def __init__(self, input_dim, num_classes=10, device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        
        if num_classes > 2:
            self.linear = nn.Linear(input_dim, num_classes)
        else:
            self.linear = nn.Linear(input_dim, 1)
        
        self.to(device)
        
    def forward(self, x):
        return self.linear(x)
    
    @property
    def coef_(self):
        """Compatibilità con scikit-learn: restituisce i coefficienti come numpy array"""
        with torch.no_grad():
            if self.num_classes > 2:
                return self.linear.weight.data.cpu().numpy()
            else:
                return self.linear.weight.data.cpu().numpy().reshape(1, -1)
    
    def fit(self, X, y, epochs=100, lr=0.01, C=1.0, verbose=False):
        """Fit SVM con Hinge Loss"""
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device)
            
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = y.to(self.device)
        
        optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=1/C)
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor)
            
            if self.num_classes > 2:
                # Multi-class hinge loss
                loss = nn.MultiMarginLoss(margin=1.0)(outputs, y_tensor)
            else:
                # Binary hinge loss
                y_binary = y_tensor.float() * 2 - 1  # Converti 0,1 in -1,1
                loss = torch.mean(torch.clamp(1 - y_binary * outputs.squeeze(), min=0))
            
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % 20 == 0:
                print(f'SVM Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')
        
        return self
    
    def predict(self, X):
        """Predizione"""
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            else:
                X_tensor = X.to(self.device)
            
            outputs = self(X_tensor)
            
            if self.num_classes > 2:
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = (outputs.squeeze() >= 0).long()
            
            return predicted.cpu().numpy()