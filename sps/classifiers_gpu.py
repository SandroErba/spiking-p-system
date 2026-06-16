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
    
    def fit(self, X, y, epochs=500, lr=0.1, weight_decay=0.0001, verbose=False):
        """Fit del modello - Parametri ottimizzati per matching con scikit-learn"""
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device)
            
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = y.to(self.device)
        
        # Normalizzazione dei dati (importante per convergenza!)
        X_mean = X_tensor.mean(dim=0, keepdim=True)
        X_std = X_tensor.std(dim=0, keepdim=True) + 1e-8
        X_tensor = (X_tensor - X_mean) / X_std
        
        num_classes = len(y_tensor.unique())
        
        if num_classes > 2:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
            y_tensor = y_tensor.float()
        
        # SGD con momentum funziona meglio di Adam per problemi lineari
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if verbose and epoch % 50 == 0:
                # Calcola accuracy
                with torch.no_grad():
                    if num_classes > 2:
                        pred = torch.argmax(outputs, dim=1)
                    else:
                        pred = (torch.sigmoid(outputs) > 0.5).long()
                    acc = (pred == y_tensor).float().mean()
                print(f'LogReg Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}')
        
        return self
    
    def predict(self, X):
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
            self.linear = nn.Linear(input_dim, num_classes, bias=True)
        else:
            self.linear = nn.Linear(input_dim, 1, bias=True)
        
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
    
    def fit(self, X, y, epochs=500, lr=0.1, C=1.0, verbose=False):
        """Fit SVM con Hinge Loss - Parametri ottimizzati"""
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device)
            
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = y.to(self.device)
        
        # Normalizzazione
        X_mean = X_tensor.mean(dim=0, keepdim=True)
        X_std = X_tensor.std(dim=0, keepdim=True) + 1e-8
        X_tensor = (X_tensor - X_mean) / X_std
        
        # Weight decay = 1/C (come in scikit-learn)
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1.0/C)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor)
            
            if self.num_classes > 2:
                loss = nn.MultiMarginLoss(margin=1.0)(outputs, y_tensor)
                with torch.no_grad():
                    pred = torch.argmax(outputs, dim=1)
                    acc = (pred == y_tensor).float().mean()
            else:
                y_binary = y_tensor.float() * 2 - 1
                loss = torch.mean(torch.clamp(1 - y_binary * outputs.squeeze(), min=0))
                with torch.no_grad():
                    pred = (outputs.squeeze() >= 0).long()
                    acc = (pred == y_tensor).float().mean()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if verbose and epoch % 50 == 0:
                print(f'SVM Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}')
        
        return self
    
    def predict(self, X):
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