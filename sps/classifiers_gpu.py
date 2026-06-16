import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogisticRegressionGPU(nn.Module):
    def __init__(self, input_dim, num_classes=10, device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
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
    
    def fit(self, X, y, epochs=2000, lr=0.01, weight_decay=0.0001, verbose=False):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device).float()
            
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = y.to(self.device).long()
        
        # Standardizzazione (media 0, std 1) - IMPORTANTE
        X_mean = X_tensor.mean(dim=0, keepdim=True)
        X_std = X_tensor.std(dim=0, keepdim=True) + 1e-8
        X_tensor = (X_tensor - X_mean) / X_std
        
        criterion = nn.CrossEntropyLoss() if self.num_classes > 2 else nn.BCEWithLogitsLoss()
        
        # SGD con learning rate costante (come sklearn)
        optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor if self.num_classes > 2 else y_tensor.float())
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if verbose and epoch % 200 == 0:
                with torch.no_grad():
                    if self.num_classes > 2:
                        pred = torch.argmax(outputs, dim=1)
                    else:
                        pred = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                    acc = (pred == y_tensor).float().mean()
                print(f'LogReg Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}')
        
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
    def __init__(self, input_dim, num_classes=10, device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
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
    
    def fit(self, X, y, epochs=2000, lr=0.01, C=1.0, verbose=False):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device).float()
            
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = y.to(self.device).long()
        
        # Standardizzazione
        X_mean = X_tensor.mean(dim=0, keepdim=True)
        X_std = X_tensor.std(dim=0, keepdim=True) + 1e-8
        X_tensor = (X_tensor - X_mean) / X_std
        
        # Weight decay = 1/(2*C) per matching con sklearn
        optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=1.0/(2.0*C))
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor)
            
            if self.num_classes > 2:
                loss = nn.MultiMarginLoss(margin=1.0)(outputs, y_tensor)
            else:
                y_binary = y_tensor.float() * 2 - 1
                loss = torch.mean(torch.clamp(1 - y_binary * outputs.squeeze(), min=0))
            
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % 200 == 0:
                with torch.no_grad():
                    if self.num_classes > 2:
                        pred = torch.argmax(outputs, dim=1)
                    else:
                        pred = (outputs.squeeze() >= 0).long()
                    acc = (pred == y_tensor).float().mean()
                print(f'SVM Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}, Acc: {acc.item():.4f}')
        
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