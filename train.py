import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import get_prepared_data

# A Linear model serves as a White Box model (inherently interpretable)
class WhiteBoxModel(nn.Module):
    def __init__(self, input_dim):
        super(WhiteBoxModel, self).__init__()
        # A single linear layer without non-linear activation functions
        # allows us to interpret the learned weights directly
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def train_model():
    X, y, feature_names = get_prepared_data()
    
    input_dim = X.shape[1]
    model = WhiteBoxModel(input_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    print("Training Explainable White Box Model (Linear Regression)...")
    
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    print("\nModel trained. Explainability (Feature Weights):")
    # Explainable AI (XAI) for a white box model:
    # We can inspect the weights to see how each feature impacts the sustainability score.
    weights = model.linear.weight.detach().numpy()[0]
    bias = model.linear.bias.detach().numpy()[0]
    
    for name, weight in zip(feature_names, weights):
        print(f"Feature: {name:20} Weight: {weight:.4f}")
    print(f"Bias                 Weight: {bias:.4f}")

if __name__ == "__main__":
    train_model()
