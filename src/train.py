import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import numpy as np

def train_autoencoder(X, model, optimizer_name, epochs, batch_size, noise_factor, patience, orthogonalize_pca=False):
    """
    Train the autoencoder. Yields progress dictionaries to integrate with Streamlit.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    # Convert data to tensor
    tensor_X = torch.Tensor(X).to(device)
    dataset = TensorDataset(tensor_X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in dataloader:
            inputs = batch[0]
            
            # Add noise for denoising autoencoder
            if noise_factor > 0:
                noisy_inputs = inputs + noise_factor * torch.randn_like(inputs)
            else:
                noisy_inputs = inputs
                
            optimizer.zero_grad()
            
            outputs, _ = model(noisy_inputs)
            loss = criterion(outputs, inputs) # compare against original clean inputs
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(dataloader.dataset)
        
        # Early Stopping Logic
        if train_loss < best_loss:
            best_loss = train_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            
        stop_training = False
        if epochs_no_improve >= patience:
            stop_training = True
            
        # Yield progress for Streamlit
        yield {
            "epoch": epoch + 1,
            "loss": train_loss,
            "best_loss": best_loss,
            "patience_counter": epochs_no_improve,
            "stopped": stop_training
        }
        
        if stop_training:
            break
            
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    model.eval()
    with torch.no_grad():
        _, latent_space = model(tensor_X)
        latent_space = latent_space.cpu().numpy()
        
    if orthogonalize_pca:
        pca = PCA(n_components=2)
        latent_space = pca.fit_transform(latent_space)
        
    yield {
        "final": True,
        "latent_space": latent_space
    }
