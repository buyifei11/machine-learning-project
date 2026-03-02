import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, nonlinearity: str):
        super(Autoencoder, self).__init__()
        
        # Parse nonlinearity
        if nonlinearity == "ReLU":
            activation = nn.ReLU()
        elif nonlinearity == "LeakyReLU":
            activation = nn.LeakyReLU()
        elif nonlinearity == "Tanh":
            activation = nn.Tanh()
        else:
            activation = nn.ReLU()
            
        # Build Encoder
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(activation)
            current_dim = h_dim
            
        # Bottleneck (2D)
        self.encoder = nn.Sequential(*encoder_layers)
        self.bottleneck = nn.Linear(current_dim, 2)
        
        # Build Decoder
        decoder_layers = []
        current_dim = 2
        
        # Reverse the hidden layers for decoder
        for h_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(activation)
            current_dim = h_dim
            
        # Output layer
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.bottleneck(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded
