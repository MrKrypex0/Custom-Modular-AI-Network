import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Assuming input data is normalized between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Example usage:
input_dim = 784  # For MNIST dataset (28x28 images flattened)
latent_dim = 32

model = Autoencoder(input_dim, latent_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if CUDA is available and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Loop
def train_autoencoder(model, dataLoader, epochs):
    for epoch in range(epochs):
        for data in dataLoader:
            img, _ = data # We don't need labels for unsupervised learning

            # Flatten the images
            img = img.view(img.size(0), -1)

            # Move data to GPU if available
            img = img.to(device)

            # Forward pass
            reconstructed, latent = model(img)
            loss = criterion(reconstructed, img)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
transforms.Normalize(mean=(0.5,), std=(0.5,))
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataLoader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train the model
train_autoencoder(model, dataLoader, epochs=10)

import matplotlib.pyplot as plt

def visualize_reconstruction(model, data):
    model.eval()
    with torch.no_grad():
        img, _ = data
        img = img.view(img.size(0), -1)
        # Move data to GPU if available
        img = img.to(device)
        reconstructed, _ = model(img)

        # Plot original and reconstructed images
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(5):
            ax = axes[0, i]
            ax.imshow(img[i].view(28, 28).cpu().numpy(), cmap='gray')
            ax.axis('off')

            ax = axes[1, i]
            ax.imshow(reconstructed[i].view(28, 28).cpu().numpy(), cmap='gray')
            ax.axis('off')

        plt.show()

# Visualize some reconstructions
visualize_reconstruction(model, next(iter(dataLoader)))
