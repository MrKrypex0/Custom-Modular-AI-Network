#
# Copyright (c) 2025 MrKrypex0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # This should be (784, 400)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # This is correct
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # This is correct

    def forward(self, x):
        h = torch.relu(self.bn1(self.fc1(x)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)  # This should be (20, 400)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # This should be (400, 784)

    def forward(self, z):
        h = torch.relu(self.bn1(self.fc1(z)))
        x_reconstructed = torch.sigmoid(self.fc2(h))
        return x_reconstructed
    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)  # Ensure eps is on the same device
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

    
# Hyperparameters
input_dim = 784  # For MNIST dataset (28x28 images flattened)
hidden_dim = 400
latent_dim = 32  # Ensure this matches the dimensions in your layers
num_epochs = 10
batch_size = 64
learning_rate = 1e-3

# Data loader for MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)  # Move model to GPU if available
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss(reduction='sum')

# Training the VAE
def loss_function(x_reconstructed, data, mu, logvar, kl_weight=1.0):
    bce_loss = criterion(x_reconstructed, data.view(-1, input_dim))
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kl_weight + kld_loss


for epoch in range(num_epochs):
    kl_weight = min(1.0, 0.01 + epoch / (num_epochs * 0.5))  # Ramp up over half the epochs
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim).to(device)  # Flatten the images and move to GPU if available

        optimizer.zero_grad()
        x_reconstructed, mu, logvar = model(data)
        loss = loss_function(x_reconstructed, data, mu, logvar, kl_weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset)}')


# Testing the VAE
# model.eval()
# test_loss = 0
# with torch.no_grad():
#     for data, _ in test_loader:
#         data = data.view(-1, input_dim).to(device)  # Flatten the images and move to GPU if available
# 
#         x_reconstructed, mu, logvar = model(data)
#         loss = loss_function(x_reconstructed, data, mu, logvar)
#         test_loss += loss.item()
# 
#     print(f'Test Loss: {test_loss / len(test_loader.dataset)}')

import matplotlib.pyplot as plt

def show_images(images, title=""):
    images = images.view(images.shape[0], 28, 28).cpu().detach().numpy()
    fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

model.eval()
with torch.no_grad():
    test_data, _ = next(iter(test_loader))
    test_data = test_data.view(-1, input_dim).to(device)  # Flatten and move to device
    x_recon, _, _ = model(test_data)
    
    idxs = [0, 1, 2, 3, 4]
    show_images(test_data[idxs], title="Original")
    show_images(x_recon[idxs], title="Reconstructed")


z = torch.randn(10, latent_dim).to(device)
with torch.no_grad():
    samples = model.decoder(z)

show_images(samples, title="Generated Samples")