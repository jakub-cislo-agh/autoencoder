import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Check if MPS is available and set device accordingly
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class TrimLayer(nn.Module):
    def __init__(self):
        super(TrimLayer, self).__init__()

    def forward(self, x):
        # Trim the result from 1x29x29 to 1x28x28 by slicing
        return x[:, :, :28, :28]

# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.Flatten(),
            nn.Linear(3136, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            torch.nn.Linear(2, 3136),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
            TrimLayer(),  # 1x29x29 -> 1x28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Transform the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=False)

# Create a DataLoader
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_data, _ in dataloader:
        # Move the batch data to the correct device
        batch_data = batch_data.to(device)

        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "autoencoder_digits.pth")

# Visualize original and reconstructed images
import matplotlib.pyplot as plt

# Load one batch of data and move to the correct device
example_data, _ = next(iter(dataloader))
example_data = example_data.to(device)  # Move data to the device

# Get reconstructed images
with torch.no_grad():
    reconstructed = model(example_data)  # Pass the data directly to the model

# Visualization (optional)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 10, figsize=(10, 2))

for i in range(10):
    # Original images
    axes[0, i].imshow(example_data[i].cpu().numpy().squeeze(), cmap="gray")
    axes[0, i].axis("off")

    # Reconstructed images
    axes[1, i].imshow(reconstructed[i].cpu().numpy().squeeze(), cmap="gray")
    axes[1, i].axis("off")

plt.show()