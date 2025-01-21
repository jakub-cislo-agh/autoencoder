import torch
import matplotlib.pyplot as plt

from autoencoder import Autoencoder
from mnist_loader import MNISTLoader

# Choose device
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model, loss function and optimizer
model = Autoencoder().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Load data
mnist = MNISTLoader()
train_loader = mnist.train_loader
test_loader = mnist.test_loader

# Learning rate scheduler

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_data, _ in train_loader:
        # Move the batch data to the correct device
        batch_data = batch_data.to(device)

        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()  # Reduce the learning rate after each epoch

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "autoencoder_digits.pth")

# Visualize original and reconstructed images
model.eval()
example_data, _ = next(iter(test_loader))
example_data = example_data.to(device)

with torch.no_grad():
    reconstructed = model(example_data)

# Visualization
fig, axes = plt.subplots(2, 10, figsize=(10, 2))

for i in range(10):
    # Original images
    axes[0, i].imshow(example_data[i].cpu().numpy().squeeze(), cmap="gray")
    axes[0, i].axis("off")

    # Reconstructed images
    axes[1, i].imshow(reconstructed[i].cpu().numpy().squeeze(), cmap="gray")
    axes[1, i].axis("off")

plt.show()