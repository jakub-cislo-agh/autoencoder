import torch
import matplotlib.pyplot as plt

from autoencoder import Autoencoder
from mnist_loader import MNISTLoader

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# hyper parameters
epochs_num = 30
learning_rate = 0.0001

# Initialize the model, loss function, optimizer and scheduler
model = Autoencoder().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Load data
mnist = MNISTLoader()
train_loader = mnist.train_loader
test_loader = mnist.test_loader

# training loop
for epoch in range(epochs_num):
    model.train()
    running_loss = 0.0
    for batch_data, _ in train_loader:
        # move batch data to the device
        batch_data = batch_data.to(device)

        # forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # optimizing
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{epochs_num}], Loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "autoencoder_digits.pth")

# Visualisation
model.eval()
example_data, _ = next(iter(test_loader))
example_data = example_data.to(device)

with torch.no_grad():
    reconstructed = model(example_data)

fig, axes = plt.subplots(2, 10, figsize=(10, 2))

for i in range(10):
    # original images
    axes[0, i].imshow(example_data[i].cpu().numpy().squeeze(), cmap="gray")
    axes[0, i].axis("off")

    # reconstructed images
    axes[1, i].imshow(reconstructed[i].cpu().numpy().squeeze(), cmap="gray")
    axes[1, i].axis("off")

plt.show()