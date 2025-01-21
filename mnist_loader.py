from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTLoader():
    def __init__ (self):
        # Transform the MNIST dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to the range [-1, 1]
        ])

        # Load the MNIST dataset
        self.train_dataset = datasets.MNIST(root="./data", train=True, transform=self.transform, download=True)
        self.test_dataset = datasets.MNIST(root="./data", train=False, transform=self.transform, download=True)

        # Create DataLoader
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
