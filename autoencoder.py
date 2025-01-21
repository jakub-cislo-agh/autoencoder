import torch.nn as nn

# Define the Autoencoder model with a larger latent space (e.g., 64-dimensional latent vector)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.LeakyReLU(0.1),  # Use LeakyReLU with negative slope
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.LeakyReLU(0.1),  # Use LeakyReLU with negative slope
            nn.Flatten(),  # Flatten the 7x7x64 to a vector
            nn.Linear(7*7*64, 64)  # Encode to a 64-dimensional latent vector
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 7*7*64),
            nn.LeakyReLU(0.1),  # LeakyReLU in the decoder as well
            nn.Unflatten(1, (64, 7, 7)),  # Reshape to (64, 7, 7)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()  # Normalize the output to [0, 1]
        )

        # Weight initialization (Xavier initialization)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x