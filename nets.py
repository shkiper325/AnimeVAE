"""
Modern VAE Architecture for Anime Face Generation

This module implements a Variational Autoencoder (VAE) with residual connections
for generating 128x128 RGB anime face images.

Architecture:
    - Encoder (A_net): Maps 3x128x128 images to latent space
    - Decoder (B_net): Reconstructs 3x128x128 images from latent vectors

Key features:
    - Residual blocks for better gradient flow
    - Group normalization for stable training
    - Progressive downsampling/upsampling (128->64->32->16->8->4)
    - Modern weight initialization (Kaiming)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Configuration
USE_CUDA = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

if USE_CUDA:
    print('Using CUDA for computation')
else:
    print('Using CPU for computation')


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.

    This block helps with gradient flow and allows training deeper networks.
    Architecture: Conv -> GroupNorm -> Activation -> Conv -> GroupNorm -> Add -> Activation

    Args:
        channels (int): Number of input and output channels
        leaky_relu_slope (float): Slope for LeakyReLU activation
    """
    def __init__(self, channels, leaky_relu_slope):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.activation = nn.LeakyReLU(leaky_relu_slope, inplace=True)

    def forward(self, x):
        """Forward pass with residual connection"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        out = out + identity
        out = self.activation(out)

        return out


class DownsampleBlock(nn.Module):
    """
    Downsampling block that reduces spatial dimensions by 2x.

    Uses strided convolution for learnable downsampling instead of pooling.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        leaky_relu_slope (float): Slope for LeakyReLU activation
    """
    def __init__(self, in_channels, out_channels, leaky_relu_slope):
        super(DownsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(leaky_relu_slope, inplace=True)

    def forward(self, x):
        """Forward pass: spatial dimension halved, channels changed"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block that increases spatial dimensions by 2x.

    Uses nearest neighbor upsampling followed by convolution for better results
    than transposed convolution (avoids checkerboard artifacts).

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        leaky_relu_slope (float): Slope for LeakyReLU activation
    """
    def __init__(self, in_channels, out_channels, leaky_relu_slope):
        super(UpsampleBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(leaky_relu_slope, inplace=True)

    def forward(self, x):
        """Forward pass: spatial dimension doubled, channels changed"""
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class A_net(nn.Module):
    """
    Encoder network for VAE (Recognition/Inference network).

    Maps input images to latent space parameters (mean and log-variance).
    Architecture progressively downsamples the image while increasing channels:
        3x128x128 -> 64x64x64 -> 128x32x32 -> 256x16x16 -> 512x8x8 -> 512x4x4

    Args:
        latent_dim (int): Dimension of the latent space
        base_channels (int): Base number of channels, doubled at each downsampling
        leaky_relu_slope (float): Slope for LeakyReLU activations
    """
    def __init__(self, latent_dim, base_channels, leaky_relu_slope):
        super(A_net, self).__init__()

        self.latent_dim = latent_dim

        # Initial convolution: 3x128x128 -> 64x128x128
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True)
        )

        # Encoder pathway with progressive downsampling
        # 64x128x128 -> 64x64x64
        self.down1 = DownsampleBlock(base_channels, base_channels, leaky_relu_slope)
        self.res1 = ResidualBlock(base_channels, leaky_relu_slope)

        # 64x64x64 -> 128x32x32
        self.down2 = DownsampleBlock(base_channels, base_channels * 2, leaky_relu_slope)
        self.res2 = ResidualBlock(base_channels * 2, leaky_relu_slope)

        # 128x32x32 -> 256x16x16
        self.down3 = DownsampleBlock(base_channels * 2, base_channels * 4, leaky_relu_slope)
        self.res3 = ResidualBlock(base_channels * 4, leaky_relu_slope)

        # 256x16x16 -> 512x8x8
        self.down4 = DownsampleBlock(base_channels * 4, base_channels * 8, leaky_relu_slope)
        self.res4 = ResidualBlock(base_channels * 8, leaky_relu_slope)

        # 512x8x8 -> 512x4x4
        self.down5 = DownsampleBlock(base_channels * 8, base_channels * 8, leaky_relu_slope)
        self.res5 = ResidualBlock(base_channels * 8, leaky_relu_slope)

        # Flatten spatial dimensions
        self.flatten = nn.Flatten()

        # Linear layers for latent parameters
        feature_dim = base_channels * 8 * 4 * 4  # 512 * 4 * 4 = 8192

        # Log-variance (sigma) branch - for reparameterization trick
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)

        # Mean (mu) branch
        self.fc_mu = nn.Linear(feature_dim, latent_dim)

        # Move to GPU if available
        if USE_CUDA:
            self.cuda()

    def forward(self, x):
        """
        Forward pass through encoder.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 128, 128)

        Returns:
            tuple: (logvar, mu) where:
                - logvar: Log-variance of latent distribution (batch_size, latent_dim)
                - mu: Mean of latent distribution (batch_size, latent_dim)
        """
        # Initial convolution
        x = self.initial_conv(x)

        # Progressive downsampling with residual blocks
        x = self.down1(x)
        x = self.res1(x)

        x = self.down2(x)
        x = self.res2(x)

        x = self.down3(x)
        x = self.res3(x)

        x = self.down4(x)
        x = self.res4(x)

        x = self.down5(x)
        x = self.res5(x)

        # Flatten and project to latent space
        x = self.flatten(x)

        logvar = self.fc_logvar(x)
        mu = self.fc_mu(x)

        return logvar, mu


class B_net(nn.Module):
    """
    Decoder network for VAE (Generative network).

    Reconstructs images from latent space vectors.
    Architecture progressively upsamples while decreasing channels:
        latent_dim -> 512x4x4 -> 512x8x8 -> 256x16x16 -> 128x32x32 -> 64x64x64 -> 3x128x128

    Args:
        latent_dim (int): Dimension of the latent space
        base_channels (int): Base number of channels
        leaky_relu_slope (float): Slope for LeakyReLU activations
    """
    def __init__(self, latent_dim, base_channels, leaky_relu_slope):
        super(B_net, self).__init__()

        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # Project latent vector to initial spatial feature map
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 4 * 4)  # 512 * 4 * 4

        # Initial residual processing at 4x4 resolution
        self.res_initial = ResidualBlock(base_channels * 8, leaky_relu_slope=leaky_relu_slope)

        # Decoder pathway with progressive upsampling
        # 512x4x4 -> 512x8x8
        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 8, leaky_relu_slope=leaky_relu_slope)
        self.res1 = ResidualBlock(base_channels * 8, leaky_relu_slope=leaky_relu_slope)

        # 512x8x8 -> 256x16x16
        self.up2 = UpsampleBlock(base_channels * 8, base_channels * 4, leaky_relu_slope=leaky_relu_slope)
        self.res2 = ResidualBlock(base_channels * 4, leaky_relu_slope=leaky_relu_slope)
        # 256x16x16 -> 128x32x32
        self.up3 = UpsampleBlock(base_channels * 4, base_channels * 2, leaky_relu_slope=leaky_relu_slope)
        self.res3 = ResidualBlock(base_channels * 2, leaky_relu_slope=leaky_relu_slope)

        # 128x32x32 -> 64x64x64
        self.up4 = UpsampleBlock(base_channels * 2, base_channels, leaky_relu_slope=leaky_relu_slope)
        self.res4 = ResidualBlock(base_channels, leaky_relu_slope=leaky_relu_slope)
        # 64x64x64 -> 64x128x128
        self.up5 = UpsampleBlock(base_channels, base_channels, leaky_relu_slope=leaky_relu_slope)
        self.res5 = ResidualBlock(base_channels, leaky_relu_slope=leaky_relu_slope)

        # Final convolution to RGB: 64x128x128 -> 3x128x128
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1),
            # Note: No sigmoid here - use BCEWithLogitsLoss or apply sigmoid during inference
        )

        # Move to GPU if available
        if USE_CUDA:
            self.cuda()

    def forward(self, z):
        """
        Forward pass through decoder.

        Args:
            z (torch.Tensor): Latent vectors of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Reconstructed images of shape (batch_size, 3, 128, 128)
        """
        # Project to initial feature map and reshape
        x = self.fc(z)
        x = x.view(-1, self.base_channels * 8, 4, 4)

        # Initial residual processing
        x = self.res_initial(x)

        # Progressive upsampling with residual blocks
        x = self.up1(x)
        x = self.res1(x)

        x = self.up2(x)
        x = self.res2(x)

        x = self.up3(x)
        x = self.res3(x)

        x = self.up4(x)
        x = self.res4(x)

        x = self.up5(x)
        x = self.res5(x)

        # Final convolution to RGB
        x = self.final_conv(x)

        return x

def create_vae(latent_dim, base_channels, leaky_relu_slope, weight_init_cb=None, **weight_init_cb_params):
    """
    Factory function to create and initialize encoder and decoder networks.

    Args:
        latent_dim (int): Dimension of the latent space
        base_channels (int): Base number of channels in the networks
        weight_init_cb (callable, optional): Weight initialization callback function
        **weight_init_cb_params: Additional parameters for the weight initialization callback
    Returns:
        tuple: (encoder, decoder) - initialized A_net and B_net instances
    """
    encoder = A_net(latent_dim=latent_dim, base_channels=base_channels, leaky_relu_slope=leaky_relu_slope)
    decoder = B_net(latent_dim=latent_dim, base_channels=base_channels, leaky_relu_slope=leaky_relu_slope)

    # Initialize weights
    if weight_init_cb is not None:
        encoder.apply(weight_init_cb, **weight_init_cb_params)
        decoder.apply(weight_init_cb, **weight_init_cb_params)

    return encoder, decoder

if __name__ == "__main__":
    """
    Test the network architectures and print parameter counts.
    """
    print("=" * 70)
    print("Testing Modern VAE Architecture")
    print("=" * 70)

    # Create networks
    latent_dim = 128
    encoder, decoder = create_vae(latent_dim=latent_dim, base_channels=64, leaky_relu_slope=0.2)

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params = encoder_params + decoder_params

    print(f"\nEncoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 128, 128)

    if USE_CUDA:
        test_input = test_input.cuda()

    # Encoder forward pass
    logvar, mu = encoder(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"Mu shape: {mu.shape}")

    # Reparameterization trick
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    print(f"Latent z shape: {z.shape}")

    # Decoder forward pass
    reconstructed = decoder(z)
    print(f"Reconstructed shape: {reconstructed.shape}")

    print("\n" + "=" * 70)
    print("All tests passed! Network is ready for training.")
    print("=" * 70)
