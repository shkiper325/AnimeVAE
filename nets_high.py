"""
Deeper VAE Architecture for Anime Face Generation

This module implements a deeper Variational Autoencoder (VAE) with 2x more residual blocks
compared to the standard version, for enhanced feature learning capacity.

Architecture:
    - Encoder (A_net): Maps 3x128x128 images to latent space with 2 residual blocks per level
    - Decoder (B_net): Reconstructs 3x128x128 images with 2 residual blocks per level

Key features:
    - 2x deeper networks with more residual blocks
    - Better feature extraction at cost of higher computational requirements
    - Same architecture principles as standard version but deeper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Configuration
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

if USE_CUDA:
    print('Using CUDA for computation (DEEP architecture)')
else:
    print('Using CPU for computation (DEEP architecture)')


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.

    This block helps with gradient flow and allows training deeper networks.
    Architecture: Conv -> GroupNorm -> Activation -> Conv -> GroupNorm -> Add -> Activation

    Args:
        channels (int): Number of input and output channels
        num_groups (int): Number of groups for GroupNorm (default: 8)
    """
    def __init__(self, channels, num_groups=8):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """Forward pass with residual connection"""
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.gn2(out)

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
        num_groups (int): Number of groups for GroupNorm (default: 8)
    """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super(DownsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """Forward pass: spatial dimension halved, channels changed"""
        x = self.conv(x)
        x = self.gn(x)
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
        num_groups (int): Number of groups for GroupNorm (default: 8)
    """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super(UpsampleBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """Forward pass: spatial dimension doubled, channels changed"""
        x = self.upsample(x)
        x = self.conv(x)
        x = self.gn(x)
        x = self.activation(x)
        return x


class A_net(nn.Module):
    """
    Deeper Encoder network for VAE (Recognition/Inference network).

    Maps input images to latent space parameters with 2x residual blocks per level.
    Architecture progressively downsamples the image while increasing channels:
        3x128x128 -> 64x64x64 -> 128x32x32 -> 256x16x16 -> 512x8x8 -> 512x4x4

    Args:
        latent_dim (int): Dimension of the latent space (default: 128)
        base_channels (int): Base number of channels (default: 128 for deep version)
    """
    def __init__(self, latent_dim=128, base_channels=128):
        super(A_net, self).__init__()

        self.latent_dim = latent_dim

        # Initial convolution: 3x128x128 -> base_channels x 128x128
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Encoder pathway with progressive downsampling
        # base_channels x 128x128 -> base_channels x 64x64
        self.down1 = DownsampleBlock(base_channels, base_channels, num_groups=8)
        self.res1a = ResidualBlock(base_channels, num_groups=8)
        self.res1b = ResidualBlock(base_channels, num_groups=8)

        # base_channels x 64x64 -> base_channels*2 x 32x32
        self.down2 = DownsampleBlock(base_channels, base_channels * 2, num_groups=8)
        self.res2a = ResidualBlock(base_channels * 2, num_groups=8)
        self.res2b = ResidualBlock(base_channels * 2, num_groups=8)

        # base_channels*2 x 32x32 -> base_channels*4 x 16x16
        self.down3 = DownsampleBlock(base_channels * 2, base_channels * 4, num_groups=8)
        self.res3a = ResidualBlock(base_channels * 4, num_groups=8)
        self.res3b = ResidualBlock(base_channels * 4, num_groups=8)

        # base_channels*4 x 16x16 -> base_channels*8 x 8x8
        self.down4 = DownsampleBlock(base_channels * 4, base_channels * 8, num_groups=8)
        self.res4a = ResidualBlock(base_channels * 8, num_groups=8)
        self.res4b = ResidualBlock(base_channels * 8, num_groups=8)

        # base_channels*8 x 8x8 -> base_channels*8 x 4x4
        self.down5 = DownsampleBlock(base_channels * 8, base_channels * 8, num_groups=8)
        self.res5a = ResidualBlock(base_channels * 8, num_groups=8)
        self.res5b = ResidualBlock(base_channels * 8, num_groups=8)

        # Flatten spatial dimensions
        self.flatten = nn.Flatten()

        # Linear layers for latent parameters
        feature_dim = base_channels * 8 * 4 * 4

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

        # Progressive downsampling with 2x residual blocks per level
        x = self.down1(x)
        x = self.res1a(x)
        x = self.res1b(x)

        x = self.down2(x)
        x = self.res2a(x)
        x = self.res2b(x)

        x = self.down3(x)
        x = self.res3a(x)
        x = self.res3b(x)

        x = self.down4(x)
        x = self.res4a(x)
        x = self.res4b(x)

        x = self.down5(x)
        x = self.res5a(x)
        x = self.res5b(x)

        # Flatten and project to latent space
        x = self.flatten(x)

        logvar = self.fc_logvar(x)
        mu = self.fc_mu(x)

        return logvar, mu


class B_net(nn.Module):
    """
    Deeper Decoder network for VAE (Generative network).

    Reconstructs images from latent space vectors with 2x residual blocks per level.
    Architecture progressively upsamples while decreasing channels:
        latent_dim -> base_channels*8 x 4x4 -> ... -> 3x128x128

    Args:
        latent_dim (int): Dimension of the latent space (default: 128)
        base_channels (int): Base number of channels (default: 128 for deep version)
    """
    def __init__(self, latent_dim=128, base_channels=128):
        super(B_net, self).__init__()

        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # Project latent vector to initial spatial feature map
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 4 * 4)

        # Initial residual processing at 4x4 resolution
        self.res_initial_a = ResidualBlock(base_channels * 8, num_groups=8)
        self.res_initial_b = ResidualBlock(base_channels * 8, num_groups=8)

        # Decoder pathway with progressive upsampling
        # base_channels*8 x 4x4 -> base_channels*8 x 8x8
        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 8, num_groups=8)
        self.res1a = ResidualBlock(base_channels * 8, num_groups=8)
        self.res1b = ResidualBlock(base_channels * 8, num_groups=8)

        # base_channels*8 x 8x8 -> base_channels*4 x 16x16
        self.up2 = UpsampleBlock(base_channels * 8, base_channels * 4, num_groups=8)
        self.res2a = ResidualBlock(base_channels * 4, num_groups=8)
        self.res2b = ResidualBlock(base_channels * 4, num_groups=8)

        # base_channels*4 x 16x16 -> base_channels*2 x 32x32
        self.up3 = UpsampleBlock(base_channels * 4, base_channels * 2, num_groups=8)
        self.res3a = ResidualBlock(base_channels * 2, num_groups=8)
        self.res3b = ResidualBlock(base_channels * 2, num_groups=8)

        # base_channels*2 x 32x32 -> base_channels x 64x64
        self.up4 = UpsampleBlock(base_channels * 2, base_channels, num_groups=8)
        self.res4a = ResidualBlock(base_channels, num_groups=8)
        self.res4b = ResidualBlock(base_channels, num_groups=8)

        # base_channels x 64x64 -> base_channels x 128x128
        self.up5 = UpsampleBlock(base_channels, base_channels, num_groups=8)
        self.res5a = ResidualBlock(base_channels, num_groups=8)
        self.res5b = ResidualBlock(base_channels, num_groups=8)

        # Final convolution to RGB: base_channels x 128x128 -> 3x128x128
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1),
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
        x = self.res_initial_a(x)
        x = self.res_initial_b(x)

        # Progressive upsampling with 2x residual blocks per level
        x = self.up1(x)
        x = self.res1a(x)
        x = self.res1b(x)

        x = self.up2(x)
        x = self.res2a(x)
        x = self.res2b(x)

        x = self.up3(x)
        x = self.res3a(x)
        x = self.res3b(x)

        x = self.up4(x)
        x = self.res4a(x)
        x = self.res4b(x)

        x = self.up5(x)
        x = self.res5a(x)
        x = self.res5b(x)

        # Final convolution to RGB
        x = self.final_conv(x)

        return x


def init_weights(m):
    """
    Initialize network weights using modern best practices.

    Uses Kaiming (He) initialization for convolutional and linear layers,
    which is optimal for ReLU-like activations (including LeakyReLU).

    Args:
        m (nn.Module): Module to initialize
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Kaiming initialization for conv layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        # Kaiming initialization for linear layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        # Standard initialization for normalization layers
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def create_vae(latent_dim=128, base_channels=128):
    """
    Factory function to create and initialize deeper encoder and decoder networks.

    Args:
        latent_dim (int): Dimension of the latent space (default: 128)
        base_channels (int): Base number of channels in the networks (default: 128)

    Returns:
        tuple: (encoder, decoder) - initialized A_net and B_net instances

    Example:
        >>> encoder, decoder = create_vae(latent_dim=256, base_channels=128)
        >>> # Forward pass
        >>> logvar, mu = encoder(images)
        >>> reconstructed = decoder(latent_samples)
    """
    encoder = A_net(latent_dim=latent_dim, base_channels=base_channels)
    decoder = B_net(latent_dim=latent_dim, base_channels=base_channels)

    # Initialize weights
    encoder.apply(init_weights)
    decoder.apply(init_weights)

    return encoder, decoder


if __name__ == "__main__":
    """
    Test the network architectures and print parameter counts.
    """
    print("=" * 70)
    print("Testing DEEP VAE Architecture")
    print("=" * 70)

    # Create networks
    latent_dim = 128
    encoder, decoder = create_vae(latent_dim=latent_dim, base_channels=128)

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
    print("All tests passed! DEEP network is ready for training.")
    print("=" * 70)
