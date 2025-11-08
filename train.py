"""
Training script for VAE on anime faces.

Supports command-line configuration for all hyperparameters and paths.
"""

import os
import argparse
import random
from datetime import datetime
import glob

import numpy as np
import cv2

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def get_latest_model(models_dir):
    """
    Find the latest model checkpoint by timestamp.

    Args:
        models_dir (str): Directory containing model checkpoints

    Returns:
        str or None: Path to latest model, or None if no models found
    """
    if not os.path.exists(models_dir):
        return None

    model_files = glob.glob(os.path.join(models_dir, 'model_*.bin'))
    if not model_files:
        return None

    # Sort by timestamp in filename
    model_files.sort()
    return model_files[-1]


def save_checkpoint(encoder, decoder, optimizer, epoch, iteration, args, models_dir):
    """
    Save model checkpoint with timestamp.

    Args:
        encoder: Encoder network
        decoder: Decoder network
        optimizer: Optimizer
        epoch (int): Current epoch
        iteration (int): Current iteration
        args: Training arguments
        models_dir (str): Directory to save checkpoint
    """
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'model_{timestamp}.bin'
    filepath = os.path.join(models_dir, filename)

    checkpoint = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
        'args': vars(args)
    }

    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filepath}")


def load_images(data_path, image_size, batch_size):
    """
    Generator that yields batches of preprocessed images.

    Args:
        data_path (str): Directory containing training images
        image_size (int): Size to resize images to (image_size x image_size)
        batch_size (int): Number of images per batch

    Yields:
        np.ndarray: Batch of images with shape (batch_size, 3, image_size, image_size)
                   normalized to [-1, 1] range
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    filenames = os.listdir(data_path)
    if not filenames:
        raise ValueError(f"No files found in: {data_path}")

    batch = np.empty((batch_size, 3, image_size, image_size), dtype=np.float32)
    batch_idx = 0

    while True:
        random.shuffle(filenames)

        for filename in filenames:
            # Read and preprocess image
            img_path = os.path.join(data_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize and convert BGR to RGB
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize to [-1, 1] and convert to CHW format
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = (img / 255.0 - 0.5) * 2.0

            batch[batch_idx] = img
            batch_idx += 1

            # Yield when batch is full
            if batch_idx == batch_size:
                yield batch.copy()
                batch_idx = 0


def save_generated_images(decoder, output_dir, iteration, num_images, latent_dim, use_cuda):
    """
    Generate and save images from random latent vectors.

    Args:
        decoder: Decoder network (B_net)
        output_dir (str): Directory to save images
        iteration (int): Current iteration number (used for folder naming)
        num_images (int): Number of images to generate
        latent_dim (int): Dimension of latent space
        use_cuda (bool): Whether to use CUDA
    """
    # Sample random latent vectors
    z = torch.randn(num_images, latent_dim)
    if use_cuda:
        z = z.cuda()

    # Generate images
    with torch.no_grad():
        generated = decoder(z).cpu().numpy()

    # Create output directory
    save_path = os.path.join(output_dir, str(iteration))
    os.makedirs(save_path, exist_ok=True)

    # Save each generated image
    for i in range(num_images):
        # Denormalize from [-1, 1] to [0, 255]
        img = np.clip(generated[i] + 1.0, 0, 2.0) / 2.0 * 255.0
        img = img.astype(np.uint8)

        # Convert from CHW to HWC and RGB to BGR
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(save_path, f"{i}.png"), img)


def compute_vae_loss(encoder, decoder, images, sigma_sq):
    """
    Compute VAE loss (reconstruction + KL divergence).

    Args:
        encoder: Encoder network (A_net)
        decoder: Decoder network (B_net)
        images (torch.Tensor): Input images
        sigma_sq (float): Variance parameter for reconstruction loss

    Returns:
        tuple: (total_loss, kl_loss, recon_loss, reconstructed) where:
            - total_loss: Combined loss for backprop
            - kl_loss: KL divergence (scalar)
            - recon_loss: Reconstruction loss (scalar)
            - reconstructed: Reconstructed images tensor
    """
    image_size = images.size(2)

    # Encode to latent distribution parameters
    logvar, mu = encoder(images)

    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()

    # Reparameterization trick: z = mu + std * eps
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + std * eps

    # Decode
    reconstructed = decoder(z)

    # Reconstruction loss (MSE)
    recon_loss = torch.sum((images - reconstructed) ** 2, dim=(1, 2, 3))
    recon_loss = recon_loss / (sigma_sq * image_size * image_size)
    recon_loss = recon_loss.mean()

    # Total loss
    total_loss = kl_loss + recon_loss

    return total_loss, kl_loss.item(), recon_loss.item(), reconstructed


def main():
    parser = argparse.ArgumentParser(description='Train VAE for anime face generation')

    # Data paths
    parser.add_argument('--data-path', type=str, default='data',
                        help='Path to training images directory')
    parser.add_argument('--output-dir', type=str, default='images',
                        help='Directory to save generated images')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save/load model checkpoints (default: models)')
    parser.add_argument('--no-auto-load', action='store_true',
                        help='Disable automatic loading of latest checkpoint')

    # Image settings
    parser.add_argument('--image-size', type=int, default=128,
                        help='Input/output image size (default: 128)')

    # Network architecture
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Dimension of latent space (default: 128)')
    parser.add_argument('--base-channels', type=int, default=None,
                        help='Base number of channels in networks (default: 64 for standard, 128 for --deeper)')
    parser.add_argument('--deeper', action='store_true',
                        help='Use deeper networks (2x residual blocks, higher capacity)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate (default: 0.0002)')
    parser.add_argument('--sigma-sq', type=float, default=0.0001,
                        help='Variance parameter for reconstruction loss (default: 0.0001)')

    # Logging and visualization
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='TensorBoard log directory (default: runs)')
    parser.add_argument('--save-freq', type=int, default=5000,
                        help='Save model checkpoint every N iterations (default: 5000)')
    parser.add_argument('--plot-freq', type=int, default=10,
                        help='Plot losses every N iterations (default: 10)')
    parser.add_argument('--save-img-freq', type=int, default=50,
                        help='Save generated images every N iterations (default: 50)')
    parser.add_argument('--num-gen-images', type=int, default=16,
                        help='Number of images to generate (default: 16)')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')

    args = parser.parse_args()

    # Import appropriate network module
    if args.deeper:
        import nets_high as nets
        print("Using DEEP architecture (nets_high)")
        # Set default base_channels for deeper networks
        if args.base_channels is None:
            args.base_channels = 96
    else:
        import nets
        print("Using standard architecture (nets)")
        # Set default base_channels for standard networks
        if args.base_channels is None:
            args.base_channels = 64

    # Setup TensorBoard
    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter(args.log_dir)
        print(f"TensorBoard logging to: {args.log_dir}")

    # Calculate epoch length
    num_files = len([f for f in os.listdir(args.data_path)
                     if os.path.isfile(os.path.join(args.data_path, f))])
    epoch_len = num_files // args.batch_size
    print(f"Dataset: {num_files} images, {epoch_len} batches per epoch")

    # Create data loader
    data_loader = load_images(args.data_path, args.image_size, args.batch_size)

    # Create networks
    encoder, decoder = nets.create_vae(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels
    )

    # Optimizers
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )

    # Training state
    start_epoch = 0
    iteration = 0

    # Auto-load latest checkpoint
    if not args.no_auto_load:
        latest_model = get_latest_model(args.models_dir)
        if latest_model:
            print(f"Loading latest checkpoint: {latest_model}")
            checkpoint = torch.load(latest_model)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0)
            iteration = checkpoint.get('iteration', 0)
            print(f"Resumed from epoch {start_epoch}, iteration {iteration}")
        else:
            print(f"No checkpoint found in {args.models_dir}, starting from scratch")
    else:
        print("Auto-load disabled, starting from scratch")

    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train for one epoch
        for batch_idx in range(epoch_len):
            # Get batch
            images = next(data_loader)
            images = nets.FloatTensor(images)

            # Forward pass and compute loss
            optimizer.zero_grad()
            total_loss, kl_loss, recon_loss, reconstructed = compute_vae_loss(
                encoder, decoder, images, args.sigma_sq
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            iteration += 1

            # Save checkpoint periodically
            if iteration % args.save_freq == 0:
                save_checkpoint(encoder, decoder, optimizer, epoch, iteration, args, args.models_dir)

            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/KL', kl_loss, iteration)
                writer.add_scalar('Loss/Reconstruction', recon_loss, iteration)
                writer.add_scalar('Loss/Total', kl_loss + recon_loss, iteration)

            # Print losses
            if iteration % args.plot_freq == 0:
                print(f"  [{batch_idx}/{epoch_len}] KL: {kl_loss:.4f}, Recon: {recon_loss:.4f}")

                # Log images to TensorBoard
                if writer is not None:
                    # Log first 8 original and reconstructed images
                    num_imgs = min(8, images.size(0))
                    img_grid_orig = torch.clamp((images[:num_imgs] + 1) / 2, 0, 1)
                    img_grid_recon = torch.clamp((reconstructed[:num_imgs] + 1) / 2, 0, 1)
                    writer.add_images('Images/Original', img_grid_orig, iteration)
                    writer.add_images('Images/Reconstructed', img_grid_recon, iteration)

            # Generate and save images
            if iteration % args.save_img_freq == 0:
                img_set_num = iteration // args.save_img_freq
                print(f"  Generating image set {img_set_num}")

                # Generate from random latent vectors
                with torch.no_grad():
                    z = torch.randn(args.num_gen_images, args.latent_dim)
                    if nets.USE_CUDA:
                        z = z.cuda()
                    generated = decoder(z)

                # Save to disk
                save_generated_images(
                    decoder, args.output_dir, img_set_num,
                    args.num_gen_images, args.latent_dim, nets.USE_CUDA
                )

                # Log to TensorBoard
                if writer is not None:
                    num_imgs = min(8, generated.size(0))
                    img_grid_gen = torch.clamp((generated[:num_imgs] + 1) / 2, 0, 1)
                    writer.add_images('Images/Generated', img_grid_gen, iteration)

    print("\nTraining completed!")

    # Save final checkpoint
    print("Saving final checkpoint...")
    save_checkpoint(encoder, decoder, optimizer, args.epochs, iteration, args, args.models_dir)

    # Close TensorBoard writer
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
