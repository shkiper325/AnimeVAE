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
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from torchvision.models import vgg19

import nets
vgg = vgg19(pretrained=True).features[:36].eval()
if nets.USE_CUDA:
    vgg = vgg.cuda()

def perceptual_loss(x, x_recon):
    return F.mse_loss(vgg(x), vgg(x_recon), reduce='mean')

def init_weights_xavier(m):
    """
    Initialize network weights using Xavier initialization.

    Args:
        m: PyTorch module to initialize
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_normal(m, args):
    """
    Initialize network weights using Normal initialization.

    Args:
        m: PyTorch module to initialize
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, args.normal_init_mean, args.normal_init_std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, args.normal_init_std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_kaiming_uniform(m, args):
    """
    Initialize network weights using Kaiming uniform initialization.

    Args:
        m: PyTorch module to initialize
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, a=args.leaky_relu_slope,
                                 mode=args.init_method_mode, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, args.normal_init_std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

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
    filename = f'model_{timestamp}_{iteration}.bin'
    filepath = os.path.join(models_dir, filename)

    checkpoint = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
        'last_lr': optimizer.param_groups[0]['lr'],
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


def compute_vae_loss(encoder, decoder, images, sigma_sq, args):
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
    kl_loss = args.beta * kl_loss.mean()

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

    # Perceptual loss
    perc_loss = args.pi * perceptual_loss(images, reconstructed)

    # Total loss
    total_loss = kl_loss + recon_loss + perc_loss

    return total_loss, kl_loss.item(), recon_loss.item(), perc_loss.item(), reconstructed


def main():
    parser = argparse.ArgumentParser(description='Train VAE for anime face generation')

    # Data paths
    parser.add_argument('--train-path', type=str, default='data/train',
                        help='Path to training images directory')
    parser.add_argument('--test-path', type=str, default='data/test',
                        help='Path to test images directory')
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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--lr-scheduler', type=str, default=None,
                        help='LR scheduler: exp (ExponentialLR), smart (ReduceLROnPlateau), or None (default: None)')
    parser.add_argument('--reduce-lr-factor', type=float, default=0.5,
                        help='Factor by which to reduce learning rate for ReduceLROnPlateau (default: 0.5)')
    parser.add_argument('--reduce-lr-patience', type=int, default=5,
                        help='Number of epochs with no improvement for ReduceLROnPlateau (default: 5)')
    parser.add_argument('--reduce-lr-verbose', action='store_true',
                        help='Print message when learning rate is reduced by ReduceLROnPlateau')
    parser.add_argument('--leaky-relu-slope', type=float, default=0.2,
                        help='Leaky ReLU slope (default: 0.2)')
    parser.add_argument('--sigma-sq', type=float, default=0.0001,
                        help='Variance parameter for reconstruction loss (default: 0.0001)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for KL divergence loss (default: 1.0)')
    parser.add_argument('--pi', type=float, default=1.0,
                        help='Weight for perceptual loss (default: 1.0)')
    parser.add_argument('--init-method', type=str, choices=['xavier', 'normal', 'kaiming_uniform'], default='kaiming_uniform',
                        help='Weight initialization method (default: kaiming_uniform)')
    parser.add_argument('--init-method-mode', type=str, choices=['fan_in', 'fan_out'], default='fan_in',
                        help='Kaiming initialization mode (default: fan_in)')
    parser.add_argument('--normal-init-mean', type=float, default=0,
                        help='Mean for normal initialization (default: 0)')
    parser.add_argument('--normal-init-std', type=float, default=0.02,
                        help='Standard deviation for normal initialization (default: 0.02)')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed for reproducibility (default: -1, no seed)')

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
    parser.add_argument('--progressbar', action='store_true',
                        help='Show progress bar during training instead of simple prints')

    args = parser.parse_args()

    # Random seed initialization
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Using random seed: {args.seed}")
    else:
        print("No random seed specified, using default randomness")

    # Import appropriate network module
    if False: # args.deeper:
        import nets_high as nets
        print("Using DEEP architecture (nets_high)")
        # Set default base_channels for deeper networks
        if args.base_channels is None:
            args.base_channels = 96
    else:
        import nets
        print("Using standard architecture (nets)")
        print("DEEP NETWORKS ARE DISABLED IN THIS VERSION")
        # Set default base_channels for standard networks
        if args.base_channels is None:
            args.base_channels = 64

    # Setup TensorBoard with unique run name
    writer = None
    if not args.no_tensorboard:
        # Create unique run name with timestamp and key parameters
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lat{args.latent_dim}_ch{args.base_channels}_lr{args.lr}"
        log_path = os.path.join(args.log_dir, run_name)
        writer = SummaryWriter(log_path)
        print(f"TensorBoard logging to: {log_path}")

        # Log all hyperparameters as text
        hparams_text = "\n".join([f"{k}: {v}" for k, v in sorted(vars(args).items())])
        writer.add_text('Hyperparameters', hparams_text, 0)

    # Calculate epoch length
    num_files = len([f for f in os.listdir(args.train_path)
                     if os.path.isfile(os.path.join(args.train_path, f))])
    epoch_len = num_files // args.batch_size
    print(f"Dataset: {num_files} images, {epoch_len} batches per epoch")

    # Create data loader
    data_loader = load_images(args.train_path, args.image_size, args.batch_size)

    # Create networks
    encoder, decoder = nets.create_vae(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        leaky_relu_slope=args.leaky_relu_slope
    )

    # Training state
    start_epoch = 0
    iteration = 0
    cold_start = True  # Track if this is a cold start

    # Auto-load latest checkpoint
    checkpoint = None
    if not args.no_auto_load:
        latest_model = get_latest_model(args.models_dir)
        if latest_model:
            print(f"Loading latest checkpoint: {latest_model}")
            checkpoint = torch.load(latest_model)
            start_epoch = checkpoint.get('epoch', 0)
            iteration = checkpoint.get('iteration', 0)
            cold_start = False
            print(f"Resumed from epoch {start_epoch}, iteration {iteration}")
        else:
            print(f"No checkpoint found in {args.models_dir}, starting from scratch")
    else:
        print("Auto-load disabled, starting from scratch")

    # Optimizers
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr if args.last_lr is None else args.last_lr,
        betas=(0.5, 0.999)
    )

    # Load optimizer state if resuming
    if checkpoint is not None:
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # LR decay scheduler
    scheduler = None
    if args.lr_scheduler == 'exp':
        total_steps = args.epochs * epoch_len
        gamma = np.exp(np.log(0.001) / total_steps)  # Final lr is 0.1% of initial
        scheduler = ExponentialLR(optimizer, gamma=gamma, last_epoch=iteration - 1 if iteration > 0 else -1)
        print(f"Using ExponentialLR scheduler (gamma={gamma:.6f})")
    elif args.lr_scheduler == 'smart':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.reduce_lr_factor,
                                     patience=args.reduce_lr_patience, verbose=args.reduce_lr_verbose)
        print(f"Using ReduceLROnPlateau scheduler (factor={args.reduce_lr_factor}, patience={args.reduce_lr_patience})")
    elif args.lr_scheduler is not None:
        raise ValueError(f"Unknown lr-scheduler: {args.lr_scheduler}. Use 'exp', 'smart', or leave empty.")

    # Initialize weights for cold start
    if cold_start:
        if args.init_method == 'normal':
            print("Initializing weights with Normal initialization")
            encoder.apply(lambda m: init_weights_normal(m, args))
            decoder.apply(lambda m: init_weights_normal(m, args))
        elif args.init_method == 'xavier':
            print("Initializing weights with Xavier initialization")
            encoder.apply(init_weights_xavier)
            decoder.apply(init_weights_xavier)
        elif args.init_method == 'kaiming_uniform':
            print("Initializing weights with Kaiming uniform initialization")
            encoder.apply(lambda m: init_weights_kaiming_uniform(m, args))
            decoder.apply(lambda m: init_weights_kaiming_uniform(m, args))
        else:
            raise ValueError(f"Unknown initialization method: {args.init_method}")

        # Save initial weights
        print("Saving initial weights...")
        save_checkpoint(encoder, decoder, optimizer, 0, 0, args, args.models_dir)

    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train for one epoch
        for batch_idx in tqdm(range(epoch_len)) if args.progressbar else range(epoch_len):
            # Get batch
            images = next(data_loader)
            images = torch.FloatTensor(images)
            if nets.USE_CUDA:
                images = images.cuda()

            # Forward pass and compute loss
            optimizer.zero_grad()
            total_loss, kl_loss, recon_loss, perc_loss, reconstructed = compute_vae_loss(
                encoder, decoder, images, args.sigma_sq, args=args
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Update learning rate
            if scheduler is not None:
                if args.lr_scheduler == 'exp':
                    scheduler.step()
                elif args.lr_scheduler == 'smart':
                    scheduler.step(total_loss)

            iteration += 1

            # Save checkpoint periodically
            if iteration % args.save_freq == 0:
                save_checkpoint(encoder, decoder, optimizer, epoch, iteration, args, args.models_dir)

            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/KL', kl_loss, iteration)
                writer.add_scalar('Loss/Reconstruction', recon_loss, iteration)
                writer.add_scalar('Loss/Perceptual', perc_loss, iteration)
                writer.add_scalar('Loss/Total', kl_loss + recon_loss + perc_loss, iteration)

            # Print losses
            if iteration % args.plot_freq == 0:
                print(f"  [{batch_idx}/{epoch_len}] KL: {kl_loss:.4f}, Recon: {recon_loss:.4f}, Perc: {perc_loss:.4f}")

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
