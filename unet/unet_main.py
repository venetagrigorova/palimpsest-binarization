import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from patchify import patchify, unpatchify
import torch.nn.functional as F
import torch
import cv2
import numpy as np

from DIBCODataset import DIBCODataset
from UNet import UNet
from unet_train import train_model

# Display a batch of images and masks to confirm augmentation
def show_batch(data_loader):
    images, masks = next(iter(data_loader))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        axes[0, i].imshow(images[i].squeeze(0).numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(masks[i].squeeze(0).numpy(), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

def show_all_patches(data_loader, max_batches=None):
    """
    Display all image–mask pairs in the dataset (or just a few batches).
    Set max_batches to None to show everything.
    """
    count = 0
    for batch_idx, (images, masks) in enumerate(data_loader):
        batch_size = images.size(0)
        fig, axes = plt.subplots(2, batch_size, figsize=(3 * batch_size, 6))
        
        for i in range(batch_size):
            img = images[i].squeeze().cpu().numpy()
            mask = masks[i].squeeze().cpu().numpy()
            
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f"Image {count+i}")
            axes[0, i].axis('off')

            axes[1, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f"Mask {count+i}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        count += batch_size
        if max_batches and batch_idx + 1 >= max_batches:
            break
        
def compare_prediction_to_ground_truth(model, image, mask, device, patch_size=256, threshold=0.6):
    """
    Visually compare U-Net prediction vs ground truth for a full image.
    Assumes image and mask are numpy arrays (grayscale, normalized to 0-1).
    """
    model.eval()

    # Pad image and mask
    H, W = image.shape
    pad_H = (patch_size - H % patch_size) % patch_size
    pad_W = (patch_size - W % patch_size) % patch_size

    padded_image = np.pad(image, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=1.0)
    padded_mask  = np.pad(mask,  ((0, pad_H), (0, pad_W)), mode='constant', constant_values=1.0)

    # Patchify
    img_patches = patchify(padded_image, (patch_size, patch_size), step=patch_size)
    pred_patches = np.zeros_like(img_patches)

    with torch.no_grad():
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                patch = img_patches[i, j]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                pred = model(patch_tensor)
                pred = torch.sigmoid(pred).squeeze().cpu().numpy()
                
                # Log min/max/mean per patch
                print(f"Patch ({i},{j}) → pred min: {pred.min():.4f}, max: {pred.max():.4f}, mean: {pred.mean():.4f}")
                
                pred_patches[i, j] = pred

    # Unpatchify and crop back to original size
    full_pred = unpatchify(pred_patches, padded_image.shape)
    full_pred = full_pred[:H, :W]
    binary_pred = (full_pred >= threshold).astype(np.float32)
    
     # Additional full-image prediction logs
    print(f"\n🔍 Full Prediction Stats:")
    print(f"→ min: {full_pred.min():.4f}, max: {full_pred.max():.4f}, mean: {full_pred.mean():.4f}")
    print(f"→ Binary prediction (threshold={threshold}): unique values = {np.unique(binary_pred)}")

    # Show side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    axes[2].imshow(binary_pred, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Model Prediction")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def load_and_train():
    current_dir = os.path.dirname(__file__)
    # Set paths
    image_dir = os.path.join(current_dir, "data", "image")
    mask_dir = os.path.join(current_dir, "data", "mask")

    # Create dataset and data loader
    dataset = DIBCODataset(image_dir, mask_dir, patch_size=256)
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Show augmented batch
    #show_batch(data_loader)
    #show_all_patches(data_loader)
    
    # Your model
    model = UNet(in_channels=1, out_channels=1)
    model_path = os.path.join(current_dir, "unet_binarization_try1_epD.pth")
    # Use GPU if available
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # or 'cuda'
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train!
    model = train_model(model, data_loader, device, epochs=20)
    

def main():
    # load model from file
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "unet_binarization_try1_epDP3.pth")
    print("Looking for model file at:", model_path)
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # or 'cuda'
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Set model to evaluation mode
    model.eval()
    
    ## Evalute
    image_dir = os.path.join(current_dir, "data", "test", "image", "dibco_img0010.tif")
    mask_dir = os.path.join(current_dir, "data", "test", "mask", "dibco_img0010_gt.tif")
    image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE) / 255.0
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE) / 255.0
    
    compare_prediction_to_ground_truth(model, image, mask, device)
    
if __name__ == "__main__":
    main()

