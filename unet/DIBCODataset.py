import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch as torch
import random
import torchvision.transforms.functional as TF
from patchify import patchify

class DIBCODataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.augmentation_pipeline = transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25
        )
        self.patches = []
        self.load_and_patchify()
        
    def load_and_patchify(self):
        img_files = sorted(os.listdir(self.image_dir))
        mask_files = sorted(os.listdir(self.mask_dir))

        for img_file, mask_file in zip(img_files, mask_files):
            img = cv2.imread(os.path.join(self.image_dir, img_file), cv2.IMREAD_GRAYSCALE)
            if img.max() > 1:
                img = img / 255.0
            mask = cv2.imread(os.path.join(self.mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
            if mask.max() > 1:
                mask = mask / 255.0
            
            if img.shape != mask.shape:
                print(f"Warning: Shape mismatch for {img_file}, resizing mask to match image")
                #mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                continue
            
            # Pad to make divisible by patch size
            H, W = img.shape
            pad_H = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_W = (self.patch_size - W % self.patch_size) % self.patch_size
            
            img = np.pad(img, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=1.0)
            mask = np.pad(mask, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=1.0)
            

            # Check if the image and mask are still the same size
            # Patchify
            img_patches = patchify(img, (self.patch_size, self.patch_size), step=self.patch_size)
            mask_patches = patchify(mask, (self.patch_size, self.patch_size), step=self.patch_size)
            
            # Flatten and store
            for i in range(img_patches.shape[0]):
                for j in range(img_patches.shape[1]):
                    if np.all(mask_patches[i, j] == 1.0):
                        #print(f"Skip pure background patch")
                        continue
                    self.patches.append((
                        img_patches[i, j],  # shape: (256, 256)
                        mask_patches[i, j]  # shape: (256, 256)
                    ))

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, index):
        image, mask = self.patches[index]

        # Convert to PIL Image for torchvision.transforms
        image = TF.to_pil_image((image * 255).astype(np.uint8))
        mask = TF.to_pil_image((mask * 255).astype(np.uint8))

        # ---------- Augmentations ----------
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            gaussian = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
            image = gaussian(image)

        image = self.augmentation_pipeline(image)
        # ------------------------------------

        # Convert to tensor and normalize
        image = TF.to_tensor(image)  # Range [0,1], shape (1, H, W)
        mask = TF.to_tensor(mask)
        
        return image, mask
