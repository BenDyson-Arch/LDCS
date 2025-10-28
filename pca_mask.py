def analyze_image_with_pca(image_path, fg_path, channels_to_remove, n_components_ica=3, visualize=True):
    """
    Analyze image using PCA and ICA for channel reduction and independent component analysis.
    
    Parameters:
    - image_path: str, path to the main image
    - fg_path: str, path to the foreground score image
    - channels_to_remove: list of int, channels to remove (1-4, 1 or 2 channels)
    - n_components_ica: int, number of ICA components (default 3)
    - visualize: bool, whether to show plots (default True)
    
    Returns:
    - reconstructed_removed_normalised: np.ndarray, 3-channel reconstructed image
    """
    # Load images
    fg_image = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
    if len(fg_image.shape) == 3:
        fg_image = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)
    # Now fg_image is grayscale (H, W)
    
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Adaptive kernel size for blurring based on image dimensions
    kernel_h = max(3, h // 10)  # Ensure odd
    if kernel_h % 2 == 0:
        kernel_h += 1
    kernel_w = max(3, w // 50)
    if kernel_w % 2 == 0:
        kernel_w += 1
    kernel_size = (kernel_h, kernel_w)

    # Apply Gaussian blur to fg with adaptive kernel
    fg_image = cv2.GaussianBlur(fg_image, kernel_size, 0)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Gamma filter to enhance contrast
    image = exposure.adjust_log(image, 1).astype(np.uint8)
    
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Normalise channels to 0-1
    fg_normalised = cv2.normalize(fg_image.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    img_normalised = cv2.normalize(image_lab.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Stack images into 4D image
    stacked_image = np.dstack((img_normalised, fg_normalised))
    
    # Use provided channels to remove
    num_to_remove = len(channels_to_remove)
    if num_to_remove not in [1, 2]:
        raise ValueError("channels_to_remove must contain 1 or 2 channels")
    
    # Remove selected channels
    indices_to_remove = sorted([c - 1 for c in channels_to_remove], reverse=True)  # remove from highest index
    reconstructed_removed = stacked_image.copy()
    for idx in indices_to_remove:
        reconstructed_removed = np.delete(reconstructed_removed, idx, axis=2)
    
    # If only 2 channels, add a third channel with 0.5
    if reconstructed_removed.shape[2] == 2:
        third_channel = np.full((reconstructed_removed.shape[0], reconstructed_removed.shape[1], 1), 0.5, dtype=reconstructed_removed.dtype)
        reconstructed_removed = np.concatenate([reconstructed_removed, third_channel], axis=2)
    
    # Normalise again
    reconstructed_removed_normalised = (reconstructed_removed - np.min(reconstructed_removed)) / (np.max(reconstructed_removed) - np.min(reconstructed_removed))
    
    # Convert to RGB if it's 3 channels (assuming LAB)
    if reconstructed_removed_normalised.shape[2] == 3:
        reconstructed_removed_normalised = cv2.cvtColor((reconstructed_removed_normalised * 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    image_lab[:,:, 0] = 0.5 * 255

    img_fg_dcs = dcs(reconstructed_removed_normalised, 
                     glob=False, 
                     window_size_factor=14,
                     stride_factor=4,
                     downscale_fact=2)
    img_plain_dcs = dcs(image, 
                        glob=False, 
                        window_size_factor=14,
                        stride_factor=4,
                        downscale_fact=2)

    if visualize:
        # Visualise as color image
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img_plain_dcs)
        plt.title("DCS of Original Image")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(img_fg_dcs)
        plt.title("DCS of PCA/ICA Processed Image")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return reconstructed_removed_normalised

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
import os
import glob
from LDCS_utils import dcs

if __name__ == "__main__":
    # Directories
    fg_dir = "fg_scores"
    image_dir = "images"  # Current directory for images
    
    # Loop over all fg files matching *_fg.*
    for fg_path in glob.glob(os.path.join(fg_dir, "*_fg.*")):
        fg_file = os.path.basename(fg_path)
        
        # Extract base name (before _fg)
        if '_fg.' in fg_file:
            base = fg_file.rsplit('_fg.', 1)[0]
            
            # Find corresponding image file with any extension
            image_pattern = os.path.join(image_dir, f"{base}.*")
            image_matches = glob.glob(image_pattern)
            
            if len(image_matches) == 1:
                image_path = image_matches[0]
                print(f"Processing {os.path.basename(image_path)} with {fg_file}")
                # Run analysis, removing channel 4 (fg)
                reconstructed = analyze_image_with_pca(image_path, fg_path, [1, 4])
            elif len(image_matches) > 1:
                print(f"Multiple images found for {base}: {image_matches}")
            else:
                print(f"No image found for {base} (from {fg_file})")