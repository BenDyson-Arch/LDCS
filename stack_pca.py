import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure

fg_image = cv2.imread("fg_score.jpg", cv2.IMREAD_UNCHANGED)
fg_image = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)

image = cv2.imread("Bobby_Kangaroo_6.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# gamma filter to enhance contrast
image = exposure.adjust_log(image, 1).astype(np.uint8)

# normalise channels to 0-1
fg_normalised = cv2.normalize(fg_image.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
img_normalised = cv2.normalize(image.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# stack images into 4D image
stacked_image = np.dstack((img_normalised, fg_normalised))

# verify stacking
print("Stacked image shape:", stacked_image.shape)  # Should be (H, W, 4)

# calculate pca across channels
mean = np.mean(stacked_image, axis=(0, 1))
centered = stacked_image - mean
covariance_matrix = np.cov(centered.reshape(-1, 4), rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# project data onto principal components
projected = centered.reshape(-1, 4) @ eigenvectors
reconstructed = projected.reshape(stacked_image.shape[0], stacked_image.shape[1], 4) + mean

# normalise reconstructed image to 0-1
reconstructed = (reconstructed - np.min(reconstructed)) / (np.max(reconstructed) - np.min(reconstructed))
reconstructed_gamma = exposure.adjust_log(reconstructed, 1)

# visualise each channel separately
fig, axes = plt.subplots(1, 4, figsize=(18, 6))
for i in range(4):
    axes[i].imshow(reconstructed_gamma[:, :, i], cmap='gray')
    axes[i].set_title(f"Channel {i+1}")
    axes[i].axis('off')
plt.show()

chanel_to_remove = None
while chanel_to_remove == None:
    chanel_to_remove = input("Enter channel number to remove (1-4): ")
    try:
        chanel_to_remove = int(chanel_to_remove)
        if chanel_to_remove < 1 or chanel_to_remove > 4:
            print("Please enter a valid channel number between 1 and 4.")
            chanel_to_remove = None
    except ValueError:
        print("Invalid input. Please enter a number between 1 and 4.")
        chanel_to_remove = None

chanel_index = chanel_to_remove - 1

# remove selected channel and return 3D image
reconstructed_removed = np.delete(reconstructed, chanel_index, axis=2)
print("Reconstructed image shape after removing channel:", reconstructed_removed.shape)

# Visualise as color image
reconstructed_removed_normalised = (reconstructed_removed - np.min(reconstructed_removed)) / (np.max(reconstructed_removed) - np.min(reconstructed_removed))
plt.imshow(reconstructed_removed_normalised)
plt.axis('off')
plt.show()