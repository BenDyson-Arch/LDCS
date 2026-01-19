#############################################################################
# Code developed by Benedict Dyson, affiliation: Griffith University        #
# Paper DOI:                                                                #
#############################################################################

import matplotlib.pyplot as plt
from LDCS_utils import dcs
import cv2

if __name__ == '__main__':
    # Load image
    img_path = r'your_image_path_here.jpg'  # Replace with your image path

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Image not found at {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB


    local_dcs = dcs(
        img, 
        glob=False,  
        window_size_factor=8, 
        stride_factor=4,
        downscale_fact=2.0
    )

    # Display results
    _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].imshow(img)
    axes[0].axis('off')

    axes[1].imshow(local_dcs)
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()