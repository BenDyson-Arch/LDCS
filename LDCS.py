#############################################################################
# Code developed by Benedict Dyson, affiliation: Griffith University        #
#############################################################################

import os
import matplotlib.pyplot as plt
from LDCS_utils import dcs
import cv2

# Load image
img_path = r"C:\Users\bened\Desktop\temp\Bobby_Kangaroo_6.JPG"
fg_path = 'fg_score.png'

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

local_dcs = dcs(
    img, 
    glob=False,  
    window_size_factor=8, 
    stride_factor=4,
    downscale_fact=2
    )

# Display results
_, axes = plt.subplots(1, 2, figsize=(20, 5))

axes[0].imshow(img)
axes[0].axis('off')

axes[1].imshow(local_dcs)
axes[1].axis('off')
plt.tight_layout()
plt.show()