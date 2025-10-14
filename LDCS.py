#############################################################################
# Code developed by Benedict Dyson, affiliation: Griffith University        #
#############################################################################

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import matplotlib.pyplot as plt
from LDCS_utils import dcs
import cv2

# Load image
img_path = r"C:\Users\bened\Desktop\temp\46_01.png"

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

local = dcs(
    img, 
    glob=False, 
    col_space="YWE", 
    window_size_factor=32, 
    stride_factor=4
    )

# Display results
fig, axes = plt.subplots(1, 2, figsize=(20, 5))

cv2.imwrite("l_dcs.jpg", cv2.cvtColor(local, cv2.COLOR_RGB2BGR))

axes[0].imshow(img)
axes[0].axis('off')

axes[1].imshow(local)
axes[1].axis('off')
plt.tight_layout()
plt.show()