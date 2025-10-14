#############################################################################
# Code developed by Benedict Dyson, affiliation: Griffith University        #
# Paper DOI:                                                                #
#############################################################################

import cv2
import numpy as np
from PIL import Image, ImageCms
import concurrent.futures
from tqdm import tqdm
import threading
from scipy.ndimage import uniform_filter

def rgb_to_colspace(image, base_space="L"):
    """
    Convert RGB image to specified color space.
    
    Parameters:
    ---------
    image: numpy array
        Input image to be processed.
    col_space: str
        Color space to be used for decorrelation stretch.
        
    Returns:
    --------
    numpy array
        Image converted to the specified color space.
    """
    if base_space == "Y":
        trans_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif base_space == "L":
        trans_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    elif base_space == "H":
        trans_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        trans_img[..., 0] = rotate_hue_to_fill(trans_img[..., 0])
    elif base_space == "S":
        trans_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        trans_img[..., 0] = rotate_hue_to_fill(trans_img[..., 0])
    elif base_space == "C":
        pil_image = Image.fromarray(image)
        rgb_profile = "iccProfiles\sRGB2014.icc"
        cmyk_profile = "iccProfiles\Coated_Fogra39L_VIGC_300.icc"
        image = ImageCms.profileToProfile(
            pil_image, 
            rgb_profile, 
            cmyk_profile, 
            outputMode='CMYK'
            )
        trans_img = np.array(image)
    else:
        trans_img = image
    #plot_channel_histograms(trans_img)
    return trans_img

def colspace_to_rgb(image, base_space="L"):
    """
    Convert image from specified color space back to RGB.
    
    Parameters:
    ---------
    image: numpy array
        Input image to be processed.
    col_space: str
        Color space to be used for decorrelation stretch.
        
    Returns:
    --------
    numpy array
        Image converted back to RGB color space.
    """
    if base_space == "Y":
        result_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    elif base_space == "L":
        result_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_LAB2RGB)
    elif base_space == "H":
        result_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB)   
    elif base_space == "S":
        result_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HLS2RGB)
    elif base_space == "C":
        result = Image.fromarray(image.astype(np.uint8), "CMYK")
        rgb_profile = "iccProfiles\sRGB2014.icc"
        cmyk_profile = "iccProfiles\Coated_Fogra39L_VIGC_300.icc"
        cmyk2rgb_transform = ImageCms.buildTransform(
            cmyk_profile, 
            rgb_profile, 
            'CMYK', 
            'RGB'
            )
        rgb_image = ImageCms.applyTransform(result, cmyk2rgb_transform)        
        result_rgb = np.array(rgb_image)
    else:
        result_rgb = image

    return result_rgb

def stretch_channel(image_channel, stretch_factor):
    """
    Stretch a single channel of the image using the specified stretch factor.
    
    Parameters:
    ---------
    image_channel: numpy array
        Input image channel to be processed.
    stretch_factor: float
        Stretch factor to be applied to the channel.
        
    Returns:
    --------
    numpy array
        Stretched image channel.
    """
    p_low, p_high = np.percentile(image_channel, [2, 98])
    
    # Ensure that the stretch factor is not zero, otherwise skip
    if stretch_factor != 0.0:
        stretched_channel = np.interp(
            image_channel,
            [p_low, p_high],
            [p_low - stretch_factor*(p_high-p_low)/2,
             p_high + stretch_factor*(p_high-p_low)/2]
        )
    return stretched_channel

def rotate_hue_to_fill(h):
    """
    Given a 2D array of hues in [0,360), find the largest empty gap
    in the circular histogram and rotate all hues so that this gap
    straddles the 360°→0° boundary.
    """
    # Flatten and sort
    hs = np.sort(h.ravel())
    # Compute consecutive gaps, including wrap‑around from 360→0
    diffs = np.diff(np.concatenate([hs, hs[:1] + 360.0]))
    # Index of largest gap
    idx = np.argmax(diffs)
    # Start of the gap
    H_lo = hs[idx] % 360.0
    # Rotate all hues so H_lo maps to 0
    h_rot = np.mod(h - H_lo + 360.0, 360.0)
    return h_rot


# Process-local scratch buffers via threading.local (works per process)
_thread_local = threading.local()

def global_dcs(image: np.ndarray,
               factor_dict: dict,
               col_space: str = "LAB",
               is_global: bool = True) -> np.ndarray:
    """
    Perform a global decorrelation stretch on `image` in the given colour space,
    using custom stretch factors per channel from factor_dict[col_space].
    """
    base_space = col_space[0]
    factors = factor_dict[col_space]
    h, w, d = image.shape

    # Convert to colour space once if global, otherwise assume already in target space
    trans = rgb_to_colspace(image, base_space) if is_global else image

    # Flatten and center
    flat = trans.reshape(-1, d).astype(np.float32)
    mean = flat.mean(axis=0)
    centered = flat - mean

    # Covariance and eigendecomposition
    cov = np.cov(centered, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)

    # Decorrelate
    decor = centered @ eigvecs

    # Stretch each decorrelated channel
    if len(factors) > 2:
        for i, f in enumerate(factors):
            decor[:, i] = stretch_channel(decor[:, i], f)
    else:
        decor[:, 0] = stretch_channel(decor[:, 0], factors[0])

    # Reconstruct, reshape, and normalize each channel
    recon = (decor @ eigvecs.T) + mean
    result = recon.reshape(h, w, d)
    for i in range(d):
        ch = result[:, :, i]
        mn, mx = ch.min(), ch.max()
        if mx > mn: # Ensures no division by zero
            result[:, :, i] = 255 * ((ch - mn) / (mx - mn))
        else:
            result[:, :, i] = 0
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Convert back to RGB only if we did the initial conversion
    return colspace_to_rgb(result, base_space) if is_global else result


def process_patch(img: np.ndarray, row: int, col: int, window: int, d: int, factor_dict: dict, col_space: str):
    """
    Process one window patch: apply decorrelation with no re-conversion
    and return (row, col, stretched_patch, weight_map).
    """
    tl = _thread_local
    if not hasattr(tl, 'scratch'):
        tl.scratch = np.empty((window, window, d), dtype=np.float32)
        tl.weight  = np.ones((window, window),    dtype=np.float32)

    patch = img[row:row+window, col:col+window]

    # Decorrelate-stretch in-place on scratch buffer
    stretched = global_dcs(patch, factor_dict, col_space, is_global=False).astype(np.float32)

    np.copyto(tl.scratch, stretched)
    return row, col, tl.scratch, tl.weight


def local_dcs(img: np.ndarray, factor_dict: dict, col_space: str = "LAB",
              window_size_factor: int = 8, stride_factor: int = 2) -> np.ndarray:
    """
    Perform a local decorrelation stretch:
      1. Pad & convert full image once.
      2. Slide a window over the image (with given factor/stride).
      3. Use a ProcessPoolExecutor to process patches, throttling to 2× workers.
      4. Accumulate weighted outputs, normalize, and convert back.
    """
    orig_h, orig_w = img.shape[:2]
    window = max(1, orig_w // window_size_factor)
    stride = max(1, window // stride_factor)

    # Pad to multiples of window
    pad_y = (window - (orig_h % window)) % window
    # Avoid areas not being covered by windows when orig_w is not divisible by window
    pad_x = stride

    padded = cv2.copyMakeBorder(img, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT)    

    # Single colour-space conversion
    base_space = col_space[0]
    cs = rgb_to_colspace(padded, base_space)
    h, w, d = cs.shape

    # Prepare accumulators
    output = np.zeros((h, w, d), dtype=np.float32)
    weight_map = np.zeros((h, w),    dtype=np.float32)

    # All window top-left coordinates
    coords = [(r, c) for r in range(0, h - window + 1, stride)
                  for c in range(0, w - window + 1, stride)
                  if r + window <= h and c + window <= w]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_patch, cs, r, c, window, d, factor_dict, col_space)
            for r, c in coords
        ]

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Merging patches"):
            rr, cc, patch, wmap = fut.result()
            output[rr:rr+window, cc:cc+window] += patch
            weight_map[rr:rr+window, cc:cc+window] += wmap

    # Normalize and convert back to RGB
    weight_map[weight_map == 0] = 1
    output /= weight_map[..., None]
    cropped = output[:orig_h, :orig_w]

    for i in range(d):
        ch = cropped[:, :, i]
        mn, mx = ch.min(), ch.max()
        if mx > mn: # Ensures no division by zero
            cropped[:, :, i] = 255 * ((ch - mn) / (mx - mn))
        else:
            cropped[:, :, i] = 0
    
    result = colspace_to_rgb(cropped, base_space)
    return np.clip(result, 0, 255).astype(np.uint8)



def dcs(image: np.ndarray, glob: bool = True, col_space: str = "LAB", window_size_factor: int = 8, stride_factor:int = 2, downscale_fact:float = None):
    """
    Perform decorrelation stretch on the input image.
    
    Parameters:
    ---------
    image: numpy array
        Input image to be processed.
    glob: bool
        If True, perform global decorrelation stretch. If False, perform local decorrelation stretch.
    col_space: str
        Color space to be used for decorrelation stretch. Default is LAB.
    window_size_factor: int
        Only for local. Factor to determine the size of the window for local processing.
    stride_factor: int
        Only for local. Factor to determine the stride for moving the window across the image.

    Returns:
    --------
    numpy array
        Decorrelated and stretched image.
    """

    dict = {
        "LAB": (1, 2, 2),
        "LL": (10, 1, 1),
        "LY": (1, 5, 2), # Emphasises yellow in an image
        "LR": (1, 2, 5), # Emphasises red in an image
        # HSV colour space
        "HSV": (1, 1, 1),
        "HH": (10, 1, 5), # Good for line detection (differences in hue)
        "HS": (5, 10, 1), # Emphasises differences in saturation
        "HV": (1, 2, 10),
        "HSV": (1, 5, 5),
        # YCbCr colour space
        "YRB": (1, 1, 1), 
        "YWE": (1, 2, 2),
        "YB": (1, 2, 10), # Retains good visual commonality with original image
        "YW": (1, 0.5, 5),
        # HLS colour space
        "SH": (10, 1, 5),
        # CMYK colour space. WARNING: This filter takes a longer time to compute. Only accepts 8bit images
        "CRY": (1, 5, 5, 2),  # Emphasises red and yellow in an image
        "CK": (1, 1, 1, 5),  # Emphasises black in an image

        "BW": (1,1,1)
        }

    # Downscale image
    if downscale_fact:
        img = cv2.resize(img, (0,0), fx=1/downscale_fact, fy=1/downscale_fact)

    # Create mask of non-zero pixels (handles orthophotos better)
    mask = np.any(image != [0, 0, 0], axis=-1)

    # Perform DCS
    if glob:
        print("Initialising global decorrelation stretch...")
        stretched = global_dcs(image, dict, col_space)
        print("Global decorrelation complete")
    else:
        print("Initialising local decorrelation stretch...")
        stretched = local_dcs(image, dict, col_space,
                              window_size_factor, stride_factor)
        print("Local decorrelation complete")

    # Restore black pixels so they remain exactly (0,0,0)
    stretched[~mask] = image[~mask]

    return stretched

def compute_tpi(depth_map: np.ndarray, window_size: int = 25) -> np.ndarray:
    """
    Compute the Terrain Position Index (TPI) from a depth map with missing data gaps.
    Missing data (assumed zero or negative) is infilled using OpenCV inpainting before TPI.

    Parameters:
    ---------
    depth_map: numpy array (2D float)
        Input depth map to be processed.
    window_size: int
        Size of the window for computing TPI.

    Returns:
    --------
    numpy array
        TPI computed from the infilled depth map, normalized to [0,1].
    """
    print("Preparing mask for inpainting...")
    # Create mask of missing pixels (depth <= 0)
    mask = (depth_map <= 0).astype(np.uint8)

    if np.count_nonzero(mask) > 0:
        print(f"Inpainting {np.count_nonzero(mask)} missing pixels using OpenCV...")

        # Normalize depth map to 0-255 uint8 for inpainting
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min == 0:
            print("Warning: depth_map has no variation.")
            depth_norm = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            depth_norm = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

        # Inpaint with Telea method (can try INPAINT_NS too)
        inpainted = cv2.inpaint(depth_norm, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Convert back to float depth range
        filled_depth = inpainted.astype(np.float32) / 255 * (depth_max - depth_min) + depth_min
    else:
        print("No missing pixels detected, skipping inpainting.")
        filled_depth = depth_map.copy()

    print("Computing Terrain Position Index (TPI)...")

    # Mask for valid pixels (after fill, all should be valid)
    valid_mask = filled_depth > 0

    depth = filled_depth.copy()
    depth[~valid_mask] = 0.0  # just in case

    # Compute local sum and valid count using uniform filter
    local_sum = uniform_filter(depth, size=window_size, mode='constant', cval=0.0)
    valid_count = uniform_filter(valid_mask.astype(np.float32), size=window_size, mode='constant', cval=0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        local_mean = local_sum / valid_count
        tpi = depth - local_mean
        tpi[valid_count == 0] = 0

    # Normalize TPI to [0,1]
    tpi_min, tpi_max = tpi.min(), tpi.max()
    if tpi_max - tpi_min > 0:
        tpi_remap = (tpi - tpi_min) / (tpi_max - tpi_min)
    else:
        tpi_remap = np.zeros_like(tpi)

    return tpi_remap



def plot_channels(array):
    import matplotlib.pyplot as plt
    # Extract individual channels
    array_dim = array.shape[2]
    
    fig, axes = plt.subplots(1, array_dim, figsize=((array_dim * 5), 5))

    for i in range(array_dim):
        axes[i].imshow(array[..., i], cmap='gray')
        axes[i].set_title(f'Channel {i+1}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()



def plot_channel_histograms(array):
    import matplotlib.pyplot as plt
    # Extract individual channels
    array_dim = array.shape[2]
    
    fig, axes = plt.subplots(1, array_dim, figsize=((array_dim * 5), 5))

    for i in range(array_dim):
        axes[i].hist(array[..., i].ravel(), bins=256, color='gray', alpha=0.7)
        axes[i].set_title(f'Histogram of Channel {i+1}')
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()