#############################################################################
# Code developed by Benedict Dyson, affiliation: Griffith University        #
# Paper DOI:                                                                #
#############################################################################

import cv2
import numpy as np
import concurrent.futures
from tqdm import tqdm
import threading

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

def global_dcs(image: np.ndarray, is_global: bool = True, gamma: bool = True) -> np.ndarray:
    """
    Perform a global decorrelation stretch on `image` in the given colour space,
    using custom stretch factors per channel from factor_dict[col_space].
    """
    h, w, d = image.shape

    # Convert to colour space once if global, otherwise assume already in target space
    trans = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) if is_global else image

    flat = trans.reshape(-1, d)

    L = flat[:,0].copy().reshape(h,w)
    AB = flat[:,1:3]

    mean = AB.mean(axis=0)
    centered = AB - mean

    cov = np.cov(centered, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)

    # Decorrelate
    decor = centered @ eigvecs

    for i in range(decor.shape[1]):
        decor[:, i] = stretch_channel(decor[:, i], stretch_factor=1.0)

    recon_ab = (decor @ eigvecs.T) + mean
    recon = np.zeros_like(flat)
    recon[:,0] = L.flatten()
    recon[:,1:3] = recon_ab[:, :2]
    result = recon.reshape((h, w, d)) 

    for i in range(d):
        ch = result[:, :, i]
        mn, mx = ch.min(), ch.max()
        if mx > mn: # Ensures no division by zero
            result[:, :, i] = 255 * ((ch - mn) / (mx - mn))
        else:
            result[:, :, i] = 0
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Convert back to RGB only if we did the initial conversion
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB) if is_global else result


def process_patch(img: np.ndarray, row: int, col: int, window: int, d: int):
    """
    Process one window patch: apply decorrelation with no re-conversion
    and return (row, col, stretched_patch, weight_map).
    """
    patch = img[row:row+window, col:col+window]

    # Decorrelate-stretch
    stretched = global_dcs(patch, is_global=False).astype(np.float32)

    weight = np.ones((window, window), dtype=np.float32)

    return row, col, stretched, weight


def local_dcs(img: np.ndarray, window_size_factor: int = 8, stride_factor: int = 2) -> np.ndarray:
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

    cs = cv2.cvtColor(padded, cv2.COLOR_RGB2LAB)
    h, w, d = cs.shape

    # Prepare accumulators
    output = np.zeros((h, w, d), dtype=np.float32)
    weight_map = np.zeros((h, w),    dtype=np.float32)

    # All window top-left coordinates
    coords = [(r, c) for r in range(0, h - window + 1, stride)
                  for c in range(0, w - window + 1, stride)
                  if r + window <= h and c + window <= w]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_patch, cs, r, c, window, d)
            for r, c in coords
        ]

        results = []
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing patches"):
            results.append(fut.result())

    # Accumulate after all processing
    for rr, cc, patch, wmap in results:
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
    
    result = cv2.cvtColor(cropped.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return np.clip(result, 0, 255).astype(np.uint8)



def dcs(image: np.ndarray, glob: bool = True, window_size_factor: int = 8, stride_factor:int = 2, downscale_fact:float = 0.0):
    """
    Perform decorrelation stretch on the input image.
    
    Parameters:
    ---------
    image: numpy array
        Input image to be processed.
    glob: bool
        If True, perform global decorrelation stretch. If False, perform local decorrelation stretch.
    window_size_factor: int
        Only for local. Factor to determine the size of the window for local processing.
    stride_factor: int
        Only for local. Factor to determine the stride for moving the window across the image.

    Returns:
    --------
    numpy array
        Decorrelated and stretched image.
    """

    # Downscale image
    if downscale_fact > 0.0:
        image = cv2.resize(image, (0,0), fx=1/downscale_fact, fy=1/downscale_fact)
        print(f"Downscaled image to {image.shape[1]}x{image.shape[0]} for processing")

    # Create mask of non-zero pixels (handles orthophotos better)
    mask = np.any(image != [0, 0, 0], axis=-1)

    # Perform DCS
    if glob:
        print("Initialising global decorrelation stretch...")
        stretched = global_dcs(image, is_global=True)
        print("Global decorrelation complete")
    else:
        print("Initialising local decorrelation stretch...")
        stretched = local_dcs(image, window_size_factor, stride_factor)
        print("Local decorrelation complete")

    # Restore black pixels so they remain exactly (0,0,0)
    stretched[~mask] = image[~mask]

    return stretched



def plot_channels(array):
    import matplotlib.pyplot as plt
    # Extract individual channels
    array_dim = array.shape[2]
    
    _, axes = plt.subplots(1, array_dim, figsize=((array_dim * 5), 5))
    
    # Ensure axes is always an array for indexing
    axes = np.array(axes).flatten()
    
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
    
    _, axes = plt.subplots(1, array_dim, figsize=((array_dim * 5), 5))
    
    # Ensure axes is always an array for indexing
    axes = np.array(axes).flatten()

    for i in range(array_dim):
        axes[i].hist(array[..., i].ravel(), bins=256, color='gray', alpha=0.7)
        axes[i].set_title(f'Histogram of Channel {i+1}')
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()