# LDCS

DStretch is widely known within the study of rock art and is used across the field of archaeology to enhance the visibility of faded or difficult to see motifs. DStretch calculates enhancements based on all RGB values in an image and so often struggles to decorrelate local areas with high correlation, particularly if there is a wide distribution of RGB values such as in dense rock art scenes, orthomosaics, or 3D model textures. Local decorrelation stretch (L-DCS) automatically separates an image into overlapping windows, performing decorrelated stretching on each, and merging them back into a single image. The results retained greater fine details in motifs and improved significantly the visibility of "hidden" motifs in densely painted areas when compared to DStretch. The flexibility afforded by our approach and the consistency of results in different scenarios make L-DCS a robust approach for automated DCS on larger datasets without manual intervention. 

## Installation

### Prerequisites
- Git
- Anaconda3 or Miniconda

### Dependencies
- Python environment (provided via conda)
- Tested on Windows (may work on other operating systems)

### Download from GitHub
Navigate in the terminal to where you want to install the repo. Then enter the command:
```
git clone https://github.com/BenDyson-Arch/LDCS
```

### Create the environment
Following successful cloning of the repo, cd to the downloaded folder:
```
cd LDCS
```

And set up the conda environment:
```
conda env create -f environment.yml
```

Then activate the environment:
```
conda activate ldcs
```

You have now downloaded and installed LDCS successfully.

## Use

First, import the function from the LDCS_utils module:

```python
from LDCS_utils import dcs
```

The `dcs` function can perform global or local decorrelation stretching on images.

### Example Usage

```python
# For local decorrelation stretching
local_dcs = dcs(
    img, 
    glob=False,  
    window_size_factor=8, 
    stride_factor=4,
    downscale_fact=2.0
)
```

### Parameters
* `img` (numpy array) - A numpy array of the image you want to process
* `glob` (bool) - When set to `True` will perform global decorrelation. When set to `False` will perform local decorrelation
* `window_size_factor` (int) - The factor with which you want to split your image horizontally. Recommend choosing smaller values (<4) for less dense rock art images and higher numbers (>8) for denser images
* `stride_factor` (int) - The factor to determine the overlap between windows. This is essential in reducing seams between windows
* `downscale_fact` (float) - An optional utility to make downscaling images easier to speed up processing

## Contributing
Contributions to the repo are welcome. If you wish to contribute please fork the repository and create a descriptively named branch (e.g. `fix/img-read-error` or `feature/multithreading`). Test any changes thoroughly.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this software in your research, please cite it as follows:

**Plain text:**
Dyson, B. (2025). LDCS [Computer software]. https://github.com/BenDyson-Arch/LDCS

**BibTeX:**
```bibtex
@article{DYSON2026e00522,
  title = {Localised decorrelation stretch (L-DCS) for improved visibility of large, dense rock art scenes},
  journal = {Digital Applications in Archaeology and Cultural Heritage},
  year = {2026},
  doi = {https://doi.org/10.1016/j.daach.2026.e00522},
  author = {Benedict Dyson and Andrea Jalandoni and Paul Taçon}
```
