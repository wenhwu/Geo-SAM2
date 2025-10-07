
# Installation and Usage

**Note:** The following installation and usage instructions are specifically for the `Windows` operating system. Compatibility with other operating systems has not been verified at this time.

## Contents

- [Installation and Usage](#installation-and-usage)
  - [Contents](#contents)
  - [Installation](#installation)
    - [Install QGIS](#install-qgis)
    - [Required Dependencies for Installing Geo-SAM2](#required-dependencies-for-installing-geo-sam2)
    - [Download the Geo-SAM2 Plugin](#download-the-geo-sam2-plugin)
    - [Configure the Plugin Folder](#configure-the-plugin-folder)
    - [Activate the Geo-SAM2 Plugin](#activate-the-geo-sam2-plugin)
  - [Usage](#usage)
    - [Geo-SAM2 Image Encoder](#geo-sam2-image-encoder)
      - [Download SAM2 Checkpoints](#download-sam2-checkpoints)
      - [Running the Geo-SAM2 Image Encoder](#running-the-geo-sam2-image-encoder)
    - [Geo-SAM2 Interactive Segmentation](#geo-sam2-interactive-segmentation)


## Installation

This section provides a step-by-step guide for installing the Geo-SAM2 plugin for QGIS. Proper installation is crucial for the plugin to function correctly.

### Install QGIS

First, it is necessary to download and install the QGIS software. You can obtain the installer from the official QGIS website: [https://qgis.org/download/](https://qgis.org/download/).

**Version Requirement:** The installed version of QGIS must be `3.40.11` or higher. It is highly recommended to use the **Long Term Release (LTR)** version, specifically `3.40 LTR for Windows`. The LTR version provides enhanced stability and long-term support, which is ideal for academic and professional work.

### Required Dependencies for Installing Geo-SAM2

First, install NVIDIA **CUDA** and **cuDNN**.

Next, launch the **OSGeo4W Shell** from the Start menu **as Administrator**, this is the dedicated command-line environment for QGIS, and execute the following steps:

- Install `torch` and `torchvision` using the official PyTorch installation commands for specific versions: [Installing Previous Versions of PyTorch](https://pytorch.org/get-started/previous-versions/).  
- Install **SAM 2** following the instructions in the official repository: [SAM 2](https://github.com/facebookresearch/sam2).

Finally, install any additional dependencies as needed.  
- **Missing dependencies will typically be reported as import errors when running the plugin**; install them accordingly.  
- **For reference only**, the complete list of library versions used in my QGIS environment is provided below:

  ```
  Package                     Version            Editable project location
  --------------------------- ------------------ ----------------------------------------------------------------------
  absl-py                     2.3.1
  aenum                       3.1.15
  affine                      2.4.0
  aiohappyeyeballs            2.6.1
  aiohttp                     3.11.13
  aiosignal                   1.3.2
  annotated-types             0.7.0
  antlr4-python3-runtime      4.9.3
  asttokens                   2.4.1
  attrs                       25.2.0
  autocommand                 2.2.2
  backports.tarfile           1.2.0
  bitsandbytes                0.45.3
  black                       25.1.0
  certifi                     2024.2.2
  chardet                     5.2.0
  charset-normalizer          3.3.2
  click                       8.1.8
  click-plugins               1.1.1
  cligj                       0.7.2
  cloudpickle                 3.1.1
  colorama                    0.4.6
  coloredlogs                 15.0.1
  contourpy                   1.2.1
  cycler                      0.12.1
  cypari                      2.5.5
  decorator                   5.1.1
  docstring_parser            0.16
  efficientnet_pytorch        0.7.1
  einops                      0.8.1
  et_xmlfile                  1.1.0
  executing                   2.1.0
  ExifRead                    3.0.0
  filelock                    3.13.1
  fiona                       1.10.1
  flatbuffers                 25.2.10
  fonttools                   4.51.0
  frozenlist                  1.5.0
  fsspec                      2024.6.1
  future                      1.0.0
  FXrays                      1.3.5
  GDAL                        3.10.2
  geographiclib               2.0
  geopandas                   1.0.1
  grpcio                      1.74.0
  httplib2                    0.22.0
  huggingface-hub             0.29.3
  humanfriendly               10.0
  hydra-core                  1.3.2
  idna                        3.7
  importlib_metadata          8.0.0
  importlib_resources         6.5.2
  inflect                     7.3.1
  iopath                      0.1.10
  ipython                     8.28.0
  jaraco.collections          5.1.0
  jaraco.context              5.3.0
  jaraco.functools            4.0.1
  jaraco.text                 3.12.1
  jedi                        0.19.1
  Jinja2                      3.1.3
  jsonargparse                4.37.0
  kiwisolver                  1.4.5
  knot-floer-homology         1.2
  kornia                      0.8.0
  kornia_rs                   0.1.8
  lightly                     1.5.19
  lightly-utils               0.0.2
  lightning                   2.5.0.post0
  lightning-utilities         0.14.0
  low-index                   1.2
  lxml                        5.3.0
  Mako                        1.3.9
  Markdown                    3.8.2
  markdown-it-py              3.0.0
  MarkupSafe                  2.1.5
  matplotlib                  3.10.0
  matplotlib-inline           0.1.7
  mdurl                       0.1.2
  mock                        5.1.0
  more-itertools              10.3.0
  mpmath                      1.3.0
  multidict                   6.1.0
  munch                       4.0.0
  mypy_extensions             1.1.0
  networkx                    3.4.1
  ninja                       1.11.1.4
  nose2                       0.14.1
  numpy                       1.26.4
  omegaconf                   2.3.0
  opencv-python               4.8.0.74
  openpyxl                    3.1.2
  orjson                      3.11.2
  OWSLib                      0.32.0
  packaging                   25.0
  pandas                      2.2.2
  parso                       0.8.4
  pathspec                    0.12.1
  pillow                      11.1.0
  pip                         25.0
  platformdirs                4.3.6
  plink                       2.4.2
  plotly                      5.20.0
  ply                         3.11
  portalocker                 3.2.0
  pretrainedmodels            0.7.4
  prompt_toolkit              3.0.48
  propcache                   0.3.0
  protobuf                    6.30.0
  psycopg2                    2.9.10
  pure_eval                   0.2.3
  pyarrow                     18.1.0
  pycocotools                 2.0.10
  pycuda                      2025.1
  pydantic                    2.10.6
  pydantic_core               2.27.2
  Pygments                    2.17.2
  pyodbc                      5.1.0
  pyogrio                     0.9.0
  PyOpenGL                    3.1.7
  pyparsing                   3.1.2
  PyPDF2                      3.0.1
  pypiwin32                   223
  pypng                       0.20220715.0
  pyproj                      3.7.0
  PyQt5                       5.15.11
  PyQt5_sip                   12.16.1
  pyreadline3                 3.5.4
  pyserial                    3.5
  python-dateutil             2.9.0.post0
  pytools                     2025.1.1
  pytz                        2024.1
  pyvers                      0.1.0
  pywin32                     306
  PyYAML                      6.0.1
  rasterio                    1.4.3
  remotior_sensus             0.4.4
  reportlab                   4.2.5
  requests                    2.31.0
  rich                        13.9.4
  rtree                       1.4.1
  safetensors                 0.5.3
  SAM-2                       1.0                C:\Program Files\QGIS 3.40.4\apps\Python312\Lib\site-packages\sam2
  scipy                       1.13.0
  segment-anything            1.0
  segmentation_models_pytorch 0.4.0
  setuptools                  75.8.0
  shapely                     2.0.6
  simplejson                  3.19.2
  sip                         6.9.1
  six                         1.16.0
  snappy                      3.1.1
  snappy-manifolds            1.2
  spherogram                  2.2.1
  stack-data                  0.6.3
  sympy                       1.13.1
  tabulate                    0.9.0
  tenacity                    8.2.3
  tensorboard                 2.20.0
  tensorboard-data-server     0.7.2
  tensorboardX                2.6.2.2
  tensordict                  0.9.1
  tensorrt                    10.0.1
  termcolor                   3.1.0
  timm                        1.0.15
  tomli                       2.0.1
  torch                       2.6.0+cu118
  torchmetrics                1.6.2
  torchvision                 0.21.0+cu118
  tqdm                        4.67.1
  traitlets                   5.14.3
  typeguard                   4.3.0
  typeshed_client             2.7.0
  typing_extensions           4.12.2
  tzdata                      2024.1
  urllib3                     2.2.1
  wcwidth                     0.2.13
  Werkzeug                    3.1.3
  wheel                       0.43.0
  wxPython                    4.2.2
  xlrd                        2.0.1
  xlwt                        1.3.0
  yacs                        0.1.8
  yarl                        1.18.3
  zipp                        3.19.2
  ```


### Download the Geo-SAM2 Plugin

The plugin can be downloaded directly from its official GitHub repository: [https://github.com/wenhwu/Geo-SAM2](https://github.com/wenhwu/Geo-SAM2).

On the GitHub page, locate the "Source code" (usually in a compressed format like `.zip`) and download it. After downloading, decompress the file and rename the resulting folder to `Geo-SAM2`.

### Configure the Plugin Folder

Once the plugin is downloaded and the folder is correctly named, you must place the entire `Geo-SAM2` folder into the QGIS `plugins` directory (like `C:/Users/xxx/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM2`). The correct directory structure should be as follows:

```
    python
    └── plugins
        └── Geo-SAM2
            ├── checkpoint
            ├── docs
            ├── ...
            ├── tools
            └── ui
```

**Note**:

1.  **Folder Naming:** It is critical to rename the folder to `Geo-SAM2` after decompression. The plugin will not be recognized by QGIS if the folder name is incorrect (e.g., `Geo-SAM2-main`).
2.  **Avoid Nested Folders:** Be cautious of nested folders that can be created during decompression. An incorrect structure like `plugins/Geo-SAM2/Geo-SAM2/...` will prevent the plugin from loading. If this occurs, you must move the inner `Geo-SAM2` folder into the `plugins` directory.

### Activate the Geo-SAM2 Plugin

After placing the plugin in the correct directory, restart QGIS to allow it to detect the new plugin.

1.  Navigate to the menu: `Plugins` \> `Manage and Install Plugins`.
2.  In the `Installed` tab, you should now see the `Geo-SAM2` plugin listed.
3.  Check the box next to it to activate the plugin.

![](source/img/Geo-SAM2_Plugin_Manage.jpg)

Once activated, the Geo-SAM2 tools will be accessible from the `Plugins` menu.

![](source/img/geo_sam2_plugin_menu.png)

Additionally, a new toolbar with `two icons` for quick access to the Geo-SAM2 tools will appear in the QGIS interface.

![](source/img/geo-sam2-icon.png)

## Usage

This section explains how to use the core functionalities of the Geo-SAM2 plugin.

### Geo-SAM2 Image Encoder

The Image Encoder is a preliminary step that processes the image to generate embeddings, which are necessary for the interactive segmentation tool.

#### Download SAM2 Checkpoints

Before using the encoder, you must download the pre-trained model checkpoints. These checkpoints are essential as they contain the learned weights of the SAM2 model. Download the following files and place them in the `./plugins/Geo-SAM2/checkpoints` directory:

  - [sam2.1\_hiera\_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
  - [sam2.1\_hiera\_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)

*Note: The `large` model provides higher accuracy but requires more computational resources, while the `tiny` model is faster and less resource-intensive.*

#### Running the Geo-SAM2 Image Encoder

The following animation shows the process of running the image encoder.

<p align="center">
  <img src="source/img/geo_sam2_image_encoder.gif" width="1000" title="Geo-SAM2 Image Encoder">
</p>

Please note that in the current version, the `Stride` is fixed at `1024` and the `Batch size` is fixed at `1`. These parameters control how the image is processed, and future updates may allow for their customization.

### Geo-SAM2 Interactive Segmentation

The interactive segmentation tool allows for the precise selection and segmentation of objects within an image. It offers two distinct modes of operation:

  - **Preview Mode:** In this mode, as you move the mouse cursor over the image, the plugin provides a real-time segmentation preview. This is useful for quickly exploring and identifying objects.
  - **Non-Preview Mode:** This mode allows for more deliberate segmentation. The segmentation result is generated only after you click to place points or draw a bounding box around the desired object. This provides greater control and precision.

The animation below illustrates the interactive segmentation process, showcasing how to select objects within a large image.

<p align="center">
  <img src="source/img/geo_sam2_large_image.gif" width="1000" title="Geo-SAM2 Interactive Segmentation">
</p>
