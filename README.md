🆕 [2025-08-14] :fire: DINOv3 backbones are now available in [Hugging Face Hub](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009) and [supported](https://huggingface.co/docs/transformers/model_doc/dinov3) by the Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) library

# DINOv3 🦖🦖🦖

**[Meta AI Research, FAIR](https://ai.meta.com/research/)**

Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, <br/>
Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, <br/>
Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, <br/>
Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, <br/>
Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, <br/>
Julien Mairal, Hervé Jégou, Patrick Labatut, Piotr Bojanowski

[ :scroll: [`Paper`](https://arxiv.org/abs/2508.10104)] [ :newspaper: [`Blog`](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)] [ :globe_with_meridians: [`Website`](https://ai.meta.com/dinov3/)] [ :book: [`BibTeX`](#citing-dinov3)]

Reference PyTorch implementation and models for DINOv3. For details, see the **[DINOv3](https://arxiv.org/abs/2508.10104)** paper.

## Overview

<div align="center">
  <img width="1364" height="1024" alt="market" src="https://github.com/user-attachments/assets/1411f491-988e-49cb-95ae-d03fe6e3c268" />

  <i></em><b>High-resolution dense features.</b><br/>We visualize the cosine similarity maps obtained with DINOv3 output features<br/> between the patches marked with a red cross and all other patches.</i>
</div>

<br/>

An extended family of versatile vision foundation models producing high-quality dense features and achieving outstanding performance on various vision tasks including outperforming the specialized state of the art across a broad range of settings, without fine-tuning

## Pretrained models

:information_source: Please follow the link provided below to get access to all the model weights: once accepted, an e-mail will be sent with the complete list of URLs pointing to all the available model weights (both backbones and adapters). These URLs can then be used to either:
- download the model or adapter weights to a local filesystem and point `torch.hub.load()` to these local weights via the `weights` or `backbone_weights` parameters, or
- directly invoke `torch.hub.load()` to download and load a backbone or an adapter from its URL via also the `weights` or `backbone_weights` parameters.

See the example code snippets below.

:warning: Please use `wget` instead of a web browser to download the weights.

ViT models pretrained on web dataset (LVD-1689M):
<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/16 distilled </td>
      <td align="right">21M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-S+/16 distilled</td>
      <td align="right">29M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-B/16 distilled</td>
      <td align="right">86M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-L/16 distilled</td>
      <td align="right">300M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-H+/16 distilled</td>
      <td align="right">840M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-7B/16</td>
      <td align="right">6,716M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
  </tbody>
</table>

ConvNeXt models pretrained on web dataset (LVD-1689M):
<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ConvNeXt Tiny</td>
      <td align="right">29M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ConvNeXt Small</td>
      <td align="right">50M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ConvNeXt Base</td>
      <td align="right">89M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ConvNeXt Large</td>
      <td align="right">198M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
  </tbody>
</table>

ViT models pretrained on satellite dataset (SAT-493M):
<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-L/16 distilled</td>
      <td align="right">300M</td>
      <td align="center">SAT-493M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-7B/16</td>
      <td align="right">6,716M</td>
      <td align="center">SAT-493M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
  </tbody>
</table>


### Pretrained backbones (via PyTorch [Hub](https://docs.pytorch.org/docs/stable/hub.html))

Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch (the only required dependency for loading the model). Installing PyTorch with CUDA support is strongly recommended.

```python
import torch

REPO_DIR = <PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV3/REPO/WAS/CLONED>

# DINOv3 ViT models pretrained on web images
dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vits16plus = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vith16plus = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

# DINOv3 ConvNeXt models pretrained on web images
dinov3_convnext_tiny = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_small = torch.hub.load(REPO_DIR, 'dinov3_convnext_small', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_base = torch.hub.load(REPO_DIR, 'dinov3_convnext_base', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_large = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

# DINOv3 ViT models pretrained on satellite imagery
dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
```

### Pretrained backbones (via Hugging Face [Transformers](https://huggingface.co/docs/transformers/))

All the backbones are available in the the [DINOv3](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009) collection on Hugging Face Hub and supported via the Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) library. Please refer to the corresponding documentation for usage, but below is a short example that demonstrates how to obtain an image embedding with either [Pipeline] or the [AutoModel] class.

```python
from transformers import pipeline
from transformers.image_utils import load_image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    task="image-feature-extraction", 
)
features = feature_extractor(image)
```

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)
```

where `model` and `pretrained_model_name` above can be one of:
- `facebook/dinov3-vits16-pretrain-lvd1689m`
- `facebook/dinov3-vits16plus-pretrain-lvd1689m`
- `facebook/dinov3-vitb16-pretrain-lvd1689m`
- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- `facebook/dinov3-vith16plus-pretrain-lvd1689m`
- `facebook/dinov3-vit7b16-pretrain-lvd1689m`
- `facebook/dinov3-convnext-base-pretrain-lvd1689m`
- `facebook/dinov3-convnext-large-pretrain-lvd1689m`
- `facebook/dinov3-convnext-small-pretrain-lvd1689m`
- `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
- `facebook/dinov3-vitl16-pretrain-sat493m`
- `facebook/dinov3-vit7b16-pretrain-sat493m`

### Image transforms

For models using the LVD-1689M weights (pretrained on web images), please use the following transform (standard ImageNet evaluation transform):

```python
import torchvision
from torchvision.transforms import v2

def make_transform(resize_size: int = 224):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])
```


For models using the SAT-493M weights (pretrained on satellite imagery), please use the following transform:


```python
import torchvision
from torchvision.transforms import v2

def make_transform(resize_size: int = 224):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])
```

### Pretrained heads - Image classification

<table style="margin: auto">
  <thead>
    <tr>
      <th>Backbone</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Head<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-7B/16</td>
      <td align="center">LVD-1689M</td>
      <td align="center">ImageNet</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
  </tbody>
</table>


The (full) classifier models can be loaded via PyTorch Hub:

```python
import torch

# DINOv3
dinov3_vit7b16_lc = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_lc', source="local", weights=<DEPTHER/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)

```

### Pretrained heads - Depther trained on SYNTHMIX dataset

<table style="margin: auto">
  <thead>
    <tr>
      <th>Backbone</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Head<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-7B/16</td>
      <td align="center">LVD-1689M</td>
      <td align="center">SYNTHMIX</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
  </tbody>
</table>


```python
depther = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_dd', source="local", weights=<DEPTHER/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)
```

Full example code of depther on an image

```python
from PIL import Image
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib import colormaps

def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

depther = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_dd', source="local", weights=<DEPTHER/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)

img_size = 1024
img = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        batch_img = batch_img
        depths = depther(batch_img)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(depths[0,0].cpu(), cmap=colormaps["Spectral"])
plt.axis("off")

```

### Pretrained heads - Detector trained on COCO2017 dataset

<table style="margin: auto">
  <thead>
    <tr>
      <th>Backbone</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Head<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-7B/16</td>
      <td align="center">LVD-1689M</td>
      <td align="center">COCO2017</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
  </tbody>
</table>


```python
detector = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_de', source="local", weights=<DETECTOR/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)
```

### Pretrained heads - Segmentor trained on ADE20K dataset

<table style="margin: auto">
  <thead>
    <tr>
      <th>Backbone</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Head<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-7B/16</td>
      <td align="center">LVD-1689M</td>
      <td align="center">ADE20K</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
  </tbody>
</table>

```python
segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", weights=<SEGMENTOR/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)
```

Full example code of segmentator on an image

```python
import sys
sys.path.append(REPO_DIR)

from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from dinov3.eval.segmentation.inference import make_inference


def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", weights=<SEGMENTOR/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)

img_size = 896
img  = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        pred_vit7b = segmentor(batch_img)  # raw predictions  
        # actual segmentation map
        segmentation_map_vit7b = make_inference(
            batch_img,
            segmentor,
            inference_mode="slide",
            decoder_head_type="m2f",
            rescale_to=(img.size[-1], img.size[-2]),
            n_output_channels=150,
            crop_size=(img_size, img_size),
            stride=(img_size, img_size),
            output_activation=partial(torch.nn.functional.softmax, dim=1),
        ).argmax(dim=1, keepdim=True)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(segmentation_map_vit7b[0,0].cpu(), cmap=colormaps["Spectral"])
plt.axis("off")
```




### Pretrained heads - Zero-shot tasks with `dino.txt`

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">Backbone</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-L/16 distilled</td>
      <td align="center">
        <a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz">vocabulary</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE">vocabulary license</a>
      </td>
    </tr>
  </tbody>
</table>

The (full) dino.txt model can be loaded via PyTorch Hub:

```python
import torch
# DINOv3
dinov3_vitl16_dinotxt_tet1280d20h24l, tokenizer = torch.hub.load(REPO_DIR, 'dinov3_vitl16_dinotxt_tet1280d20h24l', weights=<SEGMENTOR/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)
```


## Installation

The training and evaluation code requires PyTorch version >= 2.7.1 as well as a few other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

*[micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)* **(Recommended)** - Clone the repository and then create and activate a `dinov3` conda environment using the provided environment definition:

```shell
micromamba env create -f conda.yaml
micromamba activate dinov3
```

## Getting started

Several notebooks are provided to get started applying DINOv3:
- [PCA of patch features](notebooks/pca.ipynb): display the PCA of DINOv3 patch features on a foreground object (rainbow visualizations from the paper) [[Run in Google Colab]](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/pca.ipynb)
- [Foreground segmentation](notebooks/foreground_segmentation.ipynb): train a linear foreground segmentation model based on DINOv3 features [[Run in Google Colab]](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/foreground_segmentation.ipynb)
- [Dense and sparse matching](notebooks/dense_sparse_matching.ipynb): match patches from objects on two different images based on DINOv3 features [[Run in Google Colab]](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/dense_sparse_matching.ipynb)
- [Segmentation tracking](notebooks/segmentation_tracking.ipynb): video segmentation tracking using a non-parametric method based on DINOv3 features [[Run in Google Colab]](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/segmentation_tracking.ipynb)

## Data preparation

### ImageNet-1k

The root directory of the dataset should hold the following contents:

- `<ROOT>/test/ILSVRC2012_test_00000001.JPEG`
- `<ROOT>/test/[..]`
- `<ROOT>/test/ILSVRC2012_test_00100000.JPEG`
- `<ROOT>/train/n01440764/n01440764_10026.JPEG`
- `<ROOT>/train/[...]`
- `<ROOT>/train/n15075141/n15075141_9993.JPEG`
- `<ROOT>/val/n01440764/ILSVRC2012_val_00000293.JPEG`
- `<ROOT>/val/[...]`
- `<ROOT>/val/n15075141/ILSVRC2012_val_00049174.JPEG`
- `<ROOT>/labels.txt`

The provided dataset implementation expects a few additional metadata files to be present under the extra directory:

- `<EXTRA>/class-ids-TRAIN.npy`
- `<EXTRA>/class-ids-VAL.npy`
- `<EXTRA>/class-names-TRAIN.npy`
- `<EXTRA>/class-names-VAL.npy`
- `<EXTRA>/entries-TEST.npy`
- `<EXTRA>/entries-TRAIN.npy`
- `<EXTRA>/entries-VAL.npy`

These metadata files can be generated (once) with the following lines of Python code:

```python
from dinov3.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

Note that the root and extra directories do not have to be distinct directories.

### ImageNet-22k

Please adapt the [dataset class](dinov3/data/datasets/image_net_22k.py) to match your local setup.

<br />

:warning: To execute the commands provided in the next sections for training and evaluation, the `dinov3` package should be included in the Python module search path, i.e. simply prefix the command to run with `PYTHONPATH=.`.

## Training

### Fast setup: training DINOv3 ViT-L/16 on ImageNet-1k

Run DINOv3 pre-training on 4 H100-80GB nodes (32 GPUs) in a SLURM cluster environment with submitit:

```shell
 PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 4 \
  --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  train.dataset_path=ImageNet22k:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```
Training time is approximately 14 hours and the resulting checkpoint should reach 82.0% on k-NN eval and 83.5% on linear eval.

The training code saves the weights of the teacher in the eval folder every 12500 iterations for evaluation.

### Exact DINOv3 setup: training DINOv3 ViT-7B/16

DINOv3 ViT-7B/16 is trained on a private dataset. The training involves 3 stages:
- Pretraining
- Gram anchoring
- High resolution adaptation

#### Pretraining

Launch DINOV3 ViT-7B/16 pretraining on 32 nodes (256 GPUs) in a SLURM cluster environment with submitit.

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 32 \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  train.dataset_path=<DATASET>:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

#### Gram anchoring

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 32 \
  --config-file dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  train.dataset_path=<DATASET>:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
  gram.ckpt=<PATH/TO/GRAM_TEACHER_FROM_PREVIOUS_STEP>   
```

#### High-resolution adaptation


```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 32 \
  --config-file dinov3/configs/train/dinov3_vit7b16_high_res_adapt.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  train.dataset_path=<DATASET>:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
  gram.ckpt=<PATH/TO/TEACHER_FROM_GRAM> \
  student.resume_from_teacher_chkpt=<PATH/TO/TEACHER_FROM_GRAM>
```

## Multi-distillation 

### Test setup:

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 1 \
  --config-file dinov3/configs/train/multi_distillation_test.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  --multi-distillation \
  train.dataset_path=<DATASET>:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

## Evaluation

The training code regularly saves the teacher weights. In order to evaluate the model, run the following evaluation on a single node:


### Logistic regression classification on ImageNet-1k

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/log_regression.py \
  model.config_file=<PATH/TO/OUTPUT/DIR>/config.yaml \
  model.pretrained_weights=<PATH/TO/OUTPUT/DIR>/teacher_checkpoint.pth \
  output_dir=<PATH/TO/OUTPUT/DIR> \
  train.dataset=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
  eval.test_dataset=ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### k-NN classification on ImageNet-1k

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/knn.py \
  model.config_file=<PATH/TO/OUTPUT/DIR>/config.yaml \
  model.pretrained_weights=<PATH/TO/OUTPUT/DIR>/teacher_checkpoint.pth \
  output_dir=<PATH/TO/OUTPUT/DIR> \
  train.dataset=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
  eval.test_dataset=ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### Linear classification with data augmentation on ImageNet-1k

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/linear.py \
  model.config_file=<PATH/TO/OUTPUT/DIR>/config.yaml \
  model.pretrained_weights=<PATH/TO/OUTPUT/DIR>/teacher_checkpoint.pth \
  output_dir=<PATH/TO/OUTPUT/DIR> \
  train.dataset=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
  train.val_dataset=ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```


### Text alignment on DINOv3 using dino.txt

Text alignment can be done following the method from `dino.txt` aka [DINOv2 Meets Text](https://arxiv.org/abs/2412.16334).

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/text/train_dinotxt.py \
   --nodes 4 \
  # An example config for text alignment is here: dinov3/eval/text/configs/dinov3_vitl_text.yaml \ 
  trainer_config_file="<PATH/TO/DINOv3/TEXT/CONFIG>" \
  output-dir=<PATH/TO/OUTPUT/DIR>
```
Launching the above trains text alignment on 4 nodes with 8 gpus each (32 gpus in total).
Please note that the text alignment model in the DINOv3 paper was trained on a private dataset and here we have given an example config in ```dinov3/eval/text/configs/dinov3_vitl_text.yaml``` using ```CocoCaptions``` dataset for illustration purposes.
Please adapt the provided ```CocoCaptions``` dataset class, the dataset can be found [here](https://www.kaggle.com/datasets/nikhil7280/coco-image-caption)  

## License

DINOv3 code and model weights are released under the DINOv3 License. See [LICENSE.md](LICENSE.md) for additional details.

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing DINOv3

If you find this repository useful, please consider giving a star :star: and citation :t-rex::

```
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```
