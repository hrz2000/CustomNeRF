## <span style="text-decoration: underline"><font color="Tomato">Custom</font></span>ize your <span style="text-decoration: underline"><font color="Tomato">NeRF</font></span>: Adaptive Source Driven 3D Scene Editing via Local-Global Iterative Training

Pytorch implementation of [Customize your NeRF: Adaptive Source Driven 3D Scene Editing via **Local-Global Iterative Training**](https://arxiv.org/abs/2312.01663)

Runze He,
Shaofei Huang,
Xuecheng Nie,
Tianrui Hui,
Luoqi Liu,
Jiao Dai,
Jizhong Han,
Guanbin Li,
Si Liu

[![arXiv](https://img.shields.io/badge/ArXiv-2312.01663-brightgreen)](https://arxiv.org/abs/2312.01663)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://customnerf.github.io/)

---

<div align="center">
<img src="./fig1.png">
<i> CustomNeRF unifies a text description or a reference image as the editing prompt for 3D scene editing. </i>
</div>


## Updates

<!-- - [2024/3/12] Code released. -->
- [2023/12/4] Paper is available [here](https://arxiv.org/abs/2312.01663).

---

## Introduction

<img src="./pipeline.png">

In this paper, we target the adaptive source driven 3D scene editing task by proposing a CustomNeRF model that unifies a text description or a reference image as the editing prompt. However, obtaining desired editing results conformed with the editing prompt is nontrivial since there exist two significant challenges, including accurate editing of only foreground regions and multi-view consistency given a single-view reference image. 

To tackle the first challenge, we propose a Local-Global Iterative Editing (LGIE) training scheme that alternates between foreground region editing and full-image editing, aimed at foreground-only manipulation while preserving the background. 

For the second challenge, we also design a class-guided regularization that exploits class priors within the generation model to alleviate the inconsistency problem among different views in image-driven editing. Extensive experiments show that our CustomNeRF produces precise editing results under various real scenes for both text- and image-driven settings.

## Usage

### Requirements
```base
pip install -r requirements.txt

# install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Build extension
bash scripts/install_ext.sh
```

### Dataset
You can use a variety of popular NeRF datasets, as well as a set of photos taken with your own camera. For convenience, please refer to [NeRFstudio](https://github.com/nerfstudio-project/nerfstudio/) for extracting camera poses.

<!-- Here we provide a demo of some data for your reference. -->

### Preprocess
Please refer [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) for extract mask for dataset.

Perform [Custom Diffusion](https://github.com/adobe-research/custom-diffusion) fine-tuning if you need to use a reference image as an editing prompt
```base
bash custom_diffusion/tuning.sh
```

### NeRF reconstruction
```base
data_path="./data/bear"

### nerf reconstruction
python main.py -O2 \
--workspace "./outputs/bear/base" --iters 3000 \
--backbone grid --bound 2 --train_resolution_level 7 --eval_resolution_level 4 \
--data_type "nerfstudio" --data_path $data_path \
--keyword 'bear' --train_conf 0.01 --soft_mask \
# --test --eval_resolution_level 3 \
```

### NeRF editing
```base
data_path="./data/bear"

### nerf editing
python main.py -O2 \
--workspace "./outputs/bear/base" --iters 3000 \
--backbone grid --bound 2 --train_resolution_level 7 --eval_resolution_level 4 \
--data_type "nerfstudio" --data_path $data_path \
--keyword 'bear' --train_conf 0.01 --soft_mask \
\
--workspace "./outputs/bear/text_corgi" --iters 10000 \
--train_resolution_level 7 --eval_resolution_level 7 \
--editing_from './outputs/bear/base/checkpoints/df_ep0004.pth' --pretrained \
--text 'a corgi in a forest' \
--text_fg 'a corgi' \
--lambda_sd 0.01 --keep_bg 1000 \
--stage_time --detach_bg --random_bg_c --clip_view \
# --test --eval_resolution_level 3 \
```


## Acknowledgements

We thank the awesome research works [Custom Diffusion](https://github.com/adobe-research/custom-diffusion), [torch-ngp](https://github.com/ashawkey/torch-ngp), [Grouded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything).


## Citation

```bibtex
@article{he2023customize,
      title={Customize your NeRF: Adaptive Source Driven 3D Scene Editing via Local-Global Iterative Training},
      author={He, Runze and Huang, Shaofei and Nie, Xuecheng and Hui, Tianrui and Liu, Luoqi and Dai, Jiao and Han, Jizhong and Li, Guanbin and Liu, Si},
      journal={arXiv preprint arXiv:2312.01663},
      year={2023}
}
```


## Contact

If you have any comments or questions, please [open a new issue](https://github.com/TencentARC/MasaCtrl/issues/new/choose) or feel free to contact [Runze He](https://github.com/hrz2000).