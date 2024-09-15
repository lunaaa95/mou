# Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need <br><sub>Official PyTorch Implementation</sub>
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2408.15997-b31b1b.svg)](https://arxiv.org/abs/2408.15997)&nbsp;
<div align="center">
  <figure>
    <img src="https://github.com/lunaaa95/mou/blob/main/figs/overview.png" alt="overview">
  <figcaption>Overview of MoU</figcaption>
  </figure>
</div>

## 💥 Our Paper
Our paper has been released on [Arxiv](https://arxiv.org/abs/2408.15997). We introduce a new versatile model **Mixture of Universals (MoU)** to capture both short-term and long-term dynamics for enhancing perfomance in time series forecasting. MoU is composed of two novel designs: Mixture of Feature Extractors (MoF), an adaptive method designed to improve time series patch representations for short-term dependency, and Mixture of Architectures (MoA), which hierarchically integrates Mamba, FeedForward, Convolution, and Self-Attention architectures in a specialized order to model long-term dependency from a hybrid perspective. The proposed approach achieves state-of-the-art performance while maintaining relatively low computational costs. 

The overall performance of MoU for long-term forecasting is summarized in the following Table (average performance). More detailed results can be found in our paper.
| Model       | Ours(MoU) |           | ModernTCN |           | PatchTST |          |  DLinear  |        |  S-Mamba  |          |
|-------------|-----------|-----------|-----------|-----------|----------|----------|-----------|--------|-----------|----------|
| Metric      | MSE       | MAE       | MSE       | MAE       | MSE      | MAE      | MSE       | MAE    | MSE       | MAE      |
| ETTh1       | 0.397     | 0.423     | 0.404     | 0.420     | 0.413    | 0.434    | 0.423     | 0.437  | 0.450     | 0.456    |
| ETTh2       | 0.317     | 0.373     | 0.323     | 0.378     | 0.331    | 0.379    | 0.431     | 0.447  | 0.369     | 0.405    |
| ETTm1       | 0.348     | 0.382     | 0.354     | 0.381     | 0.352    | 0.382    | 0.357     | 0.379  | 0.366     | 0.396    |
| ETTm2       | 0.252     | 0.315     | 0.256     | 0.316     | 0.256    | 0.317    | 0.267     | 0.332  | 0.265     | 0.326    |
| Weather     | 0.221     | 0.262     | 0.224     | 0.267     | 0.225    | 0.264    | 0.240     | 0.300  | 0.236     | 0.273    |
| illness     | 1.500     | 0.784     | 1.519     | 0.799     | 1.513    | 0.825    | 2.169     | 1.041  | 1.977     | 0.890    |
| electricity | 0.157     | 0.253     | 0.157     | 0.253     | 0.159    | 0.253    | 0.177     | 0.274  | 0.166     | 0.262    |


## ⚡️ Preparation
### Installation
Download code:
```
git clone https://github.com/lunaaa95/mou.git
cd mou
```
A suitable [conda](https://conda.io/) environment named `mou` can be created and activated with:
```
conda create -n mou python=3.8
conda activate mou
pip install -r requirement.txt
```
### Dataset
Download datasets to folder `./dataset`. You can download all datasets from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided by [Wu, H.](https://github.com/thuml/Autoformer?tab=readme)

## 📍 Run

* We provide bash scripts for all datasets. Run bash scripts in folder "./scripts" to start time series long-term forecasting. For example,
```bash
bash scripts/MoU/etth1.sh

bash scripts/MoU/etth2.sh

bash scripts/MoU/ettm1.sh

bash scripts/MoU/ettm2.sh

bash scripts/MoU/weather.sh

bash scripts/MoU/electricity.sh

bash scripts/MoU/illness.sh
```
* We also provide other short-term encoders and long-term encoders to switch the structure of model. Change parameters `entype` for other short-term encoders and `ltencoder` for long-term encoders.
* We also give two baseline models of `PatchTST` and `DLinear` as well as their runing scripts.

## 🌟 Citation
```
@misc{peng2024mambatransformertimeseries,
      title={Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need}, 
      author={Sijia Peng and Yun Xiong and Yangyong Zhu and Zhiqiang Shen},
      year={2024},
      eprint={2408.15997},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.15997}, 
}
```
