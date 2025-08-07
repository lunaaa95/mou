# Semantics-Aware Patch Encoding and Hierarchical Dependency Modeling for Long-Term Time Series Forecasting <br><sub>Official PyTorch Implementation</sub>

<div align="center">
  <figure>
    <img src="https://github.com/lunaaa95/mou/blob/main/figs/diff.png" alt="overview">
  <figcaption>Overview of MoU</figcaption>
  </figure>
</div>

## 💥 Our Paper (Accepted in KDD 2025) 
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2408.15997-b31b1b.svg)](https://arxiv.org/abs/2408.15997)&nbsp; 
[![KDD](https://img.shields.io/badge/KDD-doi.org/10.1145/3711896.3737123-blue.svg)](https://doi.org/10.1145/3711896.3737123)&nbsp; 
![](https://img.shields.io/github/stars/Lunaaa95/MoU?style=social)


We introduce **Mixture of Universals (MoU)**, a novel framework designed to prevent semantic loss during patch encoding and efficiently enhance long-term dynamics through a hybrid approach. Specifically, MoU is consist of two novel designs: Mixture of Feature Extractors (MoF) and Mixture of Architectures (MoA). MoF introduces a semantics-aware encoding mechanism to preserve diverse temporal patterns and mitigating information loss. MoA, on the other hand, hierarchically captures long-term dependency with progressively expanded receptive field, improving model performance while maintaining relatively low computational costs. The proposed approach achieves state-of-the-art performance.

The overall performance of MoU for long-term forecasting is summarized in the following Table (average performance). More detailed results can be found in our paper.

| Model       | MoU (Ours) |       | ModernTCN |       | PatchTST |       | HDMixer |       | RMLP  |       | DLinear |       | S-Mamba |       | iTransformer |       |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Metric      | MSE          | MAE   | MSE       | MAE   | MSE        | MAE   | MSE     | MAE   | MSE   | MAE   | MSE     | MAE   | MSE     | MAE   | MSE          | MAE   |
| ETTh1       | 0.397        | 0.423 | 0.404     | 0.420 | 0.413      | 0.434 | 0.408   | 0.426 | 0.442 | 0.443 | 0.423   | 0.437 | 0.450   | 0.456 | 0.465        | 0.465 |
| ETTh2       | 0.317        | 0.373 | 0.323     | 0.379 | 0.331      | 0.379 | 0.320   | 0.374 | 0.377 | 0.414 | 0.431   | 0.447 | 0.369   | 0.405 | 0.385        | 0.414 |
| ETTm1       | 0.348        | 0.382 | 0.354     | 0.382 | 0.352      | 0.382 | 0.359   | 0.385 | 0.357 | 0.379 | 0.357   | 0.379 | 0.367   | 0.396 | 0.367        | 0.395 |
| ETTm2       | 0.252        | 0.315 | 0.256     | 0.316 | 0.256      | 0.317 | 0.257   | 0.316 | 0.263 | 0.318 | 0.267   | 0.332 | 0.265   | 0.326 | 0.271        | 0.329 |
| Weather     | 0.221        | 0.262 | 0.225     | 0.267 | 0.226      | 0.264 | 0.235   | 0.275 | 0.236 | 0.273 | 0.240   | 0.300 | 0.236   | 0.273 | 0.238        | 0.273 |
| illness     | 1.500        | 0.784 | 1.519     | 0.799 | 1.513      | 0.825 | 2.019   | 0.891 | 1.593 | 0.843 | 2.169   | 1.041 | 1.977   | 0.890 | 2.222        | 1.012 |
| electricity | 0.157        | 0.253 | 0.157     | 0.253 | 0.159      | 0.253 | 0.160   | 0.252 | 0.172 | 0.266 | 0.177   | 0.274 | 0.166   | 0.262 | 0.170        | 0.265 |


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
@inproceedings{peng2025semantics,
  title={Semantics-Aware Patch Encoding and Hierarchical Dependency Modeling for Long-Term Time Series Forecasting},
  author={Peng, Sijia and Xiong, Yun and Zhu, Yangyong and Shen, Zhiqiang},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={2269--2280},
  year={2025}
}
```
