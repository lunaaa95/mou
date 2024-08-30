# MoU
This is the implementation of paper: Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need.

## 📍 Our Paper
Our paper will soon be released on [Arxiv](https://arxiv.org/abs/2408.15997). We introduce a new versatile model **Mixture of Universals (MoU)** to capture both short-term and long-term dynamics for enhancing perfomance in time series forecasting. MoU is composed of two novel designs: Mixture of Feature Extractors (MoF), an adaptive method designed to improve time series patch representations for short-term dependency, and Mixture of Architectures (MoA), which hierarchically integrates Mamba, FeedForward, Convolution, and Self-Attention architectures in a specialized order to model long-term dependency from a hybrid perspective. The proposed approach achieves state-of-the-art performance while maintaining relatively low computational costs. 

## 📍 Overview
<div align="center">
  <figure>
    <img src="https://github.com/lunaaa95/mou/blob/main/figs/overview.png" alt="overview">
  <figcaption>Overview of MoU</figcaption>
  </figure>
</div>

## 📍 Preparation
Install Python 3.8 and necessary dependencies.
```pip
pip install -r requirements.txt
```
Download datasets to folder "dataset". You can download all datasets from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided by [Wu, H.](https://github.com/thuml/Autoformer?tab=readme)
```bash
mkdir dataset
```

## 📍 Run

Run bash scripts in folder "scripts" to start time series long-term forecasting. For example,
```bash
bash scripts/MoU/etth1.sh

bash scripts/MoU/etth2.sh

bash scripts/MoU/ettm1.sh

bash scripts/MoU/ettm2.sh

bash scripts/MoU/weather.sh

bash scripts/MoU/electricity.sh

bash scripts/MoU/illness.sh
```
