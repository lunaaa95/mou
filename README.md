# MoU
This is the implementation of paper: Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need.

## Our Paper
We introduce a new versatile model **Mixture of Universals (MoU)** to capture both short-term and long-term dynamics for enhancing perfomance in time series forecasting. MoU is composed of two novel designs: Mixture of Feature Extractors (MoF), an adaptive method designed to improve time series patch representations for short-term dependency, and Mixture of Architectures (MoA), which hierarchically integrates Mamba, FeedForward, Convolution, and Self-Attention architectures in a specialized order to model long-term dependency from a hybrid perspective. The proposed approach achieves state-of-the-art performance while maintaining relatively low computational costs. 

## Model Structure
Specifically, MoU is consist of MoF for short-term dependecy modeling and MoA for long-term dependecy modeling. 
MoF is designed to capture divergent context information for better short-term dependency modeling, the structure of MoF is shown as follows:
<div align="center">
  <figure>
    <img src="https://github.com/lunaaa95/mou/blob/main/figs/overview.png" alt="mof">
  <figcaption>Structure of MoF</figcaption>
  </figure>
</div>


The architecture of MoA is designed to hierarchically model long-term dependencies by progressively expanding its receptive fields. This approach facilitates the formation of a comprehensive representation of long-term dependencies. The structure of MoA is displayed as follows: 

<div align="center">
  <figure>
    <img src="https://github.com/lunaaa95/mou/blob/main/figs/moa.png" alt="moa">
  <figcaption>Structure of MoA</figcaption>
  </figure>
</div>

## Model Efficiency Comparison
We conduct model efficiency comparison experiment including training time and memory in use during training process, comparing MoU with other baseline models. The results are shown as follows:
<div align="center">
  <figure>
    <img src="https://github.com/lunaaa95/mou/blob/main/figs/efficiency.png" alt="eff">
  <figcaption>Model efficiency comparison</figcaption>
  </figure>
</div>

## Main Result
MoU achieves state-of-art performance compared with other baselines in time series long-term forecasting task.
<div align="center">
  <figure>
    <img src="https://github.com/lunaaa95/mou/blob/main/figs/result.png" alt="eff">
  <figcaption>Main result on seven real-world datasets for time series forcasting</figcaption>
  </figure>
</div>

## Preparation
Install Python 3.8 and necessary dependencies.
```pip
pip install -r requirements.txt
```
Download datasets to folder "dataset". You can download all datasets from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided by [Wu, H.](https://github.com/thuml/Autoformer?tab=readme)
```bash
mkdir dataset
```

## Run

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
