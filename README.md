# MoU
This is the implementation of paper: Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need.

# Preparation
Install Python 3.8 and necessary dependencies
```pip
pip install -r requirements.txt
```
Download datasets to folder "dataset". You can download all datasets from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided by [Wu, H.](https://github.com/thuml/Autoformer?tab=readme)
```bash
mkdir dataset
```

# Run

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
