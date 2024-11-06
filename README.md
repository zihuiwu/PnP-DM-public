# Code for "Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors" (NeurIPS 2024)

### 1) Install packages
```python
conda create -n pnpdm python=3.10
conda activate pnpdm
pip install -r requirements.txt
```

### 2) Download pretrained checkpoint

Download the corresponding checkpoint from the links below and move it to ```./models/```.
 - [FFHQ (color)](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing)
 - [FFHQ (grayscale)](https://caltech.box.com/s/j58w0bf2pe2t0lrzoq45du0hc55ba4lc)
 - [Blackhole (grayscale)](https://caltech.box.com/s/j58w0bf2pe2t0lrzoq45du0hc55ba4lc)

### 4) Modify the dataset and model paths in config files
You need to modify the paths in the following files so that the dataset and models can be loaded properly:
 - `./configs/data/ffhq.yaml`
 - `./configs/data/ffhq_grayscale.yaml`
 - `./configs/model/edm_unet_adm_blackhole.yaml`
 - `./configs/model/edm_unet_adm_dps_ffhq.yaml`
 - `./configs/model/edm_unet_adm_gray_ffhq.yaml`

### 4) Run experiments
All the commands for running our experiments are provided in ```commands.sh```.
The experiments are configured using [hydra](https://hydra.cc/). 
Please see its documentation for detailed usage.

### 5) Citation
Thank you for your interest in our work!
Please consider citing 
```
@inproceedings{wu2024principled,
	title={Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors},
	author={Zihui Wu and Yu Sun and Yifan Chen and Bingliang Zhang and Yisong Yue and Katherine Bouman},
	booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
	year={2024},
	url={https://openreview.net/forum?id=Xq9HQf7VNV}
}
```
Please email zwu2@caltech.edu if you run into any problems.