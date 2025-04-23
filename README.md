# csci1470-final-project-softs
Final project of Brown CSCI1470 Deep Learning

## Dataset


## Re-implement *SOFTS* (Original: PyTorch)
```
conda create -n [env_name] python=3.8
conda activate [env_name]
#ref :https://pytorch.org/get-started/previous-versions/#v1100
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install scikit-learn==1.2.2 numpy==1.22.4 pandas==1.2.4

# git clone https://github.com/Secilia-Cxy/SOFTS.git
# cd SOFTS
```

To reproduce the main results, run the script files under folder ```scripts/long_term_forecast```. 

For example, to reproduce the results of SOFTS on ETTm1 dataset, run the following command:

```sh scripts/long_term_forecast/ETT_script/SOFTS_ETTm1.sh```.

To run the script files in Windows, ```run.py``` should be modified as following:
``` python
import argparse
import random

import numpy as np
import torch
import os

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# forbid MKL / OpenMP
os.environ["OMP_NUM_THREADS"]  = "1"
os.environ["MKL_NUM_THREADS"]  = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
# torch.set_num_threads(6)
torch.set_num_threads(1)

# …………… (omitted other parameters remain the same) ……………

# Change the num_workers in the DataLoader to 0

# optimization
# parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
```


## Re-implement *SOFTS* (Our: TensorFlow)
```
# Windows:
conda create -n [env_name] python=3.10 -y
conda activate [env_name]

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
python -m pip install "tensorflow<2.11"

# numpy<2 is required
pip install "numpy<2" --force-reinstall # 1.26.4

pip install scikit-learn==1.2.2 pandas==2.2.3
```

### Our Results (TensorFlow)
| Dataset | Models Metric | MSE | MAE |
|:------:|:--------:|:-----:|:------:|
| ECL | 96 | 0.152 | 0.239 |
|  | 192 | 0.165 | 0.252 |
|  | 336  | 0.183 | 0.270 |
|  | 720  | 0.218 | 0.300 |
| ECL | Avg  | 0.179 | 0.265 |