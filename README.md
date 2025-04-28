# CSCI1470-Final-Project-SOFTS
Final project for Brown CSCI1470 Deep Learning

## Introduction

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

To run the script files in **Windows**, see **Appendix A**.


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

# git clone https://github.com/syd-mbv/csci1470-final-project-softs.git
```

## Our Results (TensorFlow)
| Dataset | Models Metric | MSE | MAE |
|:------:|:--------:|:-----:|:------:|
| ECL | 96 | 0.152 | 0.239 |
|  | 192 | 0.165 | 0.252 |
|  | 336  | 0.183 | 0.270 |
|  | 720  | 0.218 | 0.300 |
| | Avg  | 0.179 | 0.265 |
|Traffic| 96 | 0.3901	| 0.2591|
|  | 192 | 0.4118 | 0.2687 |
|  | 336  | 0.4267 | 0.2757 |
|  | 720  | 0.4602 | 0.2943 |
|  | Avg  | 0.4222 | 0.2744 |
|Weather| 96 | 0.1736 | 0.2127|
|  | 192 | 0.2182 | 0.2540 |
|  | 336 | 0.2807 | 0.2988 |
|  | 720  | 0.3791 | 0.3638 |
|  | Avg  | 0.2629 | 0.2823 |
| Solar-Energy | 96 | 0.1957 | 0.2335 |
|  | 192 | 0.2275 | 0.2567 |
|  | 336 | 0.2457 | 0.2732 |
|  | 720 | 0.2462 | 0.2729 |
|  | Avg | 0.2288 | 0.2591 |
| ETTh1 | 96 | 0.3762 | 0.3981 |
|  | 192 | 0.4280 | 0.4270 |
|  | 336 | 0.4811 | 0.4547 |
|  | 720 | 0.5076 | 0.4922 |
|  | Avg | 0.4482 | 0.4430 |
| ETTh2 | 96 | 0.3074 | 0.3558 |
|  | 192 | 0.3775 | 0.4001 |
|  | 336 | 0.4177 | 0.4301 |
|  | 720 | 0.4282 | 0.4444 |
|  | Avg | 0.3827 | 0.4076 |
| ETTm1 | 96 | 0.3270 | 0.3656 |
|  | 192 | 0.3715 | 0.3868 |
|  | 336 | 0.4195 | 0.4178 |
|  | 720 | 0.4756 | 0.4542 |
|  | Avg | 0.3984 | 0.4061 |
| ETTm2 | 96 | 0.1808 | 0.2625 |
|  | 192 | 0.2430 | 0.3027 |
|  | 336 | 0.3017 | 0.3414 |
|  | 720 | 0.4053 | 0.4033 |
|  | Avg | 0.2827 | 0.3275 |
| PEMS03 | 12 | 0.0644 | 0.1654 |
|  | 24 | 0.0842| 0.1903 |
|  | 48 | 0.1223 | 0.2299 |
|  | 96 | 0.1575 | 0.2646 |
|  | Avg | 0.1071 | 0.2123|
| PEMS04 | 12 | 0.0753 | 0.1767 |
|  | 24 | 0.0925| 0.1985 |
|  | 48 | 0.1209 | 0.2301 |
|  | 96 | 0.1479 | 0.2563 |
|  | Avg|  0.1092 |   0.2154  |
| PEMS07 | 12 | 0.0599 | 0.1566 |
|  | 24 | 0.0809| 0.1835 |
|  | 48 | 0.1058 | 0.2061 |
|  | 96 | 0.1309 | 0.2319 |
|  | Avg|  0.0944     |   0.1945    |
| PEMS08 | 12 | 0.0772 | 0.1769 |
|  | 24 | 0.1103| 0.2105 |
|  | 48 | 0.1696 | 0.2558 |
|  | 96 | 0.2210 | 0.2594 |
|  | Avg|   0.1445   |  0.2256    |


## Appendix A: How to run the original SOFTS (PyTorch) in Windows?
To run the script files in **Windows**, ```run.py``` should be modified as following.

``` python
import argparse
import random

import numpy as np
import torch
import os

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_num_threads(6)

    parser = argparse.ArgumentParser(description='SOFTS')
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_core', type=int, default=512, help='dimension of core')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--attention_type', type=str, default="full", help='the attention type of transformer')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    print('Args in experiment:')
    print(args)
    Exp = Exp_Long_Term_Forecast

    def train(args=args):
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des)
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()


    if args.is_training:
        train(args)
    else:
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Windows: spawn
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
```
## Memo: Running on Oscar
```
interact -q gpu -g 1 -n 4 -t 02:00:00 -m 10g 
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate my_tfgpu
nvidia-smi
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
./scripts/long_term_forecast/PEMS/SOFTS_04.sh
```
