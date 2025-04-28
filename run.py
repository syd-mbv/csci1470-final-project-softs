import argparse
import random
import numpy as np
import tensorflow as tf
import os

# # for test
# import cProfile, pstats

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  


os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

def main():
    # 1) 固定随机种子
    fix_seed = 2021
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    tf.random.set_seed(fix_seed)

    # 2) 限制 TensorFlow 线程并行
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(6)

    # 3) 解析参数
    parser = argparse.ArgumentParser(description='SOFTS')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='Autoformer')
    parser.add_argument('--data', type=str, required=True, default='ETTm1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_core', type=int, default=512)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--attention_type', type=str, default="full")
    parser.add_argument('--use_norm', type=int, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--save_model', action='store_true', default=True)
    args = parser.parse_args()

    # 4) TensorFlow GPU / 多 GPU 设置
    gpus = tf.config.list_physical_devices('GPU')
    args.use_gpu = bool(gpus) and args.use_gpu
    if args.use_gpu and args.use_multi_gpu:
        # 指定哪些 GPU 可见，并用 MirroredStrategy
        device_ids = [int(x) for x in args.devices.split(',') if x.strip().isdigit()]
        chosen = [gpus[i] for i in device_ids]
        tf.config.set_visible_devices(chosen, 'GPU')
        strategy = tf.distribute.MirroredStrategy()
    elif args.use_gpu:
        # 单 GPU
        tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
        strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:{args.gpu}")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    print("Args in experiment:\n", args)

    # 5) 实验类
    Exp = Exp_Long_Term_Forecast

    def train():
        setting = (
            f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
            f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
            f"dm{args.d_model}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_"
            f"fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}"
        )
        # 把模型和优化器都创建在 strategy.scope() 里
        # with strategy.scope():
        #     exp = Exp(args)
        with strategy.scope():
            exp = Exp(args, strategy=strategy)
            print(f">>>>>>> start training : {setting} >>>>>>>>>>")
            exp.train(setting)
            print(f">>>>>>> testing : {setting} <<<<<<<<")
            exp.test(setting, test=1)

    def test_only():
        setting = (
            f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
            f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
            f"dm{args.d_model}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_"
            f"fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}"
        )
        # with strategy.scope():
        #     exp = Exp(args)
        with strategy.scope():
            exp = Exp(args, strategy=strategy)
            print(f">>>>>>> testing : {setting} <<<<<<<<")
            exp.test(setting, test=1)

    # 6) 根据 is_training 决定
    if args.is_training:
        train()
    else:
        test_only()


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()

    main()

    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats("cumtime")
    # ps.print_stats(20)  # 打印耗时排名前 20 的函数