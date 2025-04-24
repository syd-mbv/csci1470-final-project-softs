# import tensorflow as tf
# import numpy as np
# from data_provider.data_loader import (
#     Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,
#     Dataset_Solar, Dataset_PEMS, Dataset_Pred, Dataset_Random
# )

# data_dict = {
#     'ETTh1': Dataset_ETT_hour,
#     'ETTh2': Dataset_ETT_hour,
#     'ETTm1': Dataset_ETT_minute,
#     'ETTm2': Dataset_ETT_minute,
#     'custom': Dataset_Custom,
#     'random': Dataset_Random,
#     'Solar':  Dataset_Solar,
#     'PEMS':   Dataset_PEMS,
# }


# def data_provider(args, flag):
#     Data = data_dict[args.data]
#     timeenc = 0 if args.embed != 'timeF' else 1

#     if flag == 'test':
#         shuffle_flag = False
#         drop_last    = False
#         batch_size   = args.batch_size
#         freq         = args.freq
#     elif flag == 'pred':
#         shuffle_flag = False
#         drop_last    = False
#         batch_size   = 1
#         freq         = args.freq
#         Data         = Dataset_Pred
#     else:  # train / val
#         shuffle_flag = True
#         drop_last    = False
#         batch_size   = args.batch_size
#         freq         = args.freq

#     data_set = Data(
#         root_path=args.root_path,
#         data_path=args.data_path,
#         flag=flag,
#         size=[args.seq_len, args.label_len, args.pred_len, args.enc_in],
#         features=args.features,
#         target=args.target,
#         timeenc=timeenc,
#         freq=freq,
#         seasonal_patterns=getattr(args, 'seasonal_patterns', None)
#     )
#     print(flag, len(data_set))

#     # 准备一个 Python 函数：给定索引，返回四个 np.ndarray
#     def _load_by_index(idx):
#         # idx 可能是 EagerTensor 或 numpy scalar
#         i = int(idx.numpy() if hasattr(idx, 'numpy') else idx)
#         seq_x, seq_y, seq_x_mark, seq_y_mark = data_set[i]
#         # 用 np.array 保证 dtype=np.float32
#         return (
#             np.array(seq_x,       dtype=np.float32),
#             np.array(seq_y,       dtype=np.float32),
#             np.array(seq_x_mark,  dtype=np.float32),
#             np.array(seq_y_mark,  dtype=np.float32),
#         )

#     # 包装成 tf.py_function
#     def _tf_load(idx):
#         x, y, xm, ym = tf.py_function(
#             func=_load_by_index,
#             inp=[idx],
#             Tout=[tf.float32, tf.float32, tf.float32, tf.float32]
#         )
#         # 明确静态形状
#         x.set_shape(data_set[0][0].shape)
#         y.set_shape(data_set[0][1].shape)
#         xm.set_shape(data_set[0][2].shape)
#         ym.set_shape(data_set[0][3].shape)
#         return x, y, xm, ym

#     # 从索引 Dataset 开始
#     ds = tf.data.Dataset.range(len(data_set))

#     if shuffle_flag:
#         ds = ds.shuffle(buffer_size=len(data_set))

#     # 用 interleave 并行调度多路 _tf_load 调用
#     tf_dataset = ds.interleave(
#         lambda idx: tf.data.Dataset.from_tensors(idx).map(
#             _tf_load,
#             num_parallel_calls=args.num_workers or tf.data.AUTOTUNE
#         ),
#         cycle_length=args.num_workers or tf.data.AUTOTUNE,
#         num_parallel_calls=args.num_workers or tf.data.AUTOTUNE,
#         deterministic=False
#     )

#     # 如果内存允许，第一次迭代后缓存数据
#     # tf_dataset = tf_dataset.cache()

#     tf_dataset = tf_dataset.batch(batch_size, drop_remainder=drop_last)
#     tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

#     # 配置私有线程池（可选）
#     options = tf.data.Options()
#     # 消除警告
#     options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
#     options.experimental_threading.private_threadpool_size = args.num_workers or tf.data.AUTOTUNE
#     options.experimental_threading.max_intra_op_parallelism = 1
#     tf_dataset = tf_dataset.with_options(options)

#     return data_set, tf_dataset

import tensorflow as tf
import numpy as np
from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,
    Dataset_Solar, Dataset_PEMS, Dataset_Pred, Dataset_Random
)

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'random': Dataset_Random,
    'Solar':  Dataset_Solar,
    'PEMS':   Dataset_PEMS,
}

def data_provider(args, flag):
    # 1) 选 Dataset 类，构造 data_set
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    if flag == 'pred':
        Data = Dataset_Pred

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len, args.enc_in],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=getattr(args, 'seasonal_patterns', None)
    )
    print(f"{flag} | # samples = {len(data_set)}")

    # 2) 取出所有样本到 numpy array（一次性）
    #    dtype=np.float32 保证 from_tensor_slices 后就是 tf.float32
    xs, ys, xms, yms = [], [], [], []
    for seq_x, seq_y, seq_x_mark, seq_y_mark in data_set:
        # 如果这是 EagerTensor，就先转到 numpy
        if isinstance(seq_x, tf.Tensor):
            seq_x = seq_x.numpy()
            seq_y = seq_y.numpy()
            seq_x_mark = seq_x_mark.numpy()
            seq_y_mark = seq_y_mark.numpy()

        # 用 np.array 保证 dtype=np.float32
        xs.append(np.array(seq_x,      dtype=np.float32))
        ys.append(np.array(seq_y,      dtype=np.float32))
        xms.append(np.array(seq_x_mark, dtype=np.float32))
        yms.append(np.array(seq_y_mark, dtype=np.float32))

    xs   = np.stack(xs, axis=0)   # shape [N, seq_len, ...]
    ys   = np.stack(ys, axis=0)
    xms  = np.stack(xms, axis=0)
    yms  = np.stack(yms, axis=0)

    # 3) 从这些 numpy 数组直接构造 Dataset
    ds = tf.data.Dataset.from_tensor_slices((xs, ys, xms, yms))

    # 4) 训练 / 验证 / 测试 的不同设置
    if flag in ['train', 'val']:
        ds = ds.shuffle(buffer_size=len(xs))
        batch_size = args.batch_size
        drop_last  = False
    elif flag == 'pred':
        batch_size = 1
        drop_last  = False
    else:  # flag == 'test'
        batch_size = args.batch_size
        drop_last  = False

    # 5) 分布式自动分片
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    ds = ds.with_options(options)

    # 6) Batch + Prefetch
    ds = ds.batch(batch_size, drop_remainder=drop_last)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return data_set, ds
