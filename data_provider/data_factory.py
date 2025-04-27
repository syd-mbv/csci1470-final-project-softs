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

# def data_provider(args, flag):
#     # 1) 构造原始 python data_set
#     Data = data_dict[args.data]
#     if flag=='pred': Data = Dataset_Pred
#     timeenc = 0 if args.embed!='timeF' else 1
#     data_set = Data(
#         root_path=args.root_path, data_path=args.data_path,
#         flag=flag, size=[args.seq_len,args.label_len,args.pred_len,args.enc_in],
#         features=args.features, target=args.target,
#         timeenc=timeenc, freq=args.freq,
#         seasonal_patterns=getattr(args,'seasonal_patterns',None)
#     )
#     N = len(data_set)
#     print(f"{flag} | # samples = {N}")

#     # 2) 决定批次与 shuffle 策略
#     if flag=='test':
#         batch_size, shuffle_flag, drop_last = args.batch_size, False, True
#     elif flag=='pred':
#         batch_size, shuffle_flag, drop_last = 1, False, False
#     else:
#         batch_size, shuffle_flag, drop_last = args.batch_size, True, True

#     # 3) 从索引生成 Dataset
#     ds = tf.data.Dataset.range(N)
#     if shuffle_flag:
#         ds = ds.shuffle(buffer_size=N)

#     # 4) 先把索引按 batch 收集
#     ds = ds.batch(batch_size, drop_remainder=drop_last)

#     # 5) 用一个 batch-level 的 py_function 一次性加载整个 batch
#     def _load_batch(idx_batch):
#         # 这段在 Python 里执行一次，idx_batch 是一个 numpy array
#         idxs = idx_batch.numpy()
#         xs, ys, xms, yms = [], [], [], []
#         for i in idxs:
#             sx, sy, sxm, sym = data_set[i]
#             xs.append(np.array(sx, dtype=np.float32))
#             ys.append(np.array(sy, dtype=np.float32))
#             xms.append(np.array(sxm, dtype=np.float32))
#             yms.append(np.array(sym, dtype=np.float32))
#         return (
#             np.stack(xs, axis=0),
#             np.stack(ys, axis=0),
#             np.stack(xms, axis=0),
#             np.stack(yms, axis=0),
#         )

#     def _tf_load_batch(idx_batch):
#         outs = tf.py_function(
#             func=_load_batch,
#             inp=[idx_batch],
#             Tout=[tf.float32, tf.float32, tf.float32, tf.float32]
#         )
#         # 设置每个张量的形状：[batch_size, seq_len, ...]
#         # 用 data_set[0] 的单样本形状铺第一维
#         sample_shapes = [arr.shape for arr in data_set[0]]
#         for t, shape in zip(outs, sample_shapes):
#             t.set_shape([batch_size] + list(shape))
#         return tuple(outs)

#     ds = ds.map(
#         _tf_load_batch,
#         num_parallel_calls=args.num_workers or tf.data.AUTOTUNE
#     )

#     # 6) 预取可选（只 prefetch）
#     ds = ds.prefetch(tf.data.AUTOTUNE)

#     # 7) 分布式分片策略 & 线程池优化
#     options = tf.data.Options()
#     options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
#     options.experimental_threading.private_threadpool_size = args.num_workers or tf.data.AUTOTUNE
#     # options.experimental_threading.max_intra_op_parallelism = 1
#     ds = ds.with_options(options)

#     return data_set, ds
def data_provider(args, flag):
    # ---------- 1. 构造 python dataset ----------
    DataCls = Dataset_Pred if flag == 'pred' else data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    ds_obj  = DataCls(
        root_path=args.root_path, data_path=args.data_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len, args.enc_in],
        features=args.features, target=args.target, timeenc=timeenc,
        freq=args.freq, seasonal_patterns=getattr(args, 'seasonal_patterns', None)
    )
    N = len(ds_obj); print(f"{flag} | #samples={N}")

    # # ---------- 2. 直接用 tf.data.Dataset.from_tensor_slices ----------
    # full = tf.data.Dataset.from_tensor_slices(ds_obj)   # 每条元素就是 4 个 float32 Tensor

    # ---------- 3. shuffle / batch ----------
    def gen():
        for i in range(N):
            yield ds_obj[i]          # __getitem__ 已返回 tf.float32 Tensor

    sample = ds_obj[0]

    def _flexible_shape(t):
        return tf.TensorSpec(shape=[None] * len(t.shape), dtype=tf.float32)

    # output_signature = tuple(
    #     tf.TensorSpec(shape=t.shape, dtype=tf.float32) for t in sample
    # )
    output_signature = tuple(_flexible_shape(t) for t in sample)

    full = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # ---------- 4) shuffle / batch ----------
    if flag == 'train':
        full = full.shuffle(buffer_size=N, reshuffle_each_iteration=True)

    drop_last = flag in ('train', 'test')
    bs        = 1 if flag == 'pred' else args.batch_size
    full      = full.batch(bs, drop_remainder=drop_last)

    # ---------- 5) cache + prefetch ----------
    tf_dataset = full.cache().prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.experimental_threading.private_threadpool_size = args.num_workers or tf.data.AUTOTUNE
    tf_dataset = tf_dataset.with_options(options)
    # --------- 5. dataset 对象仅用于 len() 与 inverse_transform ----------
    return ds_obj, tf_dataset
