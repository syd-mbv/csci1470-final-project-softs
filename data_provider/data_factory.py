import tensorflow as tf
import numpy as np
from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,
    Dataset_Solar, Dataset_PEMS, Dataset_Pred, Dataset_Random,
    Dataset_M4
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
    'M4': Dataset_M4,
}

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

    # print(">>> DEBUG start")
    # first = next(gen())           # 取一条样本
    # for i, t in enumerate(first):
    #     print(f"  element {i}: shape = {np.shape(t)}  rank = {np.ndim(t)}")
    # print(">>> DEBUG end")

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

    # drop_last = flag in ('train', 'test')
    bs        = 1 if flag == 'pred' else args.batch_size
    full      = full.batch(bs, drop_remainder=False)

    # ---------- 5) cache + prefetch ----------
    tf_dataset = full.cache().prefetch(tf.data.AUTOTUNE)
    # tf_dataset = full.prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.experimental_threading.private_threadpool_size = args.num_workers or tf.data.AUTOTUNE
    tf_dataset = tf_dataset.with_options(options)
    # --------- 5. dataset 对象仅用于 len() 与 inverse_transform ----------
    return ds_obj, tf_dataset
