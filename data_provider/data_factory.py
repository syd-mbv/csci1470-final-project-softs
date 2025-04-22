import tensorflow as tf
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
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last    = False
        batch_size   = args.batch_size
        freq         = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last    = False
        batch_size   = 1
        freq         = args.freq
        Data         = Dataset_Pred
    else:  # train / val
        shuffle_flag = True
        drop_last    = False
        batch_size   = args.batch_size
        freq         = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len, args.enc_in],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    print(flag, len(data_set))

    # --- 将 PyTorch DataLoader 替换为 tf.data.Dataset ---
    def generator():
        for i in range(len(data_set)):
            yield data_set[i]

    # 根据实际输出自行调整 output_signature
    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
        tf.TensorSpec(shape=data_set[0][0].shape, dtype=tf.float32),  # seq_x
        tf.TensorSpec(shape=data_set[0][1].shape, dtype=tf.float32),  # seq_y
        tf.TensorSpec(shape=data_set[0][2].shape, dtype=tf.float32),  # seq_x_mark
        tf.TensorSpec(shape=data_set[0][3].shape, dtype=tf.float32),  # seq_y_mark
        )
    )

    if shuffle_flag:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(data_set))

    # # map：在这里做 **CPU 侧** 的预处理 / 数据增强 / 类型转换
    # def parse_fn(x, y):
    #     # 举例：标准化或转 dtype；如果已有就可以直接 return
    #     x = tf.cast(x, tf.float32)
    #     y = tf.cast(y, tf.float32)
    #     # 也可以做更多操作，例如：x = (x - mean) / std
    #     return x, y
    # tf_dataset = tf_dataset.map(
    #     parse_fn,
    #     num_parallel_calls=tf.data.AUTOTUNE   # 多线程并行解析
    # )

    
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=drop_last)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return data_set, tf_dataset
