import tensorflow as tf


class DataEmbeddingInverted(tf.keras.layers.Layer):
    """
    TensorFlow 版 DataEmbedding_inverted
    ----------------------------------------------------------
    输入:
        x       : Tensor 形状 [Batch, Time, Variate]
        x_mark  : (可选) 时间/协变量 Tensor，形状同上
    处理流程:
        1. 交换维度 -> [Batch, Variate, Time]
        2. 若给定 x_mark, 按 Variate 维拼接
        3. 对最后一维 (Time) 施加全连接映射   c_in -> d_model
        4. Dropout
    输出:
        Tensor 形状 [Batch, Variate (+covariate), d_model]
    """
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        # 等价于 nn.Linear(c_in, d_model)
        self.value_embedding = tf.keras.layers.Dense(d_model, use_bias=True)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, x_mark=None, training=None):
        """
        Parameters
        ----------
        x : tf.Tensor, shape (B, T, C)
        x_mark : tf.Tensor or None, shape (B, T, C_mark)
        training : bool, 训练/推理模式

        Returns
        -------
        tf.Tensor, shape (B, C_total, d_model)
        """
        # 1) [B,T,C] -> [B,C,T]
        x = tf.transpose(x, perm=[0, 2, 1])

        # 2) 拼接协变量 (若存在)，同样先转置
        if x_mark is not None:
            x_mark = tf.transpose(x_mark, perm=[0, 2, 1])
            x = tf.concat([x, x_mark], axis=1)   # 沿 Variate 维拼接

        # 3) 线性映射: 最后一维 Time  -> d_model
        x = self.value_embedding(x)

        # 4) Dropout
        return self.dropout(x, training=training)
