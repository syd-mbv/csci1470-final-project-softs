import tensorflow as tf
from layers.Embed import DataEmbeddingInverted
from layers.Transformer_EncDec import Encoder, EncoderLayer

class STAR(tf.keras.layers.Layer):
    """
    STar Aggregate-Redistribute Module
    对应 PyTorch 中的 STAR(nn.Module)
    """
    def __init__(self, d_series, d_core, **kwargs):
        super().__init__(**kwargs)
        self.gen1 = tf.keras.layers.Dense(d_series)
        self.gen2 = tf.keras.layers.Dense(d_core)
        self.gen3 = tf.keras.layers.Dense(d_series)
        self.gen4 = tf.keras.layers.Dense(d_series)

    def call(self, input, *args, training=None, **kwargs):
        # input: [B, C, d_series]
        B = tf.shape(input)[0]
        C = tf.shape(input)[1]
        # 1) FFN + gelu
        combined = tf.keras.activations.gelu(self.gen1(input))  # [B, C, d_series]
        combined = self.gen2(combined)                          # [B, C, d_core]

        # 2) stochastic pooling / weighted sum
        if training:
            # softmax over channels
            ratio = tf.nn.softmax(combined, axis=1)             # [B, C, d_core]
            ratio = tf.transpose(ratio, [0, 2, 1])              # [B, d_core, C]
            flat = tf.reshape(ratio, [-1, C])                   # [B*d_core, C]
            logp = tf.math.log(flat + 1e-8)
            idx = tf.random.categorical(logp, 1)                # [B*d_core, 1]
            idx = tf.reshape(idx, [B, -1, 1])                   # [B, d_core, 1]
            idx = tf.transpose(idx, [0, 2, 1])                  # [B, 1, d_core]
            pooled = tf.gather(combined, idx[..., 0], axis=1, batch_dims=1)
            combined_mean = tf.tile(pooled, [1, C, 1])
        else:
            weight = tf.nn.softmax(combined, axis=1)
            aggregated = tf.reduce_sum(combined * weight, axis=1, keepdims=True)
            combined_mean = tf.tile(aggregated, [1, C, 1])

        # 3) MLP 融合
        cat = tf.concat([input, combined_mean], axis=-1)
        fused = tf.keras.activations.gelu(self.gen3(cat))
        output = self.gen4(fused)  # [B, C, d_series]

        return output, None


class Model(tf.keras.Model):
    """
    对应 PyTorch 中的 Model(nn.Module)
    """
    def __init__(self, configs, **kwargs):
        super().__init__(**kwargs)
        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbeddingInverted(
            c_in=configs.seq_len,
            d_model=configs.d_model,
            dropout=configs.dropout
        )

        # Encoder
        attn_layers = []
        for _ in range(configs.e_layers):
            attn_layers.append(
                EncoderLayer(
                    STAR(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
            )
        self.encoder = Encoder(attn_layers)

        # 最终投影到 pred_len
        self.projection = tf.keras.layers.Dense(configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, training=None):
        # 1) 非平稳归一化
        if self.use_norm:
            means = tf.stop_gradient(tf.reduce_mean(x_enc, axis=1, keepdims=True))
            x = x_enc - means
            stdev = tf.sqrt(tf.math.reduce_variance(x, axis=1, keepdims=True) + 1e-5)
            x = x / stdev
        else:
            x = x_enc
            means = None
            stdev = None

        # 2) Encoder
        enc_out = self.enc_embedding(x, x_mark_enc, training=training)  # [B, C, d_model]
        enc_out, attns = self.encoder(enc_out, training=training)       # [B, C, d_model]

        # 3) Projection & reshape
        dec = self.projection(enc_out)              # [B, C, pred_len]
        dec = tf.transpose(dec, [0, 2, 1])          # [B, pred_len, C]
        N = tf.shape(x_enc)[2]
        dec = dec[:, :, :N]                         # [B, pred_len, C']

        # 4) 反归一化
        if self.use_norm:
            mean = means[:, 0:1, :]
            sd   = stdev[:, 0:1, :]
            dec = dec * tf.tile(sd, [1, self.pred_len, 1])
            dec = dec + tf.tile(mean, [1, self.pred_len, 1])

        return dec

    def call(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, training=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, training=training)
        # 只返回最后 pred_len 个时刻
        return dec_out[:, -self.pred_len:, :]
