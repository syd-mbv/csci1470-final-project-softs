# layers/Transformer_EncDec.py

from typing import List, Optional
import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        attention: tf.keras.layers.Layer,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs
    ):
        super().__init__(**kwargs)
        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.conv1 = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1)
        self.conv2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.activation = (
            tf.keras.activations.relu
            if activation == "relu"
            else tf.keras.activations.gelu
        )

    def call(
        self,
        x,
        attn_mask=None,
        tau=None,
        delta=None,
        training=None,
        **kwargs
    ):
        # Multi‑head attention
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
            training=training,
            **kwargs
        )
        # Residual + dropout
        x = x + self.dropout(new_x, training=training)

        # First LayerNorm
        x_norm = self.norm1(x)

        # Point‑wise FFN
        y = self.conv1(x_norm)               # [B, L, d_ff]
        y = self.activation(y)
        y = self.dropout(y, training=training)
        y = self.conv2(y)                    # [B, L, d_model]
        y = self.dropout(y, training=training)

        # Residual + second LayerNorm
        out = self.norm2(x_norm + y)
        return out, attn


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        attn_layers: List[tf.keras.layers.Layer],
        conv_layers: Optional[List[tf.keras.layers.Layer]] = None,
        norm_layer: Optional[tf.keras.layers.Layer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attn_layers = attn_layers
        self.conv_layers = conv_layers
        self.norm = norm_layer

    def call(
        self,
        x,
        attn_mask=None,
        tau=None,
        delta=None,
        training=None,
        **kwargs
    ):
        attns = []

        if self.conv_layers is not None:
            # attention + conv 交替
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta_in = delta if i == 0 else None
                x, attn = attn_layer(
                    x,
                    attn_mask=attn_mask,
                    tau=tau,
                    delta=delta_in,
                    training=training,
                    **kwargs
                )
                x = conv_layer(x, training=training)
                attns.append(attn)
            # 最后一层 attention（不跟 conv）
            x, attn = self.attn_layers[-1](
                x, tau=tau, delta=None, training=training, **kwargs
            )
            attns.append(attn)
        else:
            # 纯 attention 堆叠
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(
                    x,
                    attn_mask=attn_mask,
                    tau=tau,
                    delta=delta,
                    training=training,
                    **kwargs
                )
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
