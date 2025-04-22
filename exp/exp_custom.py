import os
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from exp.exp_basic import Exp_Basic
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class Dataset_Custom:
    def __init__(self, data, seq_len, pred_len, freq='h', mode='pred', stride=1):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.freq = freq
        self.mode = mode
        self.stride = stride
        self.__read_data__()

    def __read_data__(self):
        if 'date' in self.data.columns:
            cols = self.data.columns[1:]
            self.data_x = self.data[cols].values.astype(np.float32)
            df_stamp = self.data[['date']]
            stamps = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            self.data_stamp = stamps.transpose(1, 0).astype(np.float32)
        else:
            self.data_x = self.data.values.astype(np.float32)
            self.data_stamp = np.zeros((len(self.data_x), 1), dtype=np.float32)

    def __len__(self):
        if self.mode != 'pred':
            return (len(self.data_x) - self.seq_len - self.pred_len + 1) // self.stride
        else:
            return (len(self.data_x) - self.seq_len + 1) // self.stride

    def to_tf_dataset(self, batch_size, shuffle, num_workers=0):
        def gen():
            for idx in range(len(self)):
                if self.mode != 'pred':
                    s = idx * self.stride
                    x = self.data_x[s : s + self.seq_len]
                    y = self.data_x[s + self.seq_len : s + self.seq_len + self.pred_len]
                    xm = self.data_stamp[s : s + self.seq_len]
                    yield x, y, xm
                else:
                    s = idx * self.stride
                    x = self.data_x[s : s + self.seq_len]
                    xm = self.data_stamp[s : s + self.seq_len]
                    yield x, xm

        # build signature
        sample = next(gen())
        if self.mode != 'pred':
            sig = (
                tf.TensorSpec(sample[0].shape, tf.float32),
                tf.TensorSpec(sample[1].shape, tf.float32),
                tf.TensorSpec(sample[2].shape, tf.float32),
            )
        else:
            sig = (
                tf.TensorSpec(sample[0].shape, tf.float32),
                tf.TensorSpec(sample[1].shape, tf.float32),
            )

        ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self))
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


class Exp_Custom(Exp_Basic):
    def __init__(self, args):
        super(Exp_Custom, self).__init__(args)

    def _build_model(self):
        # assume SOFTS.Model returns a tf.keras.Model
        return self.model_dict[self.args.model].Model(self.args)

    def _get_data(self, data, mode, stride=1):
        ds = Dataset_Custom(data, self.args.seq_len, self.args.pred_len,
                            freq=self.args.freq, mode=mode, stride=stride)
        shuffle = (mode == 'train')
        tf_ds = ds.to_tf_dataset(self.args.batch_size, shuffle, self.args.num_workers)
        return ds, tf_ds

    def _select_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)

    def _select_criterion(self):
        return tf.keras.losses.MeanSquaredError()

    def vali(self, vali_ds, criterion):
        metric = tf.keras.metrics.Mean()
        for batch in vali_ds:
            x, y, xm = batch
            preds = self.model([x, xm, None, None], training=False)
            f = -1 if self.args.features == 'MS' else 0
            preds = preds[:, -self.args.pred_len :, f :]
            y = y[:, -self.args.pred_len :, f :]
            loss = criterion(y, preds)
            metric.update_state(loss)
        return metric.result().numpy()

    def train(self, setting, train_data, vali_data=None, test_data=None):
        _, train_ds = self._get_data(train_data, mode='train')
        if vali_data is not None:
            _, vali_ds = self._get_data(vali_data, mode='test')
        if test_data is not None:
            _, test_ds = self._get_data(test_data, mode='test')

        checkpoint_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(checkpoint_dir, exist_ok=True)

        optimizer = self._select_optimizer()
        loss_fn = self._select_criterion()

        best_val = float('inf')
        wait = 0

        for epoch in range(1, self.args.train_epochs + 1):
            start = time.time()
            train_metric = tf.keras.metrics.Mean()

            for step, batch in enumerate(train_ds):
                x, y, xm = batch
                with tf.GradientTape() as tape:
                    preds = self.model([x, xm, None, None], training=True)
                    f = -1 if self.args.features == 'MS' else 0
                    preds = preds[:, -self.args.pred_len :, f :]
                    y_true = y[:, -self.args.pred_len :, f :]
                    loss = loss_fn(y_true, preds)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                train_metric.update_state(loss)

                if (step + 1) % 100 == 0:
                    print(f"\titers: {step+1}, epoch: {epoch} | loss: {train_metric.result().numpy():.7f}")

            train_loss = train_metric.result().numpy()
            vali_loss = None
            test_loss = None

            if vali_data is not None:
                vali_loss = self.vali(vali_ds, loss_fn)
                if vali_loss < best_val:
                    best_val = vali_loss
                    wait = 0
                    self.model.save_weights(os.path.join(checkpoint_dir, 'best.ckpt'))
                else:
                    wait += 1
                    if wait >= self.args.patience:
                        print("Early stopping")
                        break

            if test_data is not None:
                test_loss = self.vali(test_ds, loss_fn)

            elapsed = time.time() - start
            print(f"Epoch {epoch} | Time: {elapsed:.1f}s | Train: {train_loss:.7f} "
                  f"Vali: {vali_loss:.7f} Test: {test_loss:.7f}")

            # 如果需要调整学习率
            try:
                adjust_learning_rate(optimizer, epoch)
            except:
                pass

        # 加载最优权重
        self.model.load_weights(os.path.join(checkpoint_dir, 'best.ckpt'))
        return self.model

    def test(self, setting, test_data, stride=1):
        _, test_ds = self._get_data(test_data, mode='test', stride=stride)
        ckpt = os.path.join(self.args.checkpoints, setting, 'best.ckpt')
        print(f'Loading model from {ckpt}')
        self.model.load_weights(ckpt)

        mse_fn = tf.keras.losses.MeanSquaredError()
        mae_fn = tf.keras.losses.MeanAbsoluteError()
        mse_metric = tf.keras.metrics.Mean()
        mae_metric = tf.keras.metrics.Mean()

        for batch in test_ds:
            x, y, xm = batch
            preds = self.model([x, xm, None, None], training=False)
            f = -1 if self.args.features == 'MS' else 0
            preds = preds[:, -self.args.pred_len :, f :]
            y_true = y[:, -self.args.pred_len :, f :]
            mse_metric.update_state(mse_fn(y_true, preds))
            mae_metric.update_state(mae_fn(y_true, preds))

        print(f"mse: {mse_metric.result().numpy()}, mae: {mae_metric.result().numpy()}")

    def predict(self, setting, pred_data, stride=1):
        _, pred_ds = self._get_data(pred_data, mode='pred', stride=stride)
        ckpt = os.path.join(self.args.checkpoints, setting, 'best.ckpt')
        print(f'Loading model from {ckpt}')
        self.model.load_weights(ckpt)

        all_preds = []
        for batch in pred_ds:
            x, xm = batch
            preds = self.model([x, xm, None, None], training=False)
            f = -1 if self.args.features == 'MS' else 0
            preds = preds[:, -self.args.pred_len :, f :]
            all_preds.append(preds.numpy())

        return np.concatenate(all_preds, axis=0)
