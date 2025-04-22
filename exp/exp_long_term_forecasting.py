from data_provider.data_factory import data_provider 
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
import tensorflow as tf
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        # 构造 Keras 模型
        model = self.model_dict[self.args.model].Model(self.args)
        # 支持多 GPU
        if self.args.use_multi_gpu and self.args.use_gpu:
            strategy = tf.distribute.MirroredStrategy(
                devices=[f"/gpu:{i}" for i in self.args.device_ids]
            )
            with strategy.scope():
                model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        # 返回 (原始 dataset, tf.data.Dataset loader)
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return tf.keras.optimizers.Adam(self.args.learning_rate)

    def _select_criterion(self):
        # MSELoss
        return tf.keras.losses.MeanSquaredError()

    def vali(self, vali_data, vali_loader, criterion):
        """验证集评估"""
        total_loss = AverageMeter()
        # 评估模式
        for batch in vali_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = tf.cast(batch_x, tf.float32)
            batch_y = tf.cast(batch_y, tf.float32)

            # 对于 PEMS/Solar，无需时序标记
            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = tf.cast(batch_x_mark, tf.float32)
                batch_y_mark = tf.cast(batch_y_mark, tf.float32)

            # decoder 输入
            zeros = tf.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], zeros], axis=1)

            # 前向
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                training=False
            )
            if self.args.output_attention:
                outputs = outputs[0]

            # 仅保留最后 pred_len 步 + feature dim
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]

            # 计算 loss
            loss = criterion(batch_y_cut, outputs)
            batch_size = int(batch_x.shape[0])
            total_loss.update(float(loss.numpy()), batch_size)

        return total_loss.avg

    def train(self, setting):
        # 获取数据
        _, train_loader = self._get_data('train')
        _, vali_loader  = self._get_data('val')
        _, test_loader  = self._get_data('test')

        # 创建 checkpoint 目录
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # 训练准备
        train_steps = tf.data.experimental.cardinality(train_loader).numpy()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        time_now = time.time()
        for epoch in range(1, self.args.train_epochs + 1):
            iter_count = 0
            train_loss_list = []
            epoch_start = time.time()

            for step, batch in enumerate(train_loader, start=1):
                iter_count += 1
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = tf.cast(batch_x, tf.float32)
                batch_y = tf.cast(batch_y, tf.float32)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = tf.cast(batch_x_mark, tf.float32)
                    batch_y_mark = tf.cast(batch_y_mark, tf.float32)

                zeros = tf.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], zeros], axis=1)

                # 梯度更新
                with tf.GradientTape() as tape:
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        training=True
                    )
                    if self.args.output_attention:
                        outputs = outputs[0]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(batch_y_cut, outputs)

                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # 进度打印（每 100 step）
                if step % 100 == 0:
                    loss_val = float(loss.numpy())
                    train_loss_list.append(loss_val)

                    # 计算每步耗时
                    speed = (time.time() - time_now) / iter_count
                    # 只计算本 epoch 内的剩余步数，避免负数
                    remaining_steps = train_steps - step
                    left_time = speed * remaining_steps

                    print(f"\titers: {step}, epoch: {epoch} | loss: {loss_val:.7f}")
                    print(f'\tspeed: {speed:.4f}s/iter; left time this epoch: {left_time:.4f}s')

                    iter_count = 0
                    time_now = time.time()

            # epoch 结束
            epoch_time = time.time() - epoch_start
            print(f"Epoch: {epoch} cost time: {epoch_time:.1f}s")

            # 汇总损失
            train_loss = np.mean(train_loss_list) if train_loss_list else 0.0
            vali_loss  = self.vali(None, vali_loader, criterion)
            test_loss  = self.vali(None, test_loader, criterion)
            print(
                f"Epoch: {epoch}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} "
                f"Test Loss: {test_loss:.7f}"
            )

            # EarlyStopping
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            adjust_learning_rate(optimizer, epoch, self.args)

        # 加载最佳权重
        best_path = os.path.join(path, 'checkpoint.h5')
        self.model.load_weights(best_path)

        # 可选删除目录
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)

        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data('test')
        if test:
            ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.h5')
            print('loading model from', ckpt)
            self.model.load_weights(ckpt)

        mse_loss = tf.keras.losses.MeanSquaredError()
        mae_loss = tf.keras.losses.MeanAbsoluteError()
        mse = AverageMeter()
        mae = AverageMeter()

        self.model.trainable = False
        for batch in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = tf.cast(batch_x, tf.float32)
            batch_y = tf.cast(batch_y, tf.float32)

            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = tf.cast(batch_x_mark, tf.float32)
                batch_y_mark = tf.cast(batch_y_mark, tf.float32)

            zeros = tf.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], zeros], axis=1)

            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                training=False
            )
            if self.args.output_attention:
                outputs = outputs[0]

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]

            batch_size = int(batch_x.shape[0])
            mse.update(mse_loss(batch_y_cut, outputs).numpy(), batch_size)
            mae.update(mae_loss(batch_y_cut, outputs).numpy(), batch_size)

        print(f'mse:{mse.avg}, mae:{mae.avg}')
        return
