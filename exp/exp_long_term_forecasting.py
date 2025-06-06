from data_provider.data_factory import data_provider 
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
import tensorflow as tf
import os
import time
import math
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    # add parameter: strategy
    def __init__(self, args, strategy=None):
        super().__init__(args, strategy)
        
        if self.strategy:
            with self.strategy.scope():
                self.model     = self._build_model()
                self.optimizer = self._select_optimizer()
                self.criterion = self._select_criterion()
        else:
            with tf.device(self.device):
                self.model     = self._build_model()
                self.optimizer = self._select_optimizer()
                self.criterion = self._select_criterion()
    
    @staticmethod
    def _ceil_steps(num_samples, batch_size):
        return math.ceil(num_samples / batch_size)
    
    @staticmethod
    def _smape(y_true, y_pred, eps=1e-6):
        num = tf.abs(y_pred - y_true)
        den = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0 + eps
        return tf.reduce_mean(num / den)

    @staticmethod
    def _mase(y_true, y_pred, seasonal_diff, eps=1e-6):
        """y_* 形状 (B, L, F) 或 (L,)；seasonal_diff 同形状"""
        mae_pred = tf.reduce_mean(tf.abs(y_true - y_pred))
        mae_naive = tf.reduce_mean(tf.abs(seasonal_diff)) + eps
        return mae_pred / mae_naive

    
    def _build_model(self):
        return self.model_dict[self.args.model].Model(self.args)

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return tf.keras.optimizers.Adam(self.args.learning_rate)

    def _select_criterion(self):
        # MSELoss
        # return tf.keras.losses.MeanSquaredError()
        return tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    
    # use tf.function
    @tf.function
    def _train_step(self, bx, by, bxm, bym):
        if bxm is None:
            x_mark, y_mark = None, None
        else:
            x_mark = tf.cast(bxm, tf.float32)
            y_mark = tf.cast(bym, tf.float32)

        dec_inp = tf.zeros_like(by[:, -self.args.pred_len:, :])
        dec_inp = tf.concat([by[:, :self.args.label_len, :], dec_inp], axis=1)

        with tf.GradientTape() as tape:
            if self.args.output_attention:
                outputs, _ = self.model(bx, x_mark, dec_inp, y_mark, training=True)
            else:
                outputs = self.model(bx, x_mark, dec_inp, y_mark, training=True)

            f_dim = -1 if self.args.features == 'MS' else 0
            preds = outputs[:, -self.args.pred_len:, f_dim:]
            targets = by[:, -self.args.pred_len:, f_dim:]

            per_example_loss = self.criterion(targets, preds)
            replicas = (self.strategy.num_replicas_in_sync
                        if self.strategy is not None else 1)
 
            cur_bs      = tf.shape(bx)[0]
            global_bs   = cur_bs * replicas
            # loss = tf.nn.compute_average_loss(per_example_loss,
            #                                 global_batch_size=global_bs)
            per_loss = self.criterion(targets, preds)
            loss = (tf.nn.compute_average_loss(per_loss,
                 global_batch_size=global_bs) if self.strategy
                else tf.reduce_mean(per_loss))

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    @tf.function
    def _val_step(self, bx, by, bxm, bym):
        if bxm is None:
            x_mark, y_mark = None, None
        else:
            x_mark = tf.cast(bxm, tf.float32)
            y_mark = tf.cast(bym, tf.float32)

        dec_inp = tf.zeros_like(by[:, -self.args.pred_len:, :])
        dec_inp = tf.concat([by[:, :self.args.label_len, :], dec_inp], axis=1)

        if self.args.output_attention:
            outputs, _ = self.model(bx, x_mark, dec_inp, y_mark, training=False)
        else:
            outputs = self.model(bx, x_mark, dec_inp, y_mark, training=False)

        f_dim = -1 if self.args.features == 'MS' else 0
        preds = outputs[:, -self.args.pred_len:, f_dim:]
        targets = by[:, -self.args.pred_len:, f_dim:]
        loss = self.criterion(targets, preds)
        # return self.criterion(targets, preds)
        return tf.reduce_mean(loss)

    def vali(self, _, loader, __):
        metric = tf.keras.metrics.Mean()
        ds = loader
        if self.strategy:
            ds = self.strategy.experimental_distribute_dataset(loader)

        for batch in ds:
            bx, by, bxm, bym = batch
            # bx = tf.cast(bx, tf.float32)
            # by = tf.cast(by, tf.float32)
            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                bxm = bym = None
            else:
                bxm = tf.cast(bxm, tf.float32)
                bym = tf.cast(bym, tf.float32)

            if self.strategy:
                per = self.strategy.run(self._val_step, args=(bx, by, bxm, bym))
                loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per, axis=None)
            else:
                loss = self._val_step(bx, by, bxm, bym)
            metric.update_state(loss)

        return float(metric.result().numpy())

    def train(self, setting):

        train_set, train_loader = self._get_data('train')
        _, val_loader = self._get_data('val')
        _, test_loader = self._get_data('test')

        if self.strategy:
            train_loader = self.strategy.experimental_distribute_dataset(train_loader)

        ckpt_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(ckpt_dir, exist_ok=True)

        time_now  = time.time()

        # num_replicas = self.strategy.num_replicas_in_sync
        # train_steps = len(train_set) // num_replicas

        # train_steps = len(train_set)
        train_steps = math.ceil(len(train_set) / self.args.batch_size)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_start = time.time()
            # train_metric = tf.keras.metrics.Mean()

            for step, batch in enumerate(train_loader):
                iter_count += 1

                bx, by, bxm, bym = batch
                # bx = tf.cast(bx, tf.float32)
                # by = tf.cast(by, tf.float32)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    bxm = bym = None
                else:
                    bxm = tf.cast(bxm, tf.float32)
                    bym = tf.cast(bym, tf.float32)

                if self.strategy:
                    per = self.strategy.run(self._train_step, args=(bx, by, bxm, bym))
                    loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per, axis=None)
                else:
                    loss = self._train_step(bx, by, bxm, bym)

                # train_metric.update_state(loss)
                loss_value = float(loss)

                if (step + 1) % 100 == 0:
                    # avg_loss = train_metric.result().numpy()
                    train_loss.append(loss_value)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(step + 1, epoch + 1, loss_value))

                    # remaining = train_steps - (step + 1)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - step)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

                    iter_count = 0
                    time_now   = time.time()
                    # train_metric.reset_states()
            
            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, time.time() - epoch_start))

            # tr_loss = train_metric.result().numpy()
            tr_loss = np.average(train_loss) if train_loss else float('nan')
            val_loss = self.vali(None, val_loader, None)
            test_loss = self.vali(None, test_loader, None)
            # print(f"Epoch {epoch+1} | Train: {tr_loss:.7f}, Val: {val_loss:.7f}, Test: {test_loss:.7f}")
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, tr_loss, val_loss, test_loss))

            early_stopping(val_loss, self.model, ckpt_dir)
            if early_stopping.early_stop:
                print("Early stopping.")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        # self.model.save_weights(os.path.join(ckpt_dir, 'checkpoint.h5'))
        best_ckpt = os.path.join(ckpt_dir, 'checkpoint.h5')

        self.model.load_weights(best_ckpt)

        if not self.args.save_model:
            import shutil
            shutil.rmtree(ckpt_dir)

        return self.model

    # def test(self, setting, test=0):
    #     _, test_loader = self._get_data('test')
    #     if self.strategy:
    #         test_loader = self.strategy.experimental_distribute_dataset(test_loader)

    #     if test:
    #         ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.h5')
    #         print('loading model from', ckpt)
    #         self.model.load_weights(ckpt)

    #     mse_loss = tf.keras.losses.MeanSquaredError()
    #     mae_loss = tf.keras.losses.MeanAbsoluteError()
    #     mse = AverageMeter()
    #     mae = AverageMeter()

    #     self.model.trainable = False
    #     for batch in test_loader:
    #         batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    #         # batch_x = tf.cast(batch_x, tf.float32)
    #         # batch_y = tf.cast(batch_y, tf.float32)

    #         if 'PEMS' in self.args.data or 'Solar' in self.args.data:
    #             batch_x_mark = None
    #             batch_y_mark = None
    #         else:
    #             batch_x_mark = tf.cast(batch_x_mark, tf.float32)
    #             batch_y_mark = tf.cast(batch_y_mark, tf.float32)

    #         zeros = tf.zeros_like(batch_y[:, -self.args.pred_len:, :])
    #         dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], zeros], axis=1)

    #         outputs = self.model(
    #             batch_x, batch_x_mark, dec_inp, batch_y_mark,
    #             training=False
    #         )
    #         if self.args.output_attention:
    #             outputs = outputs[0]

    #         f_dim = -1 if self.args.features == 'MS' else 0
    #         outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #         batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]

    #         batch_size = int(batch_x.shape[0])
    #         mse.update(mse_loss(batch_y_cut, outputs).numpy(), batch_size)
    #         mae.update(mae_loss(batch_y_cut, outputs).numpy(), batch_size)

    #     print(f'mse:{mse.avg}, mae:{mae.avg}')
    #     return
    def test(self, setting, test=0):
        _, test_loader = self._get_data('test')
        if self.strategy:
            test_loader = self.strategy.experimental_distribute_dataset(test_loader)

        if test:
            ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.h5')
            print('loading model from', ckpt)
            self.model.load_weights(ckpt)

        mse_metric = tf.keras.metrics.Mean()
        mae_metric = tf.keras.metrics.Mean()
        # smape_metric = tf.keras.metrics.Mean()
        # mase_metric  = tf.keras.metrics.Mean()

        # m = getattr(self.args, 'seasonal', 1)
        @tf.function
        def _test_step(bx, by, bxm, bym):
            # 与 _val_step 同一套前处理
            if bxm is None:
                x_mark, y_mark = None, None
            else:
                x_mark = tf.cast(bxm, tf.float32)
                y_mark = tf.cast(bym, tf.float32)

            dec_inp = tf.zeros_like(by[:, -self.args.pred_len:, :])
            dec_inp = tf.concat([by[:, :self.args.label_len, :], dec_inp], axis=1)

            if self.args.output_attention:
                outputs, _ = self.model(bx, x_mark, dec_inp, y_mark, training=False)
            else:
                outputs = self.model(bx, x_mark, dec_inp, y_mark, training=False)

            f_dim   = -1 if self.args.features == 'MS' else 0
            preds   = outputs[:, -self.args.pred_len:, f_dim:]      # (B,L,F) 或 (B,L)
            targets =     by[:, -self.args.pred_len:, f_dim:]

            # # naive predict
            # naive_targets = by[:, -(self.args.pred_len+m):-m if m>0 else None, f_dim:]
            # naive_preds = tf.identity(naive_targets)

            # per sample
            mse_val = tf.reduce_mean(tf.math.squared_difference(targets, preds))
            mae_val = tf.reduce_mean(tf.math.abs(targets - preds))

            return mse_val, mae_val

            # smape = self._smape(targets, preds)
            # seasonal_diff = targets - naive_targets
            # mase  = self._mase(targets, preds, seasonal_diff)

            # smape_naive = self._smape(targets, naive_preds)
            # mase_naive  = self._mase(targets, naive_preds, seasonal_diff)

            # owa = 0.5 * (smape / smape_naive) + 0.5 * (mase / mase_naive)

            # return mse_val, mae_val, smape, mase, owa
        
        self.model.trainable = False
        # owa_metric = tf.keras.metrics.Mean()
        for batch in test_loader:
            bx, by, bxm, bym = batch
            if self.args.data in ('PEMS', 'Solar'):
                bxm = bym = None
            else:
                bxm = tf.cast(bxm, tf.float32)
                bym = tf.cast(bym, tf.float32)

            # 分布式或单卡统一处理
            if self.strategy:
                mse_rep, mae_rep = self.strategy.run(_test_step, args=(bx, by, bxm, bym))
                # mse_rep, mae_rep, smape_rep, mase_rep, owa_rep = \
                    # self.strategy.run(_test_step, args=(bx, by, bxm, bym))
                mse_val = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, mse_rep, axis=None)
                mae_val = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, mae_rep, axis=None)
                # smape_val = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, smape_rep, axis=None)
                # mase_val = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, mase_rep, axis=None)
                # owa_val = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, owa_rep, axis=None)
            else:
                mse_val, mae_val = _test_step(bx, by, bxm, bym)
                # mse_val, mae_val, smape_val, mase_val, owa_val = _test_step(bx, by, bxm, bym)

            mse_metric.update_state(mse_val)
            mae_metric.update_state(mae_val)

        print(f"mse: {mse_metric.result().numpy():.6f}, " f"mae: {mae_metric.result().numpy():.6f}")
        return