import math
import os

import numpy as np
import tensorflow as tf


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer: tf.keras.optimizers.Optimizer, epoch: int, args):
    """
    手动调整 tf.keras 优化器的学习率
    args.lradj: 'type1' | 'type2' | 'constant' | 'cosine'
    """
    if args.lradj == 'type1':
        lr_adjust = {
            epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))
        }
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'constant':
        lr_adjust = {}
    elif args.lradj == 'cosine':
        lr_adjust = {
            epoch: args.learning_rate / 2 * (
                1 + math.cos(epoch / args.train_epochs * math.pi)
            )
        }
    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        new_lr = lr_adjust[epoch]
        # 如果 learning_rate 是个 Python float
        if isinstance(optimizer.learning_rate, tf.Variable):
            optimizer.learning_rate.assign(new_lr)
        else:
            optimizer.learning_rate = new_lr
        print(f'Updating learning rate to {new_lr}')


class EarlyStopping:
    """
    简易版 EarlyStopping，保存最优权重到 `path/checkpoint`
    """
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss: float, model: tf.keras.Model, path: str):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model, path)
            self.counter = 0

    def _save_checkpoint(self, val_loss: float, model: tf.keras.Model, path: str):
        if self.verbose:
            print(
                f'Validation loss decreased '
                f'({self.val_loss_min:.6f} → {val_loss:.6f}).  Saving model ...'
            )
        os.makedirs(path, exist_ok=True)
        # 保存 TensorFlow 模型权重
        filepath = os.path.join(path, 'checkpoint.h5')
        model.save_weights(filepath, save_format='h5')
        self.val_loss_min = val_loss
