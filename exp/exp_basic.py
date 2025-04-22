import os

import tensorflow as tf

from models import SOFTS


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SOFTS': SOFTS,
        }
        self.device = self._acquire_device()
        with tf.device(self.device):
            self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        # 选择 CPU / GPU，并与原有逻辑保持一致
        if self.args.use_gpu and tf.config.list_physical_devices('GPU'):
            gpus = tf.config.list_physical_devices('GPU')

            if self.args.use_multi_gpu:
                dev_ids = [int(i) for i in self.args.devices.split(',')]
            else:
                dev_ids = [self.args.gpu]

            visible_gpus = [gpus[i] for i in dev_ids]
            tf.config.set_visible_devices(visible_gpus, 'GPU')

            for gpu in visible_gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass

            device = f'/GPU:{dev_ids[0]}'
            print(f'Use GPU: {device}')
        else:
            # 显式禁用所有 GPU
            tf.config.set_visible_devices([], 'GPU')
            device = '/CPU:0'
            print('Use CPU')

        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
