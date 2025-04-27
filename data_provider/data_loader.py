import os
import warnings
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence as Dataset

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

def _to_tf(*arrays, dtype=tf.float32):
    return [tf.convert_to_tensor(a, dtype=dtype) for a in arrays]


class _BaseTS(Dataset):
    """final label_len as decoder input, pred_len as output step 
    common to all datasets helper: segment data and implement generic __getitem__/__len__."""

    seq_len: int
    label_len: int
    pred_len: int

    def __getitem__(self, idx: int):
        # window index → encoder / decoder range
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x      = tf.constant(self.data_x[s_begin:s_end],      dtype=tf.float32)
        seq_y      = tf.constant(self.data_y[r_begin:r_end],      dtype=tf.float32)
        seq_x_mark = tf.constant(self.data_stamp[s_begin:s_end],  dtype=tf.float32)
        seq_y_mark = tf.constant(self.data_stamp[r_begin:r_end],  dtype=tf.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray):
        return self.scaler.inverse_transform(data)


# ===================================================
# ETTh/ETTm
# ===================================================

class Dataset_ETT_hour(_BaseTS):
    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: Optional[Tuple[int, int, int]] = None,
        features: str = "S",
        data_path: str = "ETTh1.csv",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        seasonal_patterns=None
    ):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features, self.target = features, target
        self.scale, self.timeenc, self.freq = scale, timeenc, freq

        self.root_path, self.data_path = root_path, data_path
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 以 12 个月为一个训练周期
        border1s = [0, 12 * 30 * 24 - self.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24,
                    12 * 30 * 24 + 4 * 30 * 24,
                    12 * 30 * 24 + 8 * 30 * 24]
        b1, b2 = border1s[self.set_type], border2s[self.set_type]

        # 特征列选择
        if self.features in ("M", "MS"):
            df_data = df_raw[df_raw.columns[1:]]
        else:  # 'S'
            df_data = df_raw[[self.target]]

        # 归一化
        if self.scale:
            self.scaler.fit(df_data.values[border1s[0]:border2s[0]])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 时间戳处理
        df_stamp = df_raw[["date"]][b1:b2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            df_stamp["weekday"] = df_stamp.date.dt.weekday
            df_stamp["hour"] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(["date"], axis=1).values
        else:  # 使用公式时间编码
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                       freq=self.freq).T

        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]
        self.data_stamp = data_stamp

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset_ETT_hour):
    """分钟级（ETTm）— 仅覆盖不同的 freq 与边界切分。"""

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        seasonal_patterns=None
    ):
        super().__init__(root_path, flag, size, features,
                         data_path, target, scale, timeenc, freq, seasonal_patterns=None)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 分钟级：一天 96 条
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        b1, b2 = border1s[self.set_type], border2s[self.set_type]

        if self.features in ("M", "MS"):
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values[border1s[0]:border2s[0]])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 时间戳
        df_stamp = df_raw[["date"]][b1:b2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            df_stamp["weekday"] = df_stamp.date.dt.weekday
            df_stamp["hour"] = df_stamp.date.dt.hour
            df_stamp["minute"] = (df_stamp.date.dt.minute // 15)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                       freq=self.freq).T

        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]
        self.data_stamp = data_stamp


class Dataset_Custom(_BaseTS):
    def __init__(
        self, root_path, flag="train", size=None,
        features="S", data_path="ETTh1.csv",
        target="OT", scale=True, timeenc=0, freq="h", seasonal_patterns=None
    ):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features, self.target = features, target
        self.scale, self.timeenc, self.freq = scale, timeenc, freq

        self.root_path, self.data_path = root_path, data_path
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        b1, b2 = border1s[self.set_type], border2s[self.set_type]

        if self.features in ("M", "MS"):
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values[border1s[0]:border2s[0]])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][b1:b2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            df_stamp["weekday"] = df_stamp.date.dt.weekday
            df_stamp["hour"] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(["date"], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                       freq=self.freq).T

        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]
        self.data_stamp = data_stamp

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_M4(_BaseTS):
    def __init__(
        self, root_path, flag="train", size=None,
        features="S", data_path="ETTh1.csv",
        target="OT", scale=True, timeenc=0, freq="h", seasonal_patterns=None
    ):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features, self.target = features, target
        self.scale, self.timeenc, self.freq = scale, timeenc, freq

        self.root_path, self.data_path = root_path, data_path
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove("date")
        if self.cols:
            cols = self.cols.copy()
            # 如果用户手动给了 cols，却忘了包含 target，就补上
            if self.target not in cols and self.target in df_raw.columns:
                cols.append(self.target)
        else:
            cols = list(df_raw.columns)
            if "date" in cols:  # 防止没有 date 列时报 KeyError
                cols.remove("date")
        

        df_raw = df_raw[["date"] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        b1, b2 = border1s[self.set_type], border2s[self.set_type]

        if self.features in ("M", "MS"):
            # df_data = df_raw[df_raw.columns[1:]]
            df_data = [c for c in cols if c != self.target]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values[border1s[0]:border2s[0]])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][b1:b2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            df_stamp["weekday"] = df_stamp.date.dt.weekday
            df_stamp["hour"] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(["date"], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                       freq=self.freq).T

        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]
        self.data_stamp = data_stamp

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Random(_BaseTS):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', **kwargs):
        # size = (seq, label, pred, channel)
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
            self.n_channel = 512
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            self.n_channel = size[3]

        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.DataFrame(
            np.random.rand(10000 + self.seq_len + self.pred_len, self.n_channel + 1),
            columns=[f"f{i}" for i in range(self.n_channel + 1)]
        )

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        b1, b2 = border1s[self.set_type], border2s[self.set_type]

        df_data = df_raw.values
        
        if self.features in ("M", "MS"):
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values[border1s[0]:border2s[0]])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]

    def __getitem__(self, idx: int):
        s_b = idx; s_e = s_b + self.seq_len
        r_b = s_e - self.label_len
        r_e = r_b + self.label_len + self.pred_len

        seq_x = tf.constant(self.data_x[s_b:s_e], dtype=tf.float32)
        seq_y = tf.constant(self.data_y[r_b:r_e], dtype=tf.float32)
        # placeholder stamps
        seq_x_mark = tf.zeros((self.seq_len, 4), dtype=tf.float32)
        seq_y_mark = tf.zeros((self.label_len + self.pred_len, 4), dtype=tf.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(_BaseTS):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', **kwargs):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        # self.scale = True
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, idx: int):
        s_b, s_e = idx, idx + self.seq_len
        r_b = s_e - self.label_len
        r_e = r_b + self.label_len + self.pred_len

        seq_x = tf.constant(self.data_x[s_b:s_e], dtype=tf.float32)
        seq_y = tf.constant(self.data_y[r_b:r_e], dtype=tf.float32)
        seq_x_mark = tf.zeros((self.seq_len, 1), dtype=tf.float32)
        seq_y_mark = tf.zeros((self.label_len + self.pred_len, 1), dtype=tf.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Solar(_BaseTS):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', **kwargs):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    # def __getitem__(self, idx):
    #     s_begin = idx
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len
    #     r_end = r_begin + self.label_len + self.pred_len

    #     seq_x = self.data_x[s_begin:s_end]
    #     seq_y = self.data_y[r_begin:r_end]
    #     seq_x_mark = np.zeros((seq_x.shape[0], 1))
    #     seq_y_mark = np.zeros((seq_y.shape[0], 1))
    #     return tuple(_to_tf(seq_x, seq_y, seq_x_mark, seq_y_mark))
    def __getitem__(self, idx: int):
        s_b, s_e = idx, idx + self.seq_len
        r_b, r_e = s_e - self.label_len, s_e - self.label_len + self.label_len + self.pred_len

        seq_x = tf.constant(self.data_x[s_b:s_e], dtype=tf.float32)
        seq_y = tf.constant(self.data_y[r_b:r_e], dtype=tf.float32)
        seq_x_mark = tf.zeros((self.seq_len, 1), dtype=tf.float32)
        seq_y_mark = tf.zeros((self.label_len + self.pred_len, 1), dtype=tf.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Pred(_BaseTS):
    """Exclusive to the prediction stage: the last label_len is the decoder input, and pred_len is the output step."""

    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, **kwargs):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 生成未来时间戳
        tmp_stamp = df_raw[["date"]][border1:border2]
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            df_stamp["weekday"] = df_stamp.date.dt.weekday
            df_stamp["hour"] = df_stamp.date.dt.hour
            df_stamp["minute"] = (df_stamp.date.dt.minute // 15)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                       freq=self.freq).T

        self.data_x = data[border1:border2]
        self.data_y = df_data.values[border1:border2] if self.inverse else self.data_x
        self.data_stamp = data_stamp

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def __getitem__(self, idx: int):
        s_b = idx
        s_e = s_b + self.seq_len
        r_b = s_e - self.label_len  # decoder input starts label_len before end
        r_e = r_b + self.label_len + self.pred_len

        seq_x      = tf.constant(self.data_x[s_b:s_e], dtype=tf.float32)
        if self.inverse:
            seq_y = tf.constant(self.data_x[r_b:r_b + self.label_len], dtype=tf.float32)
        else:
            seq_y      = tf.constant(self.data_y[r_b:r_b + self.label_len], dtype=tf.float32)
        seq_x_mark = tf.constant(self.data_stamp[s_b:s_e], dtype=tf.float32)
        seq_y_mark = tf.constant(self.data_stamp[r_b:r_e], dtype=tf.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
