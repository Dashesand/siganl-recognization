import numpy as np
import scipy
from scipy.signal import stft, istft


class SignalAugmentor:
    def __init__(self, snr_range=(10, 20), freq_shift=0.1):
        # 计算噪声水平，将信噪比（SNR）范围转换为线性尺度
        # snr_range: 信噪比范围，单位为dB，通常为负数
        # 10 ** (-np.array(snr_range) / 20): 将dB值转换为线性比例，用于后续的噪声计算
        self.noise_levels = 10  **  (-np.array(snr_range) / 20)

        # 设置频率偏移量
        # freq_shift: 频率偏移量，用于后续的信号处理
        self.freq_shift = freq_shift


    def time_warp(self, signal):
        new_length = int(len(signal) * np.random.uniform(0.8, 1.2))
        return scipy.signal.resample(signal, new_length)

    def freq_mask(self, spec):
        # 随机生成掩码的宽度，宽度为频谱高度的10%到30%之间的随机值
        mask_width = int(spec.shape[0] * np.random.uniform(0.1, 0.3))
        # 随机选择掩码的起始位置，确保掩码不会超出频谱的边界
        start = np.random.randint(0, spec.shape[0] - mask_width)
        # 将选定的频率范围置零
        spec[start:start + mask_width] = 0
        return spec

    def __call__(self, signal):
        # 时域增强
        signal = self.time_warp(signal)

        # 频域增强
        _, _, spec = stft(signal)
        spec = self.freq_mask(spec)
        signal = istft(spec)

        # 添加噪声
        noise = np.random.normal(0, np.random.uniform(*self.noise_levels), len(signal))
        return signal + noise

