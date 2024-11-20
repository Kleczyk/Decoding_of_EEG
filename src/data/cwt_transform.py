import polars as pl
import numpy as np
import mne
from tqdm import tqdm
import time
import os
import sys
import pywt
import matplotlib.pyplot as plt

import data.read_data_polars as rdp


class CwtTransform:
    def __init__(
        self, fmin: int, fmax: int, n_frex: int, wavelet_type: str, seq_length: int
    ):
        self.fmin = fmin
        self.fmax = fmax
        self.n_frex = n_frex
        self.wavelet_type = wavelet_type
        self.seq_length = seq_length

    def transform(self, signal: np.array):
        """
        This function takes in a dataframe and returns the continuous wavelet transform of the signal
        """
        time = np.linspace(
            0, self.seq_length / 160, self.seq_length
        )  # time of the signal , 160 is the sampling rate
        widths = np.geomspace(self.fmin, self.fmax, num=self.n_frex)  # range of scales
        sampling_period = np.diff(time).mean()  # 0.006251562890722681
        cwtmatr, freqs = pywt.cwt(
            signal, widths, self.wavelet_type, sampling_period=sampling_period
        )
        cwtmatr= np.abs(cwtmatr)
        return cwtmatr

    def plot_transform(self, signal: np.array):
        """
        This function takes in a dataframe and returns the continuous wavelet transform of the signal
        """
        time = np.linspace(
            0, self.seq_length / 160, self.seq_length
        )  # time of the signal , 160 is the sampling rate
        widths = np.geomspace(self.fmin, self.fmax, num=self.n_frex)  # range of scales
        sampling_period = np.diff(time).mean()  # 0.006251562890722681
        print(signal.shape)
        cwtmatr, freqs = pywt.cwt(
            signal, widths, self.wavelet_type, sampling_period=sampling_period
        )
        # cwtmatr= np.abs(cwtmatr[:-1,:-1])
        cwtmatr = np.abs(cwtmatr)

        # plot the wavelet transform
        plt.figure(figsize=(20, 3))
        print(cwtmatr.shape)
        print(time.shape)
        print(freqs.shape)
        plt.pcolormesh(time, freqs, cwtmatr)
        maxval = np.max(freqs)
        plt.yscale("log")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.title(f"Wavelet Transform with {'cgau4'}")
        plt.colorbar()
        plt.show()
        return cwtmatr

    def get_sequence_from_df(self, df: pl.DataFrame, idx: int):
        """
        This function takes in a dataframe and returns a sequence of length sequence_length starting from idx
        """
        return df[idx : idx + self.seq_length]
