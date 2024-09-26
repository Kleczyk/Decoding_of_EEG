from .read_data import read_all_file_df
from .db_contlorer import DbController
import numpy as np
import matplotlib.pyplot as plt
import pywt
import datetime


class DataHandler:
    def __init__(self, db_controller: DbController, channels_names: list, idx_exp: list = [3, 4],
                 idx_people: list = [1, 2], chunks_size: int = 640, wavelet_type: str = 'cgau4', fmin: int = 1,
                 fmax: int = 200, n_frex: int = 20):
        self.df = read_all_file_df(channels_names=channels_names, idx_exp=idx_exp, idx_people=idx_people)
        self.db_controller = db_controller
        self.chunks_size = chunks_size
        self.wavelet_type = wavelet_type
        self.fmin = fmin
        self.fmax = fmax
        self.n_frex = n_frex
        self.signal = None

    def make_cwt_transform(self) -> np.array:
        time = np.linspace(0, self.chunks_size / 160, self.chunks_size)  # time of the signal , 160 is the sampling rate
        widths = np.geomspace(self.fmin, self.fmax, num=self.n_frex)  # range of scales
        sampling_period = np.diff(time).mean()  # 0.006251562890722681
        cwtmatr, freqs = pywt.cwt(self.signal, widths, self.wavelet_type, sampling_period=sampling_period)
        # cwtmatr= np.abs(cwtmatr[:-1,:-1])
        cwtmatr = np.abs(cwtmatr)
        return cwtmatr

    def write2db(self, table: str):
        idx_s = 0
        idx_e = idx_s + self.chunks_size

        while idx_e < len(self.df):

            self.signal = self.df.iloc[idx_s:idx_e].values

            target = self.df.iloc[idx_s:idx_e]["target"].to_numpy()
            cwt_signal = self.make_cwt_transform()
            cwt_signal = cwt_signal.transpose(1, 0, 2)
            self.db_controller.insert_data_own_time(table, cwt_signal, target, idx_start=idx_s)
            idx_s = idx_e
            if idx_e + self.chunks_size > len(self.df):
                idx_s = len(self.df)
            idx_e = idx_s + self.chunks_size
            print(f"{idx_s}/{len(self.df)} {idx_s / len(self.df) * 100:.2f}")

    def plot_cwt(self, signal, wavelet_type, fmin, fmax, n_frex, length):
        time = np.linspace(0, length / 160, length)  # time of the signal , 160 is the sampling rate
        widths = np.geomspace(fmin, fmax, num=n_frex)  # range of scales
        sampling_period = np.diff(time).mean()  # 0.006251562890722681
        cwtmatr, freqs = pywt.cwt(signal, widths, wavelet_type, sampling_period=sampling_period)

        cwtmatr = np.abs(cwtmatr)
        # plot the wavelet transform
        plt.figure(figsize=(20, 3))
        plt.pcolormesh(time, freqs, cwtmatr)
        maxval = np.max(freqs)
        plt.yscale("log")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.title(f"Wavelet Transform with {'cgau4'}")
        plt.colorbar()
        plt.show()

    def cwt_data2db(self, db_controller, table):
        for i in range(len(self.df)):
            self.db_controller.insert_data(table, self.df.iloc[i].values, self.df.iloc[i]["target"])

    def get_data(self):
        return self.df.drop(columns=["target"]).values

    def get_target(self):
        return self.df["target"].values
