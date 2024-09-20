from read_data import read_all_file_df
from db_contlorer import DbController
import numpy as np
import matplotlib.pyplot as plt
import pywt


class DataHandler:
    def __init__(self, db_controller, channels_names, idx_exp=[3, 4], idx_people=[1, 2]):
        self.df = read_all_file_df(channels_names =channels_names, idx_exp=idx_exp, idx_people= idx_people)
        self.db_controller = db_controller

    def make_cwt_transform(self, signal: np.array, wavelet_type, fmin, fmax, n_frex, length=640):
        time = np.linspace(0, length / 160, length)  # time of the signal , 160 is the sampling rate
        widths = np.geomspace(fmin, fmax, num=n_frex)  # range of scales
        sampling_period = np.diff(time).mean()  # 0.006251562890722681

        cwtmatr, freqs = pywt.cwt(signal, widths, wavelet_type, sampling_period=sampling_period)
        # cwtmatr= np.abs(cwtmatr[:-1,:-1])
        cwtmatr = np.abs(cwtmatr)
        # plot the wavelet transform
        plt.figure(figsize=(20, 3))
        plt.pcolormesh(time, freqs, cwtmatr)
        plt.yscale("log")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.title(f"Wavelet Transform with {'cgau4'}")
        plt.colorbar()
        plt.show()
        return cwtmatr

    def make_chanks(self):
        pass

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

#
# channels_names = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
# channels_names = ['Fc5.', 'Af8.', 'P6..', 'P8..', 'Iz..']
#
#
# db = DbController(dbname="my_db", user="user", password="1234", host="localhost", port="5433")
# cwt_data = DataHandler(db_controller=db, channels_names=channels_names)
# print(cwt_data.df)
