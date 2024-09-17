from read_data import read_all_file_df
from db_contlorer import DbController
import numpy as np
import matplotlib.pyplot as plt
import pywt


class DataHandler:
    def __init__(self, db_controller, num_exp=[3, 4], num_people=[1, 2]):
        self.df = read_all_file_df(num_exp, num_people)
        self.db_controller = db_controller

    def make_cwt_transform(self, signal:np.array, wavelet_type, fmin, fmax, n_frex, length=640  ):
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


db = DbController(dbname="my_db", user="user", password="1234", host="localhost", port="5433")
cwt_data = DataHandler(db_controller=db)
cwt_data.plot_cwt(cwt_data.df['Fc5'][0:640].to_numpy(), 'cgau4', 1, 200, 100, len(cwt_data.df['Fc5'][0:640].to_numpy()))
print(cwt_data.df['Fc5'][0:640].to_numpy().shape)


