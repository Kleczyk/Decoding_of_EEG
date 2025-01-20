import pandas as pd
import matplotlib

import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_signal_3d_separate_windows(
    df,
    sensor_column,
    target_column='target',
    sampling_frequency=160,
    duration=40
):
    """
    Rysuje:
      - Sygnał pokolorowany klasami w płaszczyźnie z=0.
      - Każdy przedział (np. granica klasy) jako osobny biały prostokąt
        w kolejnych warstwach z = 1, 2, 3, ...

    Parametry:
    ----------
    df : pd.DataFrame
        Dane z sygnałem i kolumną target
    sensor_column : str
        Nazwa kolumny z sygnałem
    target_column : str
        Nazwa kolumny z klasą
    sampling_frequency : int
        Częstotliwość próbkowania (Hz)
    duration : int
        Czas w sekundach, ile fragmentu sygnału rysujemy

    Zwraca:
    ----------
    None
    """

    # ------------------ Przygotowanie danych ------------------
    num_samples = sampling_frequency * duration
    df_filtered = df.iloc[:num_samples].copy()

    # Oś czasu (w sekundach)
    time_values = np.arange(0, len(df_filtered)) / sampling_frequency

    # Określamy minimum i maksimum sygnału (potrzebne do narysowania wysokości prostokąta)
    sig_min = df_filtered[sensor_column].min()
    sig_max = df_filtered[sensor_column].max()

    # Chcemy podzielić dane według spójnych fragmentów klas (każda klasa tworzy "grupę")
    df_filtered['group'] = (
        df_filtered[target_column]
        .shift(fill_value=df_filtered[target_column].iloc[0])
        .ne(df_filtered[target_column])
        .cumsum()
    )

    # Przygotowujemy kolory dla klas
    classes = df_filtered[target_column].unique()
    cmap = plt.colormaps.get_cmap('tab10')
    class_to_color = {cls: cmap(i) for i, cls in enumerate(classes)}

    # ------------------ Funkcja pomocnicza do rysowania prostokąta w 3D ------------------
    def add_window_rect(
        ax,
        start_time,
        end_time,
        y_min,
        y_max,
        z_value,
        facecolor='white',
        edgecolor='black',
        alpha=0.3
    ):
        """
        Rysuje biały prostokąt w płaszczyźnie z = z_value
        (od start_time do end_time na osi X i od y_min do y_max na osi Y).
        """
        corners = [
            (start_time, y_min, z_value),
            (end_time,   y_min, z_value),
            (end_time,   y_max, z_value),
            (start_time, y_max, z_value)
        ]
        poly = Poly3DCollection([corners],
                                facecolors=facecolor,
                                edgecolors=edgecolor,
                                alpha=alpha)
        ax.add_collection3d(poly)

    # ------------------ Tworzenie figury 3D ------------------
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    # ------------------ 1) Rysowanie sygnału (z=0) ------------------
    plotted_classes = set()
    for _, group_data in df_filtered.groupby('group'):
        cls = group_data[target_column].iloc[0]
        color = class_to_color[cls]

        ax.plot(
            time_values[group_data.index],      # x - czas
            group_data[sensor_column],          # y - wartość sygnału
            zs=0,                               # z=0 => płaszczyzna XY
            zdir='z',
            color=color,
            label=f"Class {cls}" if cls not in plotted_classes else None
        )
        plotted_classes.add(cls)

    # ------------------ 2) Każdy przedział w kolejnej warstwie z=i+1 ------------------
    #    Tutaj: "przedział" definiujemy jako spójny fragment klasy
    #    (czyli group_id=1 => warstwa z=1, group_id=2 => warstwa z=2, ...)

    # Sortujemy po kolei rosnąco, żeby warstwy były w naturalnej kolejności
    groups_sorted = sorted(df_filtered['group'].unique())

    for i, g_id in enumerate(groups_sorted, start=1):
        group_data = df_filtered[df_filtered['group'] == g_id]
        start_idx = group_data.index[0]
        end_idx   = group_data.index[-1]

        start_t = start_idx / sampling_frequency
        end_t   = (end_idx + 1) / sampling_frequency  # +1 aby "zahaczyć" o ostatnią próbkę

        # Rysujemy "biały prostokąt" w warstwie z=i
        add_window_rect(
            ax,
            start_time=start_t,
            end_time=end_t,
            y_min=sig_min,
            y_max=sig_max,
            z_value=i,
            facecolor='white',
            edgecolor='black',
            alpha=0.3
        )

        # Możemy też dodać opis okna w 3D (np. "Window i (Class X)"):
        cls_name = group_data[target_column].iloc[0]
        ax.text(
            0.5 * (start_t + end_t),            # x - środek okna
            sig_max,                            # y - na górze sygnału
            i,                                  # z - ta warstwa
            f"Window {i} (Class {cls_name})",
            color='black',
            ha='center',
            va='bottom'
        )



    # ------------------ Ostatnie szlify ------------------
    ax.set_title("Sygnał (z=0) + każda grupa/okno w osobnej warstwie Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal amplitude")
    ax.set_zlabel("Window index (Z)")
    ax.legend(loc='upper left')

    # Kąt widoku, żeby w miarę czytelnie wyglądało
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.show()


import data.read_data as rd
from data.utils.all_channels_names import ALL_CHANNEL_NAMES

if __name__ == "__main__":


    df = rd.read_all_file_df(channels_names=ALL_CHANNEL_NAMES, idx_exp=[3], idx_people=[1],
                             path="/home/daniel/repos/Decoding_of_EEG/data/raw")

    plot_signal_3d_separate_windows(df, sensor_column='Poz')