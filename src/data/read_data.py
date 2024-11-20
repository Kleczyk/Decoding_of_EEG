import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
import time

global_channels_names = [
    "Fc5.",
    "Fc3.",
    "Fc1.",
    "Fcz.",
    "Fc2.",
    "Fc4.",
    "Fc6.",
    "C5..",
    "C3..",
    "C1..",
    "Cz..",
    "C2..",
    "C4..",
    "C6..",
    "Cp5.",
    "Cp3.",
    "Cp1.",
    "Cpz.",
    "Cp2.",
    "Cp4.",
    "Cp6.",
    "Fp1.",
    "Fpz.",
    "Fp2.",
    "Af7.",
    "Af3.",
    "Afz.",
    "Af4.",
    "Af8.",
    "F7..",
    "F5..",
    "F3..",
    "F1..",
    "Fz..",
    "F2..",
    "F4..",
    "F6..",
    "F8..",
    "Ft7.",
    "Ft8.",
    "T7..",
    "T8..",
    "T9..",
    "T10.",
    "Tp7.",
    "Tp8.",
    "P7..",
    "P5..",
    "P3..",
    "P1..",
    "Pz..",
    "P2..",
    "P4..",
    "P6..",
    "P8..",
    "Po7.",
    "Po3.",
    "Poz.",
    "Po4.",
    "Po8.",
    "O1..",
    "Oz..",
    "O2..",
    "Iz..",
]


def file_to_DataDrame(path, channels_names=global_channels_names):
    """
    This function takes in a file path and returns a dataframe with the data and the target values
    format:
        Fc5	        Fc3	        Fc1	        ...	Oz	        O2	        Iz	        target
    0	-0.000046	-0.000041	-0.000032	...	0.000040	0.000108	0.000055	T0
    1	-0.000054	-0.000048	-0.000034	...	0.000064	0.000114	0.000074	T0
    ...
    """

    reader = mne.io.read_raw_edf(path, preload=True)
    annotations = reader.annotations  # get the values of the annotations
    codes = annotations.description  # get the codes from the annotations
    df = pd.DataFrame(
        reader.get_data().T,
        columns=[channel.replace(".", "") for channel in reader.ch_names],
    )  # transpose the data to get the right shape
    negation_channels_names = list(
        set(df.columns) - set([name.replace(".", "") for name in channels_names])
    )
    df = df.drop(
        columns=negation_channels_names
    )  # remove the columns that are not in the channels_names list

    df = df[~(df == 0).all(axis=1)]  # remove rows with all zeros
    timeArray = np.array(
        [round(x, 10) for x in np.arange(0, len(df) / 160, 0.00625)]
    )  # create an array of time values

    codeArray = []
    counter = 0
    for timeVal in timeArray:
        if (
            timeVal in annotations.onset
        ):  # if the time value is in the onset array, add the corresponding code to the codeArray
            counter += 1
        code_of_target = int(
            codes[counter - 1].replace("T", "")
        )  # convert T0 to 0, T1 to 1, etc
        codeArray.append(code_of_target)

    df["target"] = np.array(codeArray).T
    return df


def read_all_file_df(
    channels_names, idx_exp=[3, 4], idx_people=[1, 2], path="/home/danielkleczykkleczynski/repos/Decoding_of_EEG/data/raw"
):
    """
    This function reads all the files in the path and returns a dataframe with the data and the target values
    format:
        Fc5	        Fc3	        Fc1	        ...	Oz	        O2	        Iz	        target
    0	-0.000046	-0.000041	-0.000032	...	0.000040	0.000108	0.000055	0
    1    -0.000054	-0.000048	-0.000034	...	0.000064	0.000114	0.000074	0
    ...
    Args:
        channels_names (list): The list of channels names
        idx_exp (list): The list of experiments to read
        idx_people (list): The list of people to read
        path (str): The path to the files
    Returns:
        pd.DataFrame: The dataframe containing the data and the target values
    """

    start_time = time.time()
    all_df = pd.DataFrame()
    for subject in tqdm(idx_people, desc="Writing data to DataFrame from files"):
        for file in idx_exp:
            fileName = f"{path}/S{subject:03d}/S{subject:03d}R{file:02d}.edf"
            df = file_to_DataDrame(fileName, channels_names)
            all_df = pd.concat([all_df, df], axis=0)
    end_time = time.time()

    # Obliczanie czasu trwania
    execution_time = end_time - start_time
    print(f"Czas działania pętli: {execution_time:.2f} sekund")
    return all_df
