import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
import time
from data.utils.all_channels_names import ALL_CHANNEL_NAMES


def file_to_DataDrame(path: str, channels_names: list[str] = ALL_CHANNEL_NAMES):
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
        channels_names: list[str], idx_exp: list[int] = [3, 4], idx_people: list[int] = [1, 2],
        path="/home/danielkleczykkleczynski/repos/Decoding_of_EEG/data/raw", normalize_min_max: bool = True ):
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
            if normalize_min_max:
                df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].min()) / (df.iloc[:, :-1].max() - df.iloc[:, :-1].min())
            all_df = pd.concat([all_df, df], axis=0)
    return all_df
