import numpy as np
import pandas as pd
import mne  # library for reading edf files
import pywt  # library for continuous wavelet transform
import sqlite3
import pickle
from tqdm import tqdm


def file_to_DataDrame(path):
    """
    This function takes in a file path and returns a dataframe with the data and the target values

    Args:
        path (str): The path to the file
    Returns:
        pd.DataFrame: The dataframe containing the data and the target values
    Examples:
        >>> df = file_to_DataDrame("data/S001/S001R03.edf")
        >>> print(df)
            Fc5	        Fc3	        Fc1	        ...	Oz	        O2	        Iz	        target
        0	-0.000046	-0.000041	-0.000032	...	0.000040	0.000108	0.000055	0
        1    -0.000054	-0.000048	-0.000034	...	0.000064	0.000114	0.000074	0
        ...
    """

    reader = mne.io.read_raw_edf(path, preload=True)
    annotations = reader.annotations  # get the values of the annotations
    codes = annotations.description  # get the codes from the annotations

    df = pd.DataFrame(
        reader.get_data().T,
        columns=[channel.replace(".", "") for channel in reader.ch_names],
    )  # transpose the data to get the right shape
    df = df[~(df == 0).all(axis=1)]  # remove rows with all zeros
    timeArray = np.array(
        [round(x, 10) for x in np.arange(0, len(df) / 160, 0.00625)]
    )  # create an array of time values

    codeArray = []
    counter = 0
    for timeVal in timeArray:
        if (
                timeVal in annotations.onset
        ):
            counter += 1
        code_of_target = int(
            codes[counter - 1].replace("T", "")
        )
        codeArray.append(code_of_target)

    df["target"] = np.array(codeArray).T
    return df


def read_all_file_df(num_exp=[3, 4], num_people=[1, 2], path="../../data/raw/"):
    """
    This function reads all the files in the path and returns a dataframe with the data and the target values
    format:
        Fc5	        Fc3	        Fc1	        ...	Oz	        O2	        Iz	        target
    0	-0.000046	-0.000041	-0.000032	...	0.000040	0.000108	0.000055	0
    1    -0.000054	-0.000048	-0.000034	...	0.000064	0.000114	0.000074	0
    ...
    Args:
        num_exp (list): The list of experiments to read
        num_people (list): The list of people to read
        path (str): The path to the files
    Returns:
        pd.DataFrame: The dataframe containing the data and the target values
    """
    all_df = pd.DataFrame()
    for subject in num_people:
        for file in num_exp:
            fileName = f"{path}/S{subject:03d}/S{subject:03d}R{file:02d}.edf"
            df = file_to_DataDrame(fileName)
            all_df = pd.concat([all_df, df], axis=0)
    return all_df


def create_database(db_path):
    """
    This function creates a database with a table to store the continuous wavelet transform of the signals

    Args:
        db_path (str): The path to the database
    Returns:
        None
    examples:
        >>> create_database("cwt_data.db")
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS wavelet_transforms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cwt_data BLOB,
            target INTEGER
        )
    """
    )
    conn.commit()
    conn.close()


def insert_cwt_data(db_path, cwt_data, targets):
    """
    This function takes in the continuous wavelet transform of the signals and the target values and saves them to a database
    Args:
        db_path (str): The path to the database
        cwt_data (np.array): The continuous wavelet transform of the signals
        targets (np.array): The target values
    Returns:
        None
    Examples:
        >>> insert_cwt_data("cwt_data.db", cwt_data, targets)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cwt_data = cwt_data.transpose(2, 0, 1)

    cwt_data = cwt_data.reshape(cwt_data.shape[0], -1)  # <-------- this is option
    i = 0
    for single_cwt in cwt_data:
        cwt_blob = pickle.dumps(np.array(single_cwt, dtype=np.float32))
        cursor.execute(
            "INSERT INTO wavelet_transforms (cwt_data, target) VALUES (?, ?)",
            (cwt_blob, targets[i]),
        )
        i += 1
    conn.commit()
    conn.close()


def df_to_CWTfiles(
        df, num_of_rows=1000, wave="cgau4", frq=160, resolution=100, db_path="cwt_data.db"
):
    """
    This function takes in a dataframe and saves the continuous wavelet transform of the signals to a database.

    Args:
        df (pd.DataFrame): The dataframe containing the signals
        num_of_rows (int): The number of rows to process
        wave (str): The type of wave to use
        frq (int): The frequency of the signals
        resolution (int): The resolution of the wavelet transform
        db_path (str): The path to the database
    Returns:
        None
    """
    create_database(db_path)  # Ensure this function is defined elsewhere in your code.

    # Calculate the number of chunks to process
    num_chunks = len(df) // num_of_rows + (1 if len(df) % num_of_rows != 0 else 0)

    # Create a tqdm progress bar for the loop
    for i in tqdm(range(0, len(df), num_of_rows), total=num_chunks, desc="Processing"):
        end_index = i + num_of_rows
        if end_index > len(df):
            end_index = len(df)
        signals = df.iloc[i:end_index].values
        list_cwt = []

        if signals.shape == (num_of_rows, 65):
            signals = signals.transpose(1, 0)

        for signal in signals[:-1]:  # Exclude the last item assuming it's the target
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            time = np.linspace(0, len(signal) / frq, len(signal))
            widths = np.geomspace(1, 200, num=resolution)
            sampling_period = np.diff(time).mean()
            cwtmatr, freqs = pywt.cwt(
                signal, widths, wave, sampling_period=sampling_period
            )
            cwtmatr = np.abs(cwtmatr)
            list_cwt.append(cwtmatr)

        targets = signals[-1]  # Assuming the last row are the targets
        array_cwt = np.stack(list_cwt, axis=0)
        insert_cwt_data(db_path, array_cwt, targets)  # Ensure this function is defined elsewhere in your code.
        del array_cwt


def index_of_TX(df):
    """
    This function takes in a dataframe and returns a dictionary with the index when the target changes
    Args:
        df (pd.DataFrame): The dataframe containing the signals
    Returns:
        dict: A dictionary containing the index when the target changes
    Examples:
        >>> df = file_to_DataDrame("data/S001/S001R03.edf")
        >>> print(index_of_TX(df))
        {0: [0, 0], 1: [1, 2000], 2: [2, 4000], 3: [3, 6000], 4: [4, 8000], 5: [5, 10000], 6: [6, 12000], 7: [7, 14000], 8: [8, 16000], 9: [9, 18000], 10: [10, 20000]}
    """
    key = 0
    dict_TX_index = {key: [df['target'][0], 0]}
    current_TX = df['target'][0]
    for i in range(len(df['target'])):
        if df['target'][i] != current_TX:
            key += 1
            dict_TX_index[key] = [df['target'][i], i]
            current_TX = df['target'][i]

    return dict_TX_index


def all_exp_to_array(num_person, choose_num_of_exp):
    """
    this function takes in a person number and a list of experiments and returns the meseurments and the targets of the person
    Args:
        num_person (int): The person number
        choose_num_of_exp (list): The list of experiments to read
    Returns:
        np.array: The meseurments of the person
        np.array: The targets of the person
    Examples:
        >>> all_exp_to_array(1, [3, 4, 7, 8, 11, 12])
        (array([[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [-1.00000000e-06, -1.00000000e-06, -1.00000000e-06, ...,
         -1.00000000e-06, -1.00000000e-06, -1.00000000e-06],
        [-2.00000000e-06, -2.00000000e-06, -2.00000000e-06, ...,
         -2.00000000e-06, -2.00000000e-06, -2.00000000e-06],
        ...,
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.
    """

    all_meseurments = np.zeros((len(choose_num_of_exp), 20000, 64))
    all_targets = np.zeros((len(choose_num_of_exp), 20000))
    for i in range(len(choose_num_of_exp)):
        fileName = f"files/S{num_person:03d}/S{num_person:03d}R{choose_num_of_exp[i]:02d}.edf"
        df = file_to_DataDrame(fileName)
        for j in range(20000):
            if j >= len(df):
                all_meseurments[i][j] = np.zeros(64)
                all_targets[i][j] = np.nan
            else:
                all_meseurments[i][j] = df.iloc[j].values[:-1]
                all_targets[i][j] = df.iloc[j].values[-1]
    return all_meseurments, all_targets


def all_files_to_array(choose_num_of_exp):
    """
    this function takes in a list of experiments and returns the meseurments and the targets of all the people
    Args:
        choose_num_of_exp (list): The list of experiments to read
    Returns:
        np.array: The meseurments of all the people
        np.array: The targets of all the people
    Examples:
        >>> all_files_to_array([3, 4, 7, 8, 11, 12])
        (array([[[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [-1.00000000e-06, -1.00000000e-06, -1.00000000e-06, ...,
         -1.00000000e-06, -1.00000000e-06, -1.00000000e-06],
        [-2.00000000e-06, -2.00000000e-06, -2.00000000e-06, ...,
         -2.00000000e-06, -2.00000000e-06, -2.00000000e-06],
        ...,
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
          0.00000000e+00,  0.00000000e+00,  0.000000
    """
    all_meseurments = np.zeros((109, len(choose_num_of_exp), 20000, 64))
    all_targets = np.zeros((109, len(choose_num_of_exp), 20000))
    for i in range(1, 109):
        all_meseurments[i], all_targets[i] = all_exp_to_array(i, choose_num_of_exp)
    return all_meseurments, all_targets
