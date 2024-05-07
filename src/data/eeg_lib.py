"""function to read the data from the files and convert it to a dataframe or a numpy array"""

import numpy as np
import pandas as pd
import mne



def file_to_DataDrame(path):
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

def index_of_TX(df):
    """
    df: dataframe
    return: dictionary with the index of the start of each T0, T1, T2
    """
    key = 0
    dict_TX_index = {key: [df['target'][0], 0] }
    current_TX = df['target'][0]
    for i in range(len(df['target'])):
        if df['target'][i] != current_TX:
            key += 1
            dict_TX_index[key] = [df['target'][i], i]
            current_TX = df['target'][i]
 
    return dict_TX_index

def all_exp_to_array(num_person, choose_num_of_exp):
    """
    num_person: int renage(1,109)
    choose_num_of_exp: list of int[3,4,7,8,11,12]

    return: all_meseurments, all_targets
    """
   

    all_meseurments = np.zeros((len(choose_num_of_exp),20000,64))
    all_targets = np.zeros((len(choose_num_of_exp),20000))
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
    num_person: int 
    choose_num_of_exp: list of int [3,4,7,8,11,12]

    return: all_meseurments, all_targets
    """
    all_meseurments = np.zeros((109,len(choose_num_of_exp),20000,64))
    all_targets = np.zeros((109,len(choose_num_of_exp),20000))
    for i in range(1, 109):
        all_meseurments[i], all_targets[i] = all_exp_to_array(i, choose_num_of_exp)
    return all_meseurments, all_targets

