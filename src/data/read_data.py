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