from ftplib import error_temp

import src.data.CWTDataset as CWTDataset
import src.data.egg_read_transform as ert

if __name__ == "__main__":
    df_train = ert.read_all_file_df(num_exp=[3, 4], num_people=[1, 2], path="../data/raw/")
    df_val = ert.read_all_file_df(num_exp=[3, 4], num_people=[3, 4], path="../data/raw/")
    ert.df_to_CWTdb(df_train, db_path="cwt_train_data.db")
    ert.df_to_CWTdb(df_val, db_path="cwt_val_data.db")


