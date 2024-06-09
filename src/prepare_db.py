from ftplib import error_temp

import src.data.CWTDataset as CWTDataset
import src.data.egg_read_transform as ert
import psycopg2

if __name__ == "__main__":
    db_params_train = {
        "host": "0.0.0.0",
        "user": "user",
        "password": "1234",
        "database": "dbtrain",
        "port": "54
    db_params_val = {
        "host": "0.0.0.0",
        "user": "user",
        "password": "1234",
        "database": "dbval",
        "port": "5434"
    }


    df_train = ert.read_all_file_df(num_exp=[3, 4], num_people=[1, 2], path="../data/raw/")
    df_val = ert.read_all_file_df(num_exp=[3, 4], num_people=[3, 4], path="../data/raw/")
    conn_train = psycopg2.connect(database="dbtrain", host="0.0.0.0", user="user", password="1234", port="5433")
    conn_val = psycopg2.connect(database="dbval", host="0.0.0.0", user="user", password="1234", port="5434")
    ert.manage_wavelet_transforms_table(conn_train)
    ert.df_to_CWTdb(df_train, conn_train)
    ert.manage_wavelet_transforms_table(conn_val)
    ert.df_to_CWTdb(df_train, conn_val)
    conn_train.close()
    conn_val.close()



