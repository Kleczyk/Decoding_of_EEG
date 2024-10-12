import numpy as np
import psycopg2
from psycopg2 import sql
import datetime
import pickle


class DbController:
    def __init__(self, dbname, user, password, host, port):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self.connect()
        self.shape_cwt_data = None
        self.data_type = None

    def connect(self):
        self.conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        )

    def insert_data(self, table: str, cwt_data: np.array, target: list):

        cwt_data = cwt_data.astype(np.float32)
        self.shape_cwt_data = cwt_data.shape
        self.data_type = cwt_data.dtype
        cursor = self.conn.cursor()
        query = sql.SQL(
            "INSERT INTO {} (cwt_data, target, time) VALUES (%s, %s, %s)"
        ).format(sql.Identifier(table))
        for i in range(cwt_data.shape[0]):
            binary_data = psycopg2.Binary(cwt_data[i].tobytes())
            cursor.execute(query, (binary_data, target[i], datetime.datetime.now()))
        self.conn.commit()
        cursor.close()

    def insert_data_own_time(
        self, table: str, cwt_data: np.array, target: np.array, idx_start: int
    ):
        cwt_data = cwt_data.astype(np.float32)
        cwt_data = cwt_data.reshape(cwt_data.shape[0], -1)
        self.shape_cwt_data = cwt_data.shape
        self.data_type = cwt_data.dtype
        cursor = self.conn.cursor()
        query = sql.SQL(
            "INSERT INTO {} (cwt_data, target, time) VALUES (%s, %s, %s)"
        ).format(sql.Identifier(table))
        for i in range(cwt_data.shape[0]):
            # Serialize the data using pickle
            binary_data = pickle.dumps(cwt_data[i])
            cursor.execute(
                query,
                (
                    binary_data,
                    int(target[i]),
                    datetime.datetime.fromtimestamp(idx_start + i),
                ),
            )
        self.conn.commit()
        cursor.close()

    def clear_table(self, table: str):

        cursor = self.conn.cursor()
        query = sql.SQL("DELETE FROM {}").format(sql.Identifier(table))
        cursor.execute(query)
        self.conn.commit()
        cursor.close

    def get_data_between(self, table: str, start: datetime, end: datetime):
        cursor = self.conn.cursor()
        query = sql.SQL(
            "SELECT cwt_data, target FROM {} WHERE time BETWEEN %s AND %s"
        ).format(sql.Identifier(table))
        cursor.execute(
            query,
            (
                datetime.datetime.fromtimestamp(start),
                datetime.datetime.fromtimestamp(end),
            ),
        )
        rows = cursor.fetchall()

        # Deserialize the data using pickle
        cwt_sequence = np.stack([pickle.loads(row[0]) for row in rows])
        target_sequence = np.array([row[1] for row in rows])

        cursor.close()
        return cwt_sequence, target_sequence[-1]

    def get_data(self, table: str):
        cursor = self.conn.cursor()
        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        cursor.close()

    def get_len(self, table: str) -> int:
        cursor = self.conn.cursor()
        query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
        cursor.execute(query)
        length = cursor.fetchone()[0]
        cursor.close()
        return length

    def fetch_data(self, table):
        cursor = self.conn.cursor()
        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        cursor.close()

    def get_table_len(self, table: str):
        # Funkcja zwracająca liczbę rekordów w tabeli
        cursor = self.conn.cursor()
        query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
        cursor.execute(query)
        length = cursor.fetchone()[0]  # Zwraca pierwszą wartość (czyli liczbę rekordów)
        cursor.close()
        return length

    def close(self):
        self.conn.close()
