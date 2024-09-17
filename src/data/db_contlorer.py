import psycopg2
from psycopg2 import sql
import datetime


class DbController:
    def __init__(self, dbname, user, password, host, port):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None

    def connect(self):
        self.conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )

    def insert_data(self, table, cwt_data, target):
        cursor = self.conn.cursor()
        query = sql.SQL("INSERT INTO {} (cwt_data, target, time) VALUES (%s, %s, %s)").format(sql.Identifier(table))
        cursor.execute(query, (psycopg2.Binary(cwt_data), target, datetime.datetime.now()))
        self.conn.commit()
        cursor.close()

    def insert_data_own_time(self, table, cwt_data, target, time):
        cursor = self.conn.cursor()
        query = sql.SQL("INSERT INTO {} (cwt_data, target, time) VALUES (%s, %s, %s)").format(sql.Identifier(table))
        cursor.execute(query, (psycopg2.Binary(cwt_data), target, time))
        self.conn.commit()
        cursor.close()

    def fetch_data(self, table):
        cursor = self.conn.cursor()
        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        cursor.close()

    def close(self):
        self.conn.close()