from read_data import read_all_file_df
from db_contlorer import DbController


class DataHandler:
    def __init__(self,db_controller, num_exp=[3, 4], num_people=[1, 2]):
        self.df = read_all_file_df(num_exp, num_people)
        self.db_controller = db_controller

    def make_cwt_transform(self):
        #TODO: implement cwt transform
        pass
    def cwt_data2db(self, db_controller, table):
        for i in range(len(self.df)):
            self.db_controller.insert_data(table, self.df.iloc[i].values, self.df.iloc[i]["target"])

    def get_data(self):
        return self.df.drop(columns=["target"]).values

    def get_target(self):
        return self.df["target"].values


db = DbController(dbname="my_db", user="user", password="1234", host="localhost", port="5433")
cwt_data = DataHandler(db_controller=db)


print(cwt_data.get_data())