nfrom src.data.cwt_dataset import CwtDataset
from src.data.db_contlorer import DbController
from src.data.data_handler import DataHandler

# channels_names = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
channels_names = ["Fc5.", "Af8.", "P6..", "P8..", "Iz.."]

db = DbController(
    dbname="my_db", user="user", password="1234", host="localhost", port="5433"
)
db.clear_table("training_data")
db.clear_table("validation_data")

cwt_data = DataHandler(db_controller=db, channels_names=channels_names)
cwt_data.write2db(table="training_data")

cwt_data = DataHandler(
    db_controller=db, channels_names=channels_names, idx_exp=[3, 4], idx_people=[3, 4]
)
cwt_data.write2db(table="validation_data")


training_set = CwtDataset(table="training_data", db_controller=db, sequence_length=1000)
val_set = CwtDataset(table="validation_data", db_controller=db, sequence_length=1000)
print(len(training_set))
print(training_set)
print(training_set[0][0].shape)
print(training_set[0])


print(len(val_set))

# Call the hello function
hello()
print(val_set)
print(val_set[0][0].shape)
print(val_set[0])
