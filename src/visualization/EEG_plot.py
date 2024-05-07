# %%
import numpy as np
import matplotlib.pyplot as plt
import pyedflib  # Biblioteka do obsługi plików EDF


# %%
import pandas as pd
import numpy as np
import mne

subject = 1
file = 3

fileName = f'files/S001/S{subject:03d}R{file:02d}.edf'

reader = mne.io.read_raw_edf(fileName,preload=True)
annotations = reader.annotations
codes = annotations.description

df = pd.DataFrame(reader.get_data().T, columns=[channel.replace(".","") for channel in reader.ch_names])
df = df[~(df == 0).all(axis=1)]
timeArray = np.array([round(x,5) for x in np.arange(0,len(df)/160,.00625)])

codeArray = []     
counter = 0
for timeVal in timeArray:
    if timeVal in annotations.onset:
        counter += 1
    codeArray.append(codes[counter-1])

df["target"] = np.array(codeArray).T

# %%
print(df['Fc5'][1])
print(df.head(2))

# %%
# Start T0 or T1 or T2 
def index_of_TX(df):
    key = 0
    dict_TX_index = {key: [df['target'][0], 0] }
    current_TX = df['target'][0]
    for i in range(len(df['target'])):
        if df['target'][i] != current_TX:
            key += 1
            dict_TX_index[key] = [df['target'][i], i]
            current_TX = df['target'][i]
 
    return dict_TX_index
   
        
print(index_of_TX(df))

# %%
indexs = index_of_TX(df)
ax= df.plot(y='Fc5')
ax.set_xlabel("Time [s]")
for i in range(0,len(indexs)):
    if indexs[i][0] == 'T0':
        ax.vlines(x=indexs[i][1], ymin=-0.0003, ymax=0.0003, color='r',label='T0')
    elif indexs[i][0] == 'T1':
        ax.vlines(x=indexs[i][1], ymin=-0.0003, ymax=0.0003, color='b',label='T1')
    elif indexs[i][0] == 'T2':
        ax.vlines(x=indexs[i][1], ymin=-0.0003, ymax=0.0003, color='y',label='T2')
ax.hlines(y=0, xmin=0, xmax=672, color='r')
ax.hlines(y=0, xmin=673, xmax=2000, color='g')
plt.show()



# %%
# Najpierw zainstaluj pyEDFlib, jeśli jeszcze tego nie zrobiłeś:
# !pip install pyedflib



# Zastąp 'path_to_edf_file.edf' ścieżką do Twojego pliku EDF
# file_path = '/home/daniel/repos/Decoding_of_EEG/S001R01.edf'

# # Odczytanie pliku EDF
# with pyedflib.EdfReader(file_path) as f:
#     # Pobranie liczby sygnałów
#     n = f.signals_in_file

#     # Pobranie etykiet sygnałów
#     signal_labels = f.getSignalLabels()

#     signals = []
#     # Odczytanie i wyplotowanie każdego sygnału
#     for i in range(n):
#         signal = f.readSignal(i)
#         signals.append(signal)
#         plt.figure(figsize=(12, 4))
#         plt.plot(signal)
#         plt.title(signal_labels[i])
#         plt.show()
#         if i < 1:
#             break
    
#     signals_array = np.array(signals)
    



# # %%
# print(signal_labels)
# signal= f.readSignal


# # %%
# print(signals)

# # %%
# print(np.max(signals_array))
# print(np.min(signals_array))


# # %%
# #draw plot from 0 to 1000
# x=range(0,19680,1)
# plt.figure
# plt.plot(x,signals_array)
# plt.show

# # %%
# #function of sin cos 
# def sin_cos(x):
   
#     y=(np.sin(x)**2*np.cos(x)**2)/(np.sin(x)+1)
#     return y



# # %%
# #vector of x
# x=np.linspace(0,20,100)

# #vector of y
# y=sin_cos(x)


# # %%
# #show the plot
# plt

# plt.figure()

# plt.plot(x,y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("sin(x)^2+cos(x)^2")
# plt.show()


# # %%
# # definine of torch calas
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# # %%
# # class of neural network
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(100, 1000)
#         self.fc2 = nn.Linear(1000, 1000)
#         self.fc3 = nn.Linear(1000, 1000)
#         self.fc4 = nn.Linear(1000, 100)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.sigmoid(self.fc2(x))
#         x = F.sigmoid(self.fc3(x))

#         x = self.fc4(x)
#         return x

# # %%
# # create the model
# model = Model()

# # define the loss function and the optimiser
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)




# # %%
# tensor_x = torch.Tensor(x) # transform to torch tensor
# tensor_y = torch.Tensor(y)

# # %%



# #train the model
# epochs=1000
# losses=[]
# for i in range(epochs):
#     i=i+1
#     y_pred=model.forward(tensor_x)
#     loss=criterion(y_pred,tensor_y)
#     losses.append(loss.detach().numpy())
#     if i%100==0:
#         print(f'epoch {i} loss: {loss.item()}')
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# # %%

# #plot the y_pred
# plt.figure()
# plt.plot(x,y_pred.detach().numpy())
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("sin(x)^2+cos(x)^2")
# plt.show()
# #polt loss
# plt.figure()
# plt.plot(range(epochs),losses)
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.title("loss")
# plt.show()

# # %%
while True:
    pass



