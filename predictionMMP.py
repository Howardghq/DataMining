import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 12
hidden_size = 128
output_size = 1
num_layers = 2
num_heads = 4
dropout = 0.1
learning_rate = 0.002
num_epochs = 32
batch_size = 128
FIX_LEN=20
BASE_ATTR = ['heartrate','resprate','map', 'o2sat']
BASE_ATTR_ERR = ['heartrate_err','resprate_err','map_err', 'o2sat_err']
BASE_ATTR_DELTA = ['heartrate_delta','resprate_delta','map_delta', 'o2sat_delta']
colFixLen = []
for i in range(FIX_LEN):
    for attr in BASE_ATTR+BASE_ATTR_ERR+BASE_ATTR_DELTA:
        colFixLen.append('T'+str(i)+"_"+attr)


def get_gru_params():

    def get_params_1(shape):
        return nn.Parameter(torch.tensor(np.random.uniform(0,1,size=shape), dtype=torch.float),requires_grad=True)

    def get_zeros(size):
        return nn.Parameter(torch.zeros(size, dtype=torch.float),requires_grad=True)

    def get_params_3():
        return (
            get_params_1((input_size, hidden_size)),
            get_params_1((hidden_size, hidden_size)),
            get_zeros(hidden_size)
        )

    Wxz, Whz, bz = get_params_3()
    Wxr, Whr, br = get_params_3()
    Wxh, Whh, bh = get_params_3()
    Whq = get_params_1((hidden_size,output_size))
    bq = get_zeros(output_size)
    return nn.ParameterList([Wxz,Whz,bz,Wxr,Whr,br,Wxh,Whh,bh,Whq,bq])


def init_gru_state(batch_size, hidden_size):
    return (torch.zeros(batch_size,hidden_size),)


def gru(batch, state, params):
    Wxz, Whz, bz, Wxr, Whr, br, Wxh, Whh, bh, Whq, bq = params
    H, = state
    outputs = []
    for X in batch:
        print("X.shape:",X.shape,"Wxz.shape:",Wxz.shape,"H.shape:",H.shape,"Whz.shape:",Whz.shape,"bz.shape:",bz.shape)
        Z = torch.sigmoid(torch.matmul(X, Wxz) + torch.matmul(H, Whz) + bz)
        R = torch.sigmoid(torch.matmul(X, Wxr) + torch.matmul(H, Whr) + br)
        H_tilda = torch.tanh(torch.matmul(X, Wxh) + R * torch.matmul(H,Whh)+bh)
        H = Z*H+(1-Z)*H_tilda
        Y=torch.matmul(H,Whq)+bq
        outputs.append(Y)
    return outputs, (H,)


class PatientDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.ids = []
        self.labels = []
        self.inputs = []
        for index, row in self.data.iterrows():
            self.ids.append(torch.Tensor([row['id']]))
            self.labels.append(torch.Tensor([row['label']]))
            tmp = torch.Tensor(row[colFixLen].values)
            self.inputs.append(tmp.reshape(FIX_LEN,input_size))
        print('\tPatientDataset init over. patient num:\t', len(self.ids))


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = {
            'patient_id': self.ids[idx],
            'inputs': self.inputs[idx],
            'label': self.labels[idx]
        }
        return sample


train_dataset = PatientDataset("./dataset/train_clean"+str(FIX_LEN)+".csv")
test_dataset = PatientDataset("./dataset/test_clean"+str(FIX_LEN)+".csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


params = get_gru_params()
fc_dense = nn.Linear(hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    state = init_gru_state(batch_size,hidden_size)
    train_loss = 0.0
    train_samples = 0
    train_label = np.array([])
    train_pred = np.array([])
    for batch in train_loader:
        # print('batch:\t',batch)
        inputs = batch['inputs']
        labels = batch['label'].reshape(-1)
        # print('labels:\t',labels)
        # print('inputs_shape:\t',inputs.shape)

        (hidden_output, state) = gru(inputs,state,params)
        print(hidden_output)
        print(hidden_output[0].shape)
        hidden_output = torch.cat(hidden_output, dim=0)
        print(hidden_output.shape)
        break
        #
        # # print('model(inputs):\t',tmp)
        # # print('model(inputs)_shape:\t',tmp.shape)
        # # print('model(inputs)[-1]:\t',tmp[-1])
        # # print('model(inputs)[-1]_shape:\t',tmp[-1].shape)
        # # print('outputs:\t',outputs)
        # # print('outputs_shape:\t',outputs.shape)
        # # print('labels:\t',labels)
        # # print('labels_shape:\t',labels.shape)
        #
        # loss = criterion(outputs, labels)
        # # print(loss)
        # train_loss += loss.item()
        # train_samples += 1
        #
        # for id in range(len(outputs)):
        #     if int(labels[id].item()) == 0:
        #         train_label = np.append(train_label, 0)
        #     else:
        #         train_label = np.append(train_label, 1)
        #     if outputs[id].item() < 0.5:
        #         train_pred = np.append(train_pred, 0)
        #     else:
        #         train_pred = np.append(train_pred, 1)
        #
        # loss.backward()
        # optimizer.step()

    average_loss = train_loss / train_samples
    print(f"Epoch {epoch+1}/{num_epochs}, \tTrain Loss: {average_loss:.4f}, Accuracy: {accuracy_score(train_label, train_pred):.4f}, Precision: {precision_score(train_label, train_pred):.4f}, Recall: {recall_score(train_label, train_pred):.4f}, F1: {f1_score(train_label, train_pred):.4f}")

