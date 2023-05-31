import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

FIX_LEN=20
BASE_ATTR = ['heartrate','resprate','map', 'o2sat']
BASE_ATTR_ERR = ['heartrate_err','resprate_err','map_err', 'o2sat_err']
BASE_ATTR_DELTA = ['heartrate_delta','resprate_delta','map_delta', 'o2sat_delta']
colFixLen = []
for i in range(FIX_LEN):
    for attr in BASE_ATTR+BASE_ATTR_ERR+BASE_ATTR_DELTA:
        colFixLen.append('T'+str(i)+"_"+attr)


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
            # print('tmp',tmp)
            # print('tmp2d', tmp.reshape(24,8))
        print('\tPatientDataset init over. patient num:\t', len(self.ids))


    def __len__(self):
        # return len(self.data['id'].unique())
        return len(self.ids)

    def __getitem__(self, idx):

        sample = {
            'patient_id': self.ids[idx],
            'inputs': self.inputs[idx],
            'label': self.labels[idx]
        }

        # patient_id = self.data['id'].unique()[idx]
        # patient_data = self.data[self.data['id'] == patient_id]
        #
        #
        # inputs = torch.Tensor(patient_data[col24].values)
        # label = torch.Tensor([patient_data.iloc[-1]['label']])
        #
        # sample = {
        #     'patient_id': patient_id,
        #     'inputs': inputs,
        #     'label': label
        # }
        return sample


# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout):
#         super(TransformerModel, self).__init__()
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout),
#             num_layers
#         )
#         self.fc = nn.Linear(input_dim, output_dim)
    
#     def forward(self, x):
#         x = self.transformer_encoder(x)
#         x = self.fc(x[:, -1, :]) 
#         return x


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.rnn(x)
        out = self.fc(h)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h)
        return out


# input_size = 9
input_size = 12
hidden_size = 128
output_size = 1
num_layers = 2
num_heads = 4
dropout = 0.1
learning_rate = 0.002
num_epochs = 32
batch_size = 128

train_dataset = PatientDataset("./dataset/train_clean"+str(FIX_LEN)+".csv")
test_dataset = PatientDataset("./dataset/test_clean"+str(FIX_LEN)+".csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

choice = "RNN"

if choice == "RNN":
    model = RNNModel(input_size, hidden_size, output_size)
elif choice == "LSTM":
    model = LSTMModel(input_size, hidden_size, output_size)
elif choice == "GRU":
    model = GRUModel(input_size, hidden_size, output_size)
# model = TransformerModel(input_size, hidden_size, num_layers, num_heads, output_size, dropout)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
train_f1s = []
train_accuracies = []
test_losses = []
test_f1s = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_samples = 0
    train_label = np.array([])
    train_pred = np.array([])
    for batch in train_loader:
        # print('batch:\t',batch)
        inputs = batch['inputs']
        labels = batch['label'].reshape(-1)
        # if len(labels)<2:
        #     print('inputs:\t',inputs)
        # print('labels:\t',labels)
        # print('inputs_shape:\t',inputs.shape)

        optimizer.zero_grad()

        tmp = model(inputs)
        outputs = tmp.reshape(-1) if len(tmp)==len(labels) else tmp[-1].reshape(-1)
        # print('model(inputs):\t',tmp)
        # print('model(inputs)_shape:\t',tmp.shape)
        # print('model(inputs)[-1]:\t',tmp[-1])
        # print('model(inputs)[-1]_shape:\t',tmp[-1].shape)
        # print('outputs:\t',outputs)
        # print('outputs_shape:\t',outputs.shape)
        # print('labels:\t',labels)
        # print('labels_shape:\t',labels.shape)

        loss = criterion(outputs, labels)
        # print(loss)
        train_loss += loss.item()
        train_samples += 1

        for id in range(len(outputs)):
            if int(labels[id].item()) == 0:
                train_label = np.append(train_label, 0)
            else:
                train_label = np.append(train_label, 1)
            if outputs[id].item() < 0.5:
                train_pred = np.append(train_pred, 0)
            else:
                train_pred = np.append(train_pred, 1)

        loss.backward()
        optimizer.step()

    average_loss = train_loss / train_samples
    print(f"Epoch {epoch+1}/{num_epochs}, \tTrain Loss: {average_loss:.4f}, Accuracy: {accuracy_score(train_label, train_pred):.4f}, Precision: {precision_score(train_label, train_pred):.4f}, Recall: {recall_score(train_label, train_pred):.4f}, F1: {f1_score(train_label, train_pred):.4f}")
    train_losses.append(average_loss)
    train_f1s.append(f1_score(train_label, train_pred))
    train_accuracies.append(accuracy_score(train_label, train_pred))

    model.eval()
    test_loss = 0.0
    test_samples = 0
    test_label = np.array([])
    test_pred = np.array([])
    for batch in test_loader:
        inputs = batch['inputs']
        labels = batch['label'].reshape(-1)

        tmp = model(inputs)
        outputs = tmp.reshape(-1) if len(tmp)==len(labels) else tmp[-1].reshape(-1)

        loss = criterion(outputs, labels)
        test_loss += loss.item()
        test_samples += 1

        for id in range(len(outputs)):
            if int(labels[id].item()) == 0:
                test_label = np.append(test_label, 0)
            else:
                test_label = np.append(test_label, 1)
            if outputs[id].item() < 0.5:
                test_pred = np.append(test_pred, 0)
            else:
                test_pred = np.append(test_pred, 1)

    average_loss = test_loss / test_samples
    print(f"\t\t\tTest Loss: {average_loss:.4f}, Accuracy: {accuracy_score(test_label, test_pred):.4f}, Precision: {precision_score(test_label, test_pred):.4f}, Recall: {recall_score(test_label, test_pred):.4f}, F1: {f1_score(test_label, test_pred):.4f}")
    test_losses.append(average_loss)
    test_f1s.append(f1_score(test_label, test_pred))
    test_accuracies.append(accuracy_score(test_label, test_pred))

    if epoch == num_epochs-1:
        fpr, tpr, thresholds = roc_curve(test_label, test_pred)

        plt.figure(figsize=(9, 6))
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.savefig("../figure/" + choice+ "_ROC.jpg")

        confusion = confusion_matrix(test_label, test_pred)

        plt.figure(figsize=(9, 6))
        plt.imshow(confusion, interpolation='nearest', cmap=plt.confusion.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        plt.xlabel('Predicted Condition')
        plt.ylabel('Actual Condition')
        plt.savefig("../figure/" + choice+"_Confusion.jpg")

plt.figure(figsize=(9, 6))
plt.plot(list(range(1, num_epochs + 1)), train_accuracies, label="train")
plt.plot(list(range(1, num_epochs + 1)), test_accuracies, label="test")
plt.legend(loc=4)
plt.title(choice)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("../figure/" + choice + "_Accuracy.jpg", bbox_inches='tight')

plt.figure(figsize=(9, 6))
plt.plot(list(range(1, num_epochs + 1)), train_f1s, label="train")
plt.plot(list(range(1, num_epochs + 1)), test_f1s, label="test")
plt.legend(loc=4)
plt.title(choice)
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.savefig("../figure/" + choice + "_F1.jpg", bbox_inches='tight')

plt.figure(figsize=(9, 6))
plt.plot(list(range(1, num_epochs + 1)), train_losses, label="train")
plt.plot(list(range(1, num_epochs + 1)), test_losses, label="test")
plt.legend(loc=4)
plt.title(choice)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("../figure/" + choice + "_Loss.jpg", bbox_inches='tight')
