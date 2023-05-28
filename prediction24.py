import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

col24 = []
for i in range(24):
    for attr in ['heartrate', 'resprate', 'map', 'o2sat']:
        col24.append('T' + str(i) + "_" + attr)
    for attr in ['heartrate_err', 'resprate_err', 'map_err', 'o2sat_err']:
        col24.append('T' + str(i) + "_" + attr)


class PatientDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data['id'].unique())

    def __getitem__(self, idx):
        patient_id = self.data['id'].unique()[idx]
        patient_data = self.data[self.data['id'] == patient_id]


        inputs = torch.Tensor(patient_data[col24].values)
        label = torch.Tensor([patient_data.iloc[-1]['label']])

        sample = {
            'patient_id': patient_id,
            'inputs': inputs,
            'label': label
        }
        return sample


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout),
            num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :]) 
        return x


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


train_dataset = PatientDataset('../dataset/train_clean24.csv')
test_dataset = PatientDataset('../dataset/test_clean24.csv')

# input_size = 9
input_size = len(col24)
hidden_size = 64
output_size = 1
num_layers = 2
num_heads = 3
dropout = 0.1
learning_rate = 0.001
num_epochs = 10
batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = GRUModel(input_size, hidden_size, output_size)
# model = TransformerModel(input_size, hidden_size, num_layers, num_heads, output_size, dropout)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_samples = 0
    train_label = np.array([])
    train_pred = np.array([])
    for batch in train_loader:
        inputs = batch['inputs']
        labels = batch['label']
        # print(inputs)
        # print(labels)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs.reshape(1), labels.reshape(1))
        # print(loss)
        train_loss += loss.item()
        train_samples += 1
        if int(labels.reshape(1).item()) == 0:
            train_label = np.append(train_label, 0)
        else:
            train_label = np.append(train_label, 1)
        if outputs.reshape(1).item() < 0.5:
            train_pred = np.append(train_pred, 0)
        else:
            train_pred = np.append(train_pred, 1)

        loss.backward()
        optimizer.step()

    average_loss = train_loss / train_samples
    print(f"Epoch {epoch+1}/{num_epochs}, \tTrain Loss: {average_loss:.4f}, Accuracy: {accuracy_score(train_label, train_pred):.4f}, Precision: {precision_score(train_label, train_pred):.4f}, Recall: {recall_score(train_label, train_pred):.4f}, F1: {f1_score(train_label, train_pred):.4f}")

    model.eval()
    test_loss = 0.0
    test_samples = 0
    test_label = np.array([])
    test_pred = np.array([])
    for batch in test_loader:
        inputs = batch['inputs']
        labels = batch['label']

        outputs = model(inputs)

        loss = criterion(outputs.reshape(1), labels.reshape(1))
        if int(labels.reshape(1).item()) == 0:
            test_label = np.append(test_label, 0)
        else:
            test_label = np.append(test_label, 1)
        if outputs.reshape(1).item() < 0.5:
            test_pred = np.append(test_pred, 0)
        else:
            test_pred = np.append(test_pred, 1)

        test_loss += loss.item()
        test_samples += 1

    average_loss = test_loss / test_samples
    print(f"\t\t\t\tTest Loss: {average_loss:.4f}, Accuracy: {accuracy_score(test_label, test_pred):.4f}, Precision: {precision_score(test_label, test_pred):.4f}, Recall: {recall_score(test_label, test_pred):.4f}, F1: {f1_score(test_label, test_pred):.4f}")
