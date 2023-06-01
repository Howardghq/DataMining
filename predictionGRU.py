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
# import d2lzh as d2l

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 12
hidden_size = 128
output_size = 1
num_layers = 2
num_heads = 4
dropout = 0.1
learning_rate = 0.002
num_epochs = 50
batch_size = 128
FIX_LEN=20
choice = "GRUManual"
BASE_ATTR = ['heartrate','resprate','map', 'o2sat']
BASE_ATTR_ERR = ['heartrate_err','resprate_err','map_err', 'o2sat_err']
BASE_ATTR_DELTA = ['heartrate_delta','resprate_delta','map_delta', 'o2sat_delta']
colFixLen = []
for i in range(FIX_LEN):
    for attr in BASE_ATTR+BASE_ATTR_ERR+BASE_ATTR_DELTA:
        colFixLen.append('T'+str(i)+"_"+attr)


def get_gru_params():

    def get_params_1(shape):
        # return nn.Parameter(torch.tensor(np.random.uniform(0,1,size=shape), dtype=torch.float),requires_grad=True)
        return nn.Parameter(torch.tensor(np.random.normal(scale=0.01,size=shape), dtype=torch.float),requires_grad=True)

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
        # print("X.shape:",X.shape,"Wxz.shape:",Wxz.shape,"H.shape:",H.shape,"Whz.shape:",Whz.shape,"bz.shape:",bz.shape)
        Z = torch.sigmoid(torch.matmul(X, Wxz) + torch.matmul(H, Whz) + bz)
        R = torch.sigmoid(torch.matmul(X, Wxr) + torch.matmul(H, Whr) + br)
        H_tilda = torch.tanh(torch.matmul(X, Wxh) + R * torch.matmul(H,Whh)+bh)
        H = Z*H+(1-Z)*H_tilda
        Y=torch.matmul(H,Whq)+bq
        # print("\tY.shape:",Y.shape,"\tH.shape:",H.shape)
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


def grad_clipping(params, theta):
    norm = torch.tensor([0.0])
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


train_dataset = PatientDataset("./dataset/train_clean"+str(FIX_LEN)+".csv")
test_dataset = PatientDataset("./dataset/test_clean"+str(FIX_LEN)+".csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


params = get_gru_params()
# fc_dense = nn.Linear(hidden_size, output_size)
step_weight = [pow(0.9,FIX_LEN-i-1) for i in range(FIX_LEN)]
# criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(step_weight), pos_weight=torch.tensor([8]))
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]))
optimizer = optim.Adam(params, lr=learning_rate)

train_losses = []
train_f1s = []
train_accuracies = []
test_losses = []
test_f1s = []
test_accuracies = []

for epoch in range(num_epochs):
    state = init_gru_state(batch_size,hidden_size)
    train_loss = 0.0
    train_samples = 0
    train_label = np.array([])
    train_pred = np.array([])
    for batch in train_loader:
        # state = init_gru_state(batch_size, hidden_size)
        # print('len(batch):\t',len(batch))
        inputs = batch['inputs'].transpose(0,1)
        labels = batch['label'].reshape(-1)
        if len(labels) != batch_size:
            continue
        # print('labels:\t',labels)
        # print('inputs_shape:\t',inputs.shape)

        if state is not None:
            for s in state:
                s.detach_()
        optimizer.zero_grad()

        (hidden_output, state) = gru(inputs,state,params)
        # print(hidden_output)
        # print('len(hidden_output):',len(hidden_output))
        hidden_output = torch.concat(hidden_output,dim=1)
        # print('hidden_output.shape:',hidden_output.shape)
        pred_output = hidden_output[:,-1]
        # print('pred_output:',pred_output)

        # hidden_output = hidden_output.view(-1,hidden_size)
        # print(hidden_output.shape)
        # hidden_output = fc_dense(hidden_output)
        # print('after fc, shape:',hidden_output.shape)

        # print('labels:\t',labels)
        # print('pred_output.shape:',pred_output.shape)
        # print('labels_shape:\t',labels.shape)
        #
        # print('labels_shape:\t', torch.tensor(labels).unsqueeze(1).repeat(1,FIX_LEN).shape,'output_shape:\t', hidden_output.shape)
        # loss = criterion(hidden_output, labels.unsqueeze(1).repeat(1,FIX_LEN)) # with weight
        loss = criterion(pred_output, labels)
        # print(loss)
        train_loss += loss.item()
        train_samples += 1

        for id in range(len(labels)):
            if int(labels[id].item()) == 0:
                train_label = np.append(train_label, 0)
            else:
                train_label = np.append(train_label, 1)
            if pred_output[id].item() < 0.5:
                train_pred = np.append(train_pred, 0)
            else:
                train_pred = np.append(train_pred, 1)

        torch.autograd.set_detect_anomaly(True)
        loss.backward() #retain_graph=True
        grad_clipping(params,1e-2)
        optimizer.step()


    average_loss = train_loss / train_samples
    print(f"Epoch {epoch+1}/{num_epochs}, \tTrain Loss: {average_loss:.4f}, Accuracy: {accuracy_score(train_label, train_pred):.4f}, Precision: {precision_score(train_label, train_pred):.4f}, Recall: {recall_score(train_label, train_pred):.4f}, F1: {f1_score(train_label, train_pred):.4f}")
    train_losses.append(average_loss)
    train_f1s.append(f1_score(train_label, train_pred))
    train_accuracies.append(accuracy_score(train_label, train_pred))

    with torch.no_grad():
        # model.eval()
        # print(params[0])
        test_loss = 0.0
        test_samples = 0
        test_label = np.array([])
        test_pred = np.array([])
        test_scores = np.array([])
        for batch in test_loader:
            inputs = batch['inputs'].transpose(0,1)
            labels = batch['label'].reshape(-1)
            if len(labels) != batch_size:
                continue
            (hidden_output, _) = gru(inputs,state,params)
            hidden_output = torch.concat(hidden_output,dim=1)
            pred_output = hidden_output[:,-1]

            # loss = criterion(hidden_output, labels.unsqueeze(1).repeat(1,FIX_LEN)) # with weight
            loss = criterion(pred_output, labels)
            test_loss += loss.item()
            test_samples += 1

            for id in range(len(labels)):
                test_scores = np.append(test_scores, float(pred_output[id].item()))
                if int(labels[id].item()) == 0:
                    test_label = np.append(test_label, 0)
                else:
                    test_label = np.append(test_label, 1)
                if pred_output[id].item() < 0.5:
                    test_pred = np.append(test_pred, 0)
                else:
                    test_pred = np.append(test_pred, 1)

    average_loss = test_loss / test_samples
    print(f"\t\t\tTest Loss: {average_loss:.4f}, Accuracy: {accuracy_score(test_label, test_pred):.4f}, Precision: {precision_score(test_label, test_pred):.4f}, Recall: {recall_score(test_label, test_pred):.4f}, F1: {f1_score(test_label, test_pred):.4f}")
    # print(params[0])
    test_losses.append(average_loss)
    test_f1s.append(f1_score(test_label, test_pred))
    test_accuracies.append(accuracy_score(test_label, test_pred))
    
    if epoch == num_epochs-1:
        fpr, tpr, thresholds = roc_curve(test_label, test_scores)

        plt.figure(figsize=(9, 6))
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.savefig("./figure/" + choice+ "_ROC.jpg")

        confusion = confusion_matrix(test_label, test_pred)

        plt.figure(figsize=(9, 6))
        plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        plt.xlabel('Predicted Condition')
        plt.ylabel('Actual Condition')
        plt.savefig("./figure/" + choice+"_Confusion.jpg")

plt.figure(figsize=(9, 6))
plt.plot(list(range(1, num_epochs + 1)), train_accuracies, label="train")
plt.plot(list(range(1, num_epochs + 1)), test_accuracies, label="test")
plt.legend(loc=4)
plt.title(choice)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("./figure/" + choice + "_Accuracy.jpg", bbox_inches='tight')

plt.figure(figsize=(9, 6))
plt.plot(list(range(1, num_epochs + 1)), train_f1s, label="train")
plt.plot(list(range(1, num_epochs + 1)), test_f1s, label="test")
plt.legend(loc=4)
plt.title(choice)
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.savefig("./figure/" + choice + "_F1.jpg", bbox_inches='tight')

plt.figure(figsize=(9, 6))
plt.plot(list(range(1, num_epochs + 1)), train_losses, label="train")
plt.plot(list(range(1, num_epochs + 1)), test_losses, label="test")
plt.legend(loc=4)
plt.title(choice)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("./figure/" + choice + "_Loss.jpg", bbox_inches='tight')
