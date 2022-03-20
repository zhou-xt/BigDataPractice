import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import seaborn as sn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.autograd import Variable as var
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.01
cirterion = nn.CrossEntropyLoss()
batch_size = 64
TOTAL_EPOCHS = 20

train_data = pd.read_csv('WoeData.csv')
test_data = pd.read_csv('TestWoeData.csv')
Y_train = train_data['SeriousDlqin2yrs']
Y_train = torch.tensor(Y_train)
Y_train = Y_train.reshape(84084, 1, 1)

X_train = train_data.drop(['SeriousDlqin2yrs'], axis=1)
X_train = np.array(X_train)
X_train = var(torch.tensor(X_train))
X_train = X_train.reshape(84084, 1, 6)

Y_test = test_data['SeriousDlqin2yrs']
Y_test = torch.tensor(Y_test)
Y_test = Y_test.reshape(36036, 1, 1)

X_test = test_data.drop(['SeriousDlqin2yrs'], axis=1)
X_test = np.array(X_test)
X_test = torch.tensor(X_test)
X_test = X_test.reshape(36036, 1, 6)

train_loader = DataLoader(TensorDataset(X_train.float(), Y_train.float()),
                          shuffle=True, batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test.float(), Y_test.float()),
                         shuffle=True, batch_size=batch_size)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 6, 2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.conv3 = nn.Conv1d(16, 32, 1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = CNN()
net.to(device)
optimizer = optim.Adam(net.parameters(), lr)  # 定义优化器，使用Adam优化器

net.train()

losses = []
correct = 0
total = 0

for epoch in range(TOTAL_EPOCHS):
    for i, (x, y) in enumerate(train_loader):
        x = x.float().to(device)
        y = y.long().to(device)
        y = y.squeeze()
        optimizer.zero_grad()
        outputs = net(x)
        loss = cirterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum()
    print('Train   Epoch : %d/%d,   Loss: %.4f,   accuracy:  %d/%d  %.4f %%' % (
        epoch + 1, TOTAL_EPOCHS, np.mean(losses), correct / (epoch + 1), len(train_loader.dataset),
        100 * correct / total))

net.eval()
correct = 0
total = 0
for i, (x, y) in enumerate(test_loader):
    x = x.float().to(device)
    y = y.long()
    y = y.squeeze()
    outputs = net(x).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum()
print('Test   accrary: %.4f %%' % (100 * correct / total))
y = y.cpu().tolist()
predicted = predicted.cpu().tolist()
print("precision score: ", precision_score(y, predicted))
print("recall score: ", recall_score(y, predicted))
print("F1 score: ", f1_score(y, predicted))
sn.heatmap(confusion_matrix(y, predicted), annot=True)
plt.show()
y = np.array(y)
predicted = np.array(predicted)
print('AUC:', roc_auc_score(y, predicted))
