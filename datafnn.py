import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.01
cirterion = nn.CrossEntropyLoss()
batch_size = 1024
TOTAL_EPOCHS = 100

train_data = pd.read_csv('WoeData.csv')
test_data = pd.read_csv('TestWoeData.csv')
Y_train = train_data['SeriousDlqin2yrs']
X_train = train_data.drop(['SeriousDlqin2yrs'], axis=1)
Y_test = test_data['SeriousDlqin2yrs']
X_test = test_data.drop(['SeriousDlqin2yrs'], axis=1)


class tabularDataset():
    def __init__(self, X, Y):
        self.x = X.values
        self.y = Y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return (self.x[[idx]], self.y[idx])


train_ds = tabularDataset(X_train, Y_train)
test_ds = tabularDataset(X_test, Y_test)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(6, 500)
        self.lin2 = nn.Linear(500, 100)
        self.lin3 = nn.Linear(100, 2)

    def forward(self, x_in):
        x = F.relu(self.lin1(x_in))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = torch.sigmoid(x)
        return x


net = FNN()
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
        optimizer.zero_grad()
        outputs = net(x)
        outputs = outputs.squeeze()
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
    outputs = net(x).cpu()
    outputs = outputs.squeeze()
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
