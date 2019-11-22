import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class CustomDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.from_numpy(x)
        if y is None:
            self.y = torch.from_numpy(np.zeros((len(self), 1), dtype=float))
        else:
            self.y = torch.from_numpy(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_dim(self):
        return self.x.shape[1]

    def data_augment(self, threshold_year=2000, epsilon=0.1):
        x_aug = []
        y_aug = []
        for i, year in enumerate(self.y):
            if year <= threshold_year:
                x1 = self.x[i, :] * (1 + epsilon)
                x_aug.append(x1)
                y_aug.append(year)
                x2 = self.x[i, :] * (1 - epsilon)
                x_aug.append(x2)
                y_aug.append(year)
        if len(x_aug) > 0:
            x_aug = torch.stack(x_aug)
            y_aug = torch.stack(y_aug)
            self.x = torch.cat((self.x, x_aug))
            self.y = torch.cat((self.y, y_aug))


class Regressor(nn.Module):
    def __init__(self, input_size=90, output_size=1):
        super(Regressor, self).__init__()

        # self.linear = nn.Linear(input_size, output_size)

        self.fc_1 = nn.Linear(input_size, 128)
        self.norm1 = nn.BatchNorm1d(128)
        self.fc_2 = nn.Linear(128, 64)
        self.norm2 = nn.BatchNorm1d(64)
        self.fc_3 = nn.Linear(64, 32)
        self.norm3 = nn.BatchNorm1d(32)
        self.fc_4 = nn.Linear(32, output_size)

    def forward(self, x):
        # y = self.linear(x)
        # return y

        x = self.norm1(self.fc_1(x))
        x = F.relu(x)
        x = self.norm2(self.fc_2(x))
        x = F.relu(x)
        x = self.norm3(self.fc_3(x))
        x = F.relu(x)
        x = self.fc_4(x)
        return x


def load_data(fn):
    data = loadmat(fn)
    X = data['trainx']
    # X = np.abs(X)
    # corr = [15, 17, 21]
    # vif = [0, 3, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    # X = np.delete(X, vif, axis=1)
    X = X.astype(np.float32)
    y = data['trainy']
    y = y.astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_test = data['testx']
    X_test = X_test.astype(np.float32)
    X_test = scaler.transform(X_test)

    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 4}
    test_params = {'batch_size': 64,
                   'shuffle': False,
                   'num_workers': 4}
    train_set = CustomDataset(X_train, y_train)
    # train_set.data_augment()
    train_loader = DataLoader(train_set, **params)
    val_set = CustomDataset(X_val, y_val)
    val_loader = DataLoader(val_set, **params)
    test_set = CustomDataset(X_test)
    test_loader = DataLoader(test_set, **test_params)
    return train_loader, val_loader, test_loader


def train(train_loader, val_loader, model, optimizer, criterion, epochs=10):
    train_loss = []
    val_mae = []
    for epoch in range(epochs):
        train_epoch_loss = []
        running_loss = 0.0
        model.train()
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            # regularization_loss = 0
            # for param in model.parameters():
            #    regularization_loss += torch.sum(torch.abs(param))
            # loss += 0.01 * regularization_loss
            loss.backward()
            optimizer.step()
            mean_loss = loss.item()
            train_epoch_loss.append(mean_loss)
            running_loss += mean_loss
            if i % 1000 == 999:
                print('[epoch: %d, batch:  %5d] train loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 1000.0))
                running_loss = 0.0
        train_loss.append(np.mean(train_epoch_loss))

        val_epoch_mae = []
        model.eval()
        with torch.no_grad():
            for i, (X_batch, y_batch) in enumerate(val_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                mae = np.mean(np.abs(y_pred.cpu().numpy() - y_batch.cpu().numpy()))
                val_epoch_mae.append(mae)
            val_mean_loss = np.mean(val_epoch_mae)
            print('[epoch: %d] val mae: %.5f' %
                  (epoch + 1, val_mean_loss.item()))
            val_mae.append(val_mean_loss)
    return train_loss, val_mae


def test(test_loader, model):
    result = []
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            y_pred = y_pred.cpu().numpy()
            y_pred = np.squeeze(y_pred)
            result = np.concatenate([result, y_pred])
    return result


def save_result(result, fn):
    df = pd.DataFrame(result)
    df.to_csv(fn, header=False, index=False)


def main(in_fn, out_fn):
    train_loader, val_loader, test_loader = load_data(in_fn)

    model = Regressor()
    model = model.to(device)

    lr = 0.001
    momentum = 0.3
    epochs = 25
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    train_loss, val_mae = train(train_loader, val_loader, model, optimizer, criterion, epochs=epochs)

    plt.plot(train_loss, label='train')
    plt.plot(val_mae, label='val')
    plt.show()
    print(train_loss)
    print(val_mae)

    result = test(test_loader, model)
    save_result(result, out_fn)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(0)
    torch.manual_seed(0)

    main('./MSdata.mat', './output.csv')
