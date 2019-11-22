from scipy.io import loadmat
import numpy as np
import sklearn
import gc
import sys
from func import *

# Cross-validated 5 folds:
def create_folds(trainx, trainy, k=5):
    folds_x, folds_y = [], []
    n = len(trainx)
    size = n//k
    np.random.seed(4771)
    shuffled_idx = np.random.choice([i for i in range(n)], size = n, replace = False)
    #print(shuffled_idx[:10])
    for i in range(k-1):
        idx = shuffled_idx[i*size:(i+1)*size]
        folds_x.append(trainx[idx])
        folds_y.append(trainy[idx])

    idx = shuffled_idx[(i+1)*size:]
    folds_x.append(trainx[idx])
    folds_y.append(trainy[idx])
    return [folds_x, folds_y]

def create_data(folds, i=0):
    x_test, y_test = folds[0][i], folds[1][i]
    x_train = np.concatenate(folds[0][0:i] + folds[0][i+1:])
    y_train = np.concatenate(folds[1][0:i] + folds[1][i+1:])
    return x_train, y_train, x_test, y_test


argv = sys.argv
assert(len(argv) == 3)
choice = argv[1]
flag = argv[2]
assert(choice == 'cnn' or choice == 'mlp')
assert(flag == 'load' or flag == 'new')

print("Loading Data...")
data = loadmat('MSdata.mat')
X_train = data['trainx']
Y_train = data['trainy']
X_test = data['testx']

folds = create_folds(X_train, Y_train, k=5)
x_train, y_train, x_val, y_val = create_data(folds, i=0)
x_train.shape, x_val.shape


train_dataset = SongDataset(x_train, y_train, type_ = choice)
train_dataloader = DataLoader(train_dataset, 
                              batch_size=128,
                              shuffle=True, 
                              num_workers=4)
val_dataset = SongDataset(x_val, y_val, type_ = choice)
val_dataloader = DataLoader(val_dataset, 
                            batch_size=128,
                            shuffle=False, 
                            num_workers=4)


if flag == 'load':
	print("Loading {} model...".format(choice))
	if choice == 'cnn':
		model_name = 'res_dense.pth'
	else:
		model_name = 'MLP.pth'
	model = torch.load(model_name)
else:
	print("Creating new {} model...".format(choice))
	if choice == 'cnn':
		model = ResDense().cuda()
		model_name = 'res_dense.pth'
	else:
		model = DenseNet().cuda()
		model_name = 'MLP.pth'


print("Start Training...")

criterion = nn.SmoothL1Loss()
criterion_val = nn.L1Loss()
#optimizer = optim.Adam(model.parameters(), lr=0.000001)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

cur_min = 5.2
epochs = 20
print('Epoch: 0/' + str(epochs) + ' Start...')
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    trn_loss = []
    for batch_idx, sample in enumerate(train_dataloader):
        optimizer.zero_grad()
        data = sample['x'].cuda()
        target = sample['y'].cuda()    
        pred = model(data)
        loss = criterion(target.float(), pred.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        trn_loss.append(loss.item())
        if batch_idx and batch_idx % 500 == 0:
            print('  [batch:  %5d] loss: %.5f' % (batch_idx, running_loss/500))
            running_loss = 0.0

            
    model.eval()
    val_loss = []
    with torch.no_grad():
        for val_sample in val_dataloader:
            data = val_sample['x'].cuda()
            target = val_sample['y'].cuda()
            pred = model(data)
            loss = criterion_val(target.float(), pred.float())
            val_loss.append(loss.item())
    Val_loss = round(np.mean(val_loss),3)
    print('Epoch: ' + str(epoch+1) + '/' + str(epochs) + '   Trn Loss: ' + str(round(np.mean(trn_loss), 3)) + '   Val Loss: ' + str(Val_loss))
    # if Val_loss < cur_min:
    #     torch.save(model, "Res_"+str(Val_loss)+'.pth')
    #     cur_min = Val_loss
    gc.collect()


torch.save(model, model_name)