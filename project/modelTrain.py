import h5py
import numpy as np

import torch
import torch.nn as NN
from torch.utils.data import Dataset, DataLoader
# from modelDesign.CNN.modelDesign import AIModel
# from modelDesign.ResNet.modelDesign import AIModel
from modelDesign.RDN.modelDesign import AIModel


class MyDataset(Dataset):
    def __init__(self, mat_file):
        mat = h5py.File(mat_file, 'r')
        self.X = np.transpose(mat['H_in'][:]).astype(np.float32)
        self.Y = np.transpose(mat['H_out'][:]).astype(np.float32)
        self.SNR = np.transpose(mat['SNR'][:]).astype(np.float32)

        del mat
        self.len = len(self.X)

    def __len__(self):
        # return len(self.X)
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx]
        snr = self.SNR[idx]
        y = self.Y[idx]
        return x, snr, y


BATCH_SIZE = 100
LEARNING_RATE = 0.001
TOTAL_EPOCHS = 10
model_save = 'modelSubmit.pth'

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")

model = AIModel().to(DEVICE)
# model = torch.load(model_save).to(DEVICE)
print(model)

if __name__ == '__main__':
    file_name = 'Training_Data.mat'

    print('The current dataset is : %s' % file_name)
    mat = h5py.File(file_name, 'r')

    train_dataset = MyDataset(mat_file=file_name)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    criterion = NN.MSELoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(TOTAL_EPOCHS):
        for i, (x, snr, y) in enumerate(train_loader):
            x = x.float().to(DEVICE)
            y = y.to(DEVICE)
            snr = snr.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x, snr)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (
                epoch + 1, TOTAL_EPOCHS, i + 1, len(train_dataset) // BATCH_SIZE, loss.data.item()))
    torch.save(model.cpu(), model_save)
