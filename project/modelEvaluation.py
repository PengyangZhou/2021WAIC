import torch
import numpy as np
import h5py
from modelDesign import AIModel


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real + 1j * x_imag
    x_hat_C = x_hat_real + 1j * x_hat_imag
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")

# Data loading
file_name = 'Test_Data.mat'

print('The current dataset is : %s' % file_name)
mat = h5py.File(file_name, 'r')
num_H = mat['N'][:]
X1 = np.transpose(mat['H_in_0dB'][:])
Y1 = np.transpose(mat['H_out_0dB'][:])
SNR1 = np.transpose(mat['SNR_0dB'][:])

X2 = np.transpose(mat['H_in_5dB'][:])
Y2 = np.transpose(mat['H_out_5dB'][:])
SNR2 = np.transpose(mat['SNR_5dB'][:])

X3 = np.transpose(mat['H_in_10dB'][:])
Y3 = np.transpose(mat['H_out_10dB'][:])
SNR3 = np.transpose(mat['SNR_10dB'][:])

X4 = np.transpose(mat['H_in_15dB'][:])
Y4 = np.transpose(mat['H_out_15dB'][:])
SNR4 = np.transpose(mat['SNR_15dB'][:])

X5 = np.transpose(mat['H_in_20dB'][:])
Y5 = np.transpose(mat['H_out_20dB'][:])
SNR5 = np.transpose(mat['SNR_20dB'][:])

# Model loading
model_address = 'modelSubmit.pth'
model_loaded = torch.load(model_address).to(DEVICE)

X1 = torch.from_numpy(X1.astype(np.float32)).to(DEVICE)
X2 = torch.from_numpy(X2.astype(np.float32)).to(DEVICE)
X3 = torch.from_numpy(X3.astype(np.float32)).to(DEVICE)
X4 = torch.from_numpy(X4.astype(np.float32)).to(DEVICE)
X5 = torch.from_numpy(X5.astype(np.float32)).to(DEVICE)

SNR1 = torch.from_numpy(SNR1.astype(np.float32)).to(DEVICE)
SNR2 = torch.from_numpy(SNR2.astype(np.float32)).to(DEVICE)
SNR3 = torch.from_numpy(SNR3.astype(np.float32)).to(DEVICE)
SNR4 = torch.from_numpy(SNR4.astype(np.float32)).to(DEVICE)
SNR5 = torch.from_numpy(SNR5.astype(np.float32)).to(DEVICE)

y_test1 = model_loaded(X1, SNR1).cpu().data.numpy()
y_test2 = model_loaded(X2, SNR2).cpu().data.numpy()
y_test3 = model_loaded(X3, SNR3).cpu().data.numpy()
y_test4 = model_loaded(X4, SNR4).cpu().data.numpy()
y_test5 = model_loaded(X5, SNR5).cpu().data.numpy()

result1 = - 10 * np.log(NMSE(Y1, y_test1)) / np.log(10)
result2 = - 10 * np.log(NMSE(Y2, y_test2)) / np.log(10)
result3 = - 10 * np.log(NMSE(Y3, y_test3)) / np.log(10)
result4 = - 10 * np.log(NMSE(Y4, y_test4)) / np.log(10)
result5 = - 10 * np.log(NMSE(Y5, y_test5)) / np.log(10)

score = (result1 + result2 + result3 + result4 + result5) / 5

print('The score of 0dB is ' + np.str(result1))
print('The score of 5dB is ' + np.str(result2))
print('The score of 10dB is ' + np.str(result3))
print('The score of 15dB is ' + np.str(result4))
print('The score of 20dB is ' + np.str(result5))

print('The final score is ' + np.str(score))

print('END')
