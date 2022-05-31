from torch.utils.data import sampler, Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryData(Dataset):
    """
    A customized data loader for Traffic.
    """

    def __init__(self, matrixs, labels, history_labels):
        """ Intialize the Traffic dataset

        Args:
            - data: numpy datatype
        """
        self.matrixs = torch.DoubleTensor(matrixs.astype(float))
        self.matrixs = self.matrixs.to(torch.float32)
        self.labels = torch.DoubleTensor(labels.astype(float))
        self.labels = self.labels.to(torch.float32)
        self.history_labels = torch.DoubleTensor(history_labels.astype(float))
        self.history_labels = self.history_labels.to(torch.float32)
        self.len = matrixs.shape[0]

    # probably the most important to customize.
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        matrix = self.matrixs[index]
        label = self.labels[index]
        history_label = self.history_labels[index]
        return matrix, label, history_label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

def data_preprocess(pro_data, direc, input_length, output_length, frame, dt):
    all_input = []
    for i in range(int(pro_data.shape[0])-2*input_length):
        if pro_data[i,0] == pro_data[i+input_length-1,0] and pro_data[i,7] == pro_data[i+input_length-1,7] and \
            pro_data[i+input_length-1,1]-pro_data[i,1]==(input_length-1)*frame and pro_data[i+input_length-1,0] == pro_data[i+input_length-1+output_length,0] and \
            pro_data[i+input_length-1, 7] == pro_data[i+input_length-1+output_length, 7] and pro_data[i+input_length-1+output_length, 1] \
            -pro_data[i+input_length-1, 1] == output_length*frame:
            no_use = pro_data[i+input_length:i+2*input_length]
            no_use = np.array(no_use)
            the_output = no_use[:, :]
            all_together = np.hstack((pro_data[i:i+input_length][:,:], the_output))
            all_input.append(all_together)

    x = []
    y = []
    labels = []
    hist_labels = []
    for i in range(len(all_input)):
        temp = all_input[i]
        begin_pos = temp[0, 2]
        temp_y = temp[:output_length,12]-begin_pos
        temp_y_pre = temp[:output_length,13]-begin_pos
        temp_yv_pre = temp[:output_length,-1]
        temp[:,2] = temp[:,2]-begin_pos
        temp[:,3] = temp[:,3]-begin_pos
        #if temp_y[0] <= 0:
            #continue
        if direc == 1:
            temp[:,2] = temp[:,2]*temp[0,8]
            temp[:,3] = temp[:,3]*temp[0,8]
            temp_y = temp_y*temp[0,8]
            temp_y_pre = temp_y_pre * temp[0, 8]
        yv_pre = temp[-1,9]
        y_pre = temp[-1,3]
        if yv_pre != 0:
            test_yv_pre = yv_pre*np.ones(output_length)
            test_y_pre = np.linspace(y_pre, y_pre+yv_pre*output_length*dt, output_length+1)[1:]
        else:
            test_yv_pre = np.zeros(output_length)
            test_y_pre = np.linspace(y_pre, y_pre+temp[-1,4]*output_length*dt, output_length+1)[1:]
        if temp_y[0] <= temp[-1,2] or temp_y[0]<=0 or temp_y[0]>=temp_y_pre[0]:
            continue
        y.append([temp_y])
        x.append(temp[:,[2,4,5,6]])
        labels.append([temp_y_pre,temp_yv_pre])
        hist_labels.append([test_y_pre, test_yv_pre])

    x = np.array(x)
    y = np.array(y)
    labels = np.array(labels)
    hist_labels = np.array(hist_labels)
    #打乱顺序
    #all_num = range(0, len(x), 1)
    #num = random.sample(all_num, len(x))
    #x = x[num]
    #y = y[num]
    #labels= labels[num]
    #hist_labels= hist_labels[num]
    x_train = x[:int(0.8 * len(x)), :, :]
    x_val = x[int(0.8 * len(x)):int(0.9 * len(x)), :, :]
    x_test = x[int(0.9 * len(x)):, :, :]
    y_train = y[:int(0.8 * len(x)), :, :]
    y_val = y[int(0.8 * len(x)):int(0.9 * len(x)), :, :]
    y_test = y[int(0.9 * len(x)):, :, :]
    train_labels = labels[:int(0.8 * len(x)), :, :]
    val_labels = labels[int(0.8 * len(x)):int(0.9 * len(x)), :, :]
    test_labels = labels[int(0.9 * len(x)):, :, :]
    test_pre_labels = hist_labels[int(0.9 * len(x)):, :, :]

    y_train = y_train.transpose(0, 2, 1)
    train_labels = train_labels.transpose(0, 2, 1)
    y_test = y_test.transpose(0, 2, 1)
    test_labels = test_labels.transpose(0, 2, 1)
    test_pre_labels = test_pre_labels.transpose(0, 2, 1)
    y_val = y_val.transpose(0, 2, 1)
    val_labels = val_labels.transpose(0, 2, 1)

    max_num = max([np.max(x_train[:, :, 0]), np.max(y_train[:, :, 0]), np.max(x_test[:, :, 0]), np.max(y_test[:, :, 0]),
                   np.max(x_val[:, :, 0]), np.max(y_val[:, :, 0])])
    min_num = min([np.min(x_train[:, :, 0]), np.min(y_train[:, :, 0]), np.min(x_test[:, :, 0]), np.min(y_test[:, :, 0]),
                   np.min(x_val[:, :, 0]), np.min(y_val[:, :, 0])])

    x_train[:, :, [0]] = x_train[:, :, [0]] - min_num
    y_train[:, :, [0]] = y_train[:, :, [0]] - min_num
    train_labels[:, :, [0]] = train_labels[:, :, [0]] - min_num
    x_test[:, :, [0]] = x_test[:, :, [0]] - min_num
    y_test[:, :, [0]] = y_test[:, :, [0]] - min_num
    test_labels[:, :, [0]] = test_labels[:, :, [0]] - min_num
    test_pre_labels[:, :, [0]] = test_pre_labels[:, :, [0]] - min_num
    x_val[:, :, [0]] = x_val[:, :, [0]] - min_num
    y_val[:, :, [0]] = y_val[:, :, [0]] - min_num
    val_labels[:, :, [0]] = val_labels[:, :, [0]] - min_num

    x_train = x_train / (max_num - min_num)
    y_train = y_train / (max_num - min_num)
    train_labels = train_labels / (max_num - min_num)
    x_test = x_test / (max_num - min_num)
    y_test = y_test / (max_num - min_num)
    test_labels = test_labels / (max_num - min_num)
    test_pre_labels = test_pre_labels / (max_num - min_num)
    x_val = x_val / (max_num - min_num)
    y_val = y_val / (max_num - min_num)
    val_labels = val_labels / (max_num - min_num)

    return x_train, y_train, train_labels, x_test, y_test, test_labels,test_pre_labels, x_val, y_val, val_labels, max_num, min_num

def all_step_error(outputs, labels):
    out = ((outputs - labels) ** 2) ** 0.5
    lossVal = torch.mean(out[:, :, 0], dim=0)
    err_mean = torch.zeros([6])
    err_final = torch.zeros([6])
    for i in range(6):
        err_mean[i] = torch.mean(lossVal[:5 * (i + 1)])
        err_final[i] = lossVal[5 * (i + 1) - 1]
    return err_mean, err_final


def step_error(outputs, labels):
    out = ((outputs - labels) ** 2) ** 0.5
    lossVal = torch.mean(out[:, :, 0], dim=0)
    lossmean = torch.mean(lossVal)
    lossfinal = lossVal[-1]

    return lossmean, lossfinal

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask