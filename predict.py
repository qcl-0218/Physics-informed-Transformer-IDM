import torch
import math
import copy
import time
import numpy as np
import pandas as pd
from model1_test import PIT_IDM_1
from model2_test import PIT_IDM_2
#from model1 import PIT_IDM_1
#from model2 import PIT_IDM_2
from utils import TrajectoryData, data_preprocess, step_error, all_step_error, _generate_square_subsequent_mask
from torch.utils.data import sampler, Dataset, DataLoader

all_pd_data=pd.read_csv(r'\predict_demo2.csv')
all_pd_data = all_pd_data[['Vehicle_ID', 'Frame_ID', 'y', 'Pre_y', 'v', 'spacing', 'delta_v', 'preceding', 'direction', 'Pre_v']]

pro_data = all_pd_data.to_numpy()
#输入数据处理、缩放过程
output_length = 30
dt = 0.1
input_x = pro_data[:,[2,4,5,6]]
v_pre = pro_data[-1,9]
y_pre = pro_data[-1,3]
if v_pre != 0:
    fut_v_pre = v_pre*np.ones(output_length)
    fut_y_pre = np.linspace(y_pre, y_pre+v_pre*output_length*dt, output_length+1)[1:]
else:
    fut_v_pre = np.zeros(output_length)
    fut_y_pre = np.linspace(y_pre, y_pre+pro_data[-1,4]*output_length*dt, output_length+1)[1:]
input_hist = np.array([fut_y_pre, fut_v_pre]).transpose(1,0)
max_num = np.max(input_x[:,0])+185
min_num = np.min(input_x[:,0])
input_x[:,0] = input_x[:,0]-min_num
input_hist[:,0] = input_hist[:,0]-min_num
input_x = input_x/(max_num-min_num)
input_hist = input_hist/(max_num-min_num)
input_x = torch.tensor(input_x).unsqueeze(0).to(torch.float32)
input_hist = torch.tensor(input_hist).unsqueeze(0).to(torch.float32)

#ninput = 1
#ntoken = 1
#ninp = 14
#nhead = 2
#nhid = 28
#fusion_size = 3
#nlayers = 3
#dropout = 0.1
#output_length = 30
#s_0 = 1.667/(max_num-min_num)
#T = 0.504
#a = 0.430/(max_num-min_num)
#b = 3.216/(max_num-min_num)
#v_d = 16.775/(max_num-min_num)
#dt = 0.1
#lr_PUNN = 0.0005
#lr_PINN = 0.0000001
#epoch_num = 10
#alpha = 0.7
#model_1 = PIT_IDM_1(ninput, ntoken, ninp, nhead, nhid, fusion_size, nlayers, dropout, output_length, s_0, T, a, b, v_d, dt, lr_PUNN, lr_PINN, epoch_num, alpha)
#模型预测过程
#model_location = r'\PIT-IDM(1)_CKQ4_0917.tar'
#outputs = model_1.predict(input_x, input_hist, model_location)
#output = outputs.detach().numpy()
#output = output*(max_num-min_num)+min_num
#print(output, output.shape)

ninput = 1
ntoken = 1
ninp = 50
nhead = 10
nhid = 50
fusion_size = 3
nlayers = 2
dropout = 0.1
output_length = 30
s_0 = 1.667/(max_num-min_num)
T = 0.504
a = 0.430/(max_num-min_num)
b = 3.216/(max_num-min_num)
v_d = 16.775/(max_num-min_num)
dt = 0.1
lr_PUNN = 0.0005
lr_PINN = 0.0000001
epoch_num = 500
alpha = 0.7
model_2 = PIT_IDM_2(ninput, ntoken, ninp, nhead, nhid, fusion_size, nlayers, dropout, output_length, s_0, T, a, b, v_d, dt, lr_PUNN, lr_PINN, epoch_num, alpha)
model_location = r'\PIT-IDM(2)_CKQ4_0917.tar'
start_of_seq = torch.Tensor([0]).unsqueeze(0).unsqueeze(1).repeat(input_x.shape[0], 1, 1)
dec_input = start_of_seq
for i in range(output_length):
    target_mask = _generate_square_subsequent_mask(dec_input.shape[1])
    outputs = model_2.predict(input_x, dec_input, target_mask, input_hist, model_location)
    dec_input = torch.cat((dec_input, outputs[:, -1:, :]), 1)
output = outputs.detach().numpy()
output = output*(max_num-min_num)+min_num
print(output, output.shape)
print(max_num,min_num)
