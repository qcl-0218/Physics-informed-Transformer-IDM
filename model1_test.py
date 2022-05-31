import math
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from utils import step_error
criterion = nn.MSELoss()

# 无tensor梯度的IDM模型
def model_IDM(inputs_IDM, his_labels, output_length, s_0, T, a, b, v_d, dt):
    v_pred = torch.zeros((inputs_IDM.shape[0], output_length, 1))
    y_pred = torch.zeros((inputs_IDM.shape[0], output_length, 1))
    acc = torch.zeros((inputs_IDM.shape[0], output_length, 1))
    y = inputs_IDM[:, 0]
    v = inputs_IDM[:, 1]
    s = inputs_IDM[:, 2]
    delta_v = inputs_IDM[:, 3]

    s_x = s_0 + torch.max(torch.tensor(0), v * T + ((v * delta_v) / (2 * (a * b) ** 0.5)))
    # s_x = torch.tensor(2.5)+ torch.max(torch.tensor(0), v*torch.tensor(1.25)+((v*delta_v)/(2*(torch.tensor(1.75)*torch.tensor(1.25))**0.5)))
    a_f = a * (1 - (v / v_d) ** 4 - (s_x / s) ** 2)
    # a_f = torch.tensor(1.75)*(1-(v/torch.tensor(30))**4-(s_x/s)**2)
    v_pred[:, 0, 0] = v + a_f * dt
    for i in range(len(v_pred)):
        if v_pred[i, 0, 0] <= 0:
            v_pred[i, 0, 0] = 0
    y_pred[:, 0, 0] = y + v_pred[i, 0, 0] * dt
    acc[:, 0, 0] = a_f

    for i in range(y_pred.shape[0]):
        for j in range(output_length - 1):
            v = v_pred[i, j, 0]
            delta_v = his_labels[i, j, 1] - v_pred[i, j, 0]
            s = his_labels[i, j, 0] - y_pred[i, j, 0]
            # s_x = self.s_0 + self.T*v - ((v * delta_v)/(2*(self.a*self.b)**0.5))
            # s_x = s_0 +  v*T-((v*delta_v)/(2*(a*b)**0.5))
            s_x = s_0 + torch.max(torch.tensor(0), v * T + ((v * delta_v) / (2 * (a * b) ** 0.5)))
            # acc_temp = self.a*(1-(v/self.v_d)**4-(s_x/s)**2)
            acc_temp = a * (1 - (v / v_d) ** 4 - (s_x / s) ** 2)
            v2 = v + acc_temp * dt
            if v2 <= 0:
                v2 = 0
                acc_temp = (v2 - v) / dt
            y1 = y_pred[i, j, 0]
            y2 = y1 + v2 * dt
            acc[i, j + 1, 0] = acc_temp
            v_pred[i, j + 1, 0] = v2
            y_pred[i, j + 1, 0] = y2

    return y_pred

#PINN部分 (IDM)
class IDMModel(nn.Module):
    def __init__(self, s_0, T, a, b, v_d):
        super(IDMModel, self).__init__()
        self.model_type = 'IDM'
        self.dt = 0.1
        self.s_0 = torch.tensor([1.667], requires_grad=True)
        self.T = torch.tensor([0.504], requires_grad=True)
        self.a = torch.tensor([0.430], requires_grad=True)
        self.b = torch.tensor([3.216], requires_grad=True)
        self.v_d = torch.tensor([16.775], requires_grad=True)

        self.s_0 = torch.nn.Parameter(self.s_0)
        self.T = torch.nn.Parameter(self.T)
        self.a = torch.nn.Parameter(self.a)
        self.b = torch.nn.Parameter(self.b)
        self.v_d = torch.nn.Parameter(self.v_d)

        self.s_0.data.fill_(s_0)
        self.T.data.fill_(T)
        self.a.data.fill_(a)
        self.b.data.fill_(b)
        self.v_d.data.fill_(v_d)

    def forward(self, inputs_IDM, his_labels):
        y = inputs_IDM[:, 0]
        v = inputs_IDM[:, 1]
        s = inputs_IDM[:, 2]
        delta_v = inputs_IDM[:, 3]

        s_x = self.s_0 + v * self.T + ((v * delta_v) / (2 * (self.a * self.b) ** 0.5))
        a_f = self.a * (1 - (v / self.v_d) ** 4 - (s_x / s) ** 2)
        v_pred = v + a_f * self.dt
        for i in range(len(v_pred)):
            if v_pred[i] <= 0:
                v_pred[i] == 0
        output_IDM = y + v_pred * self.dt
        return output_IDM.unsqueeze(1).unsqueeze(2), torch.Tensor(self.s_0.data.cpu().numpy()), torch.Tensor(
            self.T.data.cpu().numpy()), torch.Tensor(self.a.data.cpu().numpy()), torch.Tensor(
            self.b.data.cpu().numpy()), torch.Tensor(self.v_d.data.cpu().numpy())

#PUNN部分 (Transformer)
class TransformerModel(nn.Module):

    def __init__(self, ninput, ntoken, ninp, nhead, nhid, fusion_size, nlayers, dropout, output_length, s_0, T, a, b, v_d, dt):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.embedding_layer = nn.Linear(ninput, ninp)
        self.encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, nlayers)
        # self.relu = nn.ReLU()
        # self.leaky_relu = nn.LeakyReLU(0.1)
        # self.sig = nn.Sigmoid()
        self.decoder = nn.Linear(ninp, ntoken)
        self.dropout_in = nn.Dropout(dropout)
        self.fusion_layer_1 = nn.Linear(fusion_size, ntoken)
        self.fusion_layer_2 = nn.Linear(fusion_size, ntoken)
        self.output_length = output_length
        self.s_0 = s_0
        self.T = T
        self.a = a
        self.b = b
        self.v_d = v_d
        self.dt = dt

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 1
        # self.transformer_encoder.weight.data.uniform_(-initrange, initrange)
        self.embedding_layer.bias.data.zero_()
        self.embedding_layer.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, his_labels):
        #Transformer 模型输出
        src_inputs = inputs[:, :, 0].unsqueeze(2)
        src = src_inputs.transpose(1, 0)
        src = self.embedding_layer(src)
        # pos_src = self.pos_encoder(src)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask

        enc_src = self.transformer_encoder(src, self.src_mask)
        enc_src = enc_src[-1]
        enc_src = enc_src.repeat(self.output_length, 1, 1)
        output = self.decoder(enc_src)
        output = output.transpose(0, 1)
        #历史信息
        dv = (src_inputs[:, -1, 0] - src_inputs[:, -(1+self.output_length), 0]) / self.output_length
        hist = torch.zeros(output.shape)
        for i in range(src_inputs.shape[0]):
            hist[i, :, 0] = torch.linspace(src_inputs[i, -1, 0].item(), src_inputs[i, -1, 0].item() + dv[i].item() * self.output_length,
                                           (self.output_length+1))[1:]
        #物理信息
        output_IDM = model_IDM(inputs[:, -1, :], his_labels[:, :, :], self.output_length, self.s_0, self.T, self.a, self.b, self.v_d, self.dt)
        #结果输出端融合
        fusion = torch.cat([output, hist, output_IDM], axis=2)
        final_output = self.fusion_layer_1(fusion)

        return final_output

class PIT_IDM_1(nn.Module):
    def __init__(self, ninput, ntoken, ninp, nhead, nhid, fusion_size, nlayers, dropout, output_length, s_0, T, a, b, v_d, dt, lr_PUNN, lr_PINN, epoch_num, alpha):
        super(PIT_IDM_1, self).__init__()
        self.dt = dt
        self.output_length = output_length

        self.PUNN = TransformerModel(ninput, ntoken, ninp, nhead, nhid, fusion_size, nlayers, dropout, output_length, s_0, T, a, b, v_d, dt)
        self.PINN = IDMModel(s_0, T, a, b, v_d)

        #self.optimizer = torch.optim.Adam(
            #[{'params': self.PUNN.parameters(), 'lr': 0.0005}, {'params': self.PINN.parameters(), 'lr': 0.0000001}])
        self.optimizer = torch.optim.Adam(
            [{'params': self.PUNN.parameters(), 'lr': lr_PUNN}, {'params': self.PINN.parameters(), 'lr': lr_PINN}])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 15, gamma=0.1)

        self.epoches = epoch_num
        self.alpha = alpha

    def net_PUNN(self, inputs, his_labels):
        output_trans = self.PUNN(inputs, his_labels)
        return output_trans

    def net_PINN(self, inputs_IDM, his_labels):
        output_IDM = self.PINN(inputs_IDM, his_labels)
        return output_IDM

    #def train(self, dataloaders, dataset_sizes, max_num, min_num, save_file, dataset_name):
        ##the training part has been deleted because the confidentiality agreement
    def predict(self, inputs, his_labels, model_location):
        self.PUNN.load_state_dict(torch.load(model_location))
        self.PUNN.eval()
        out = self.net_PUNN(inputs, his_labels)
        return out


