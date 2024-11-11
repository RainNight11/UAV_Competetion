import math

import numpy as np
from torch.autograd import Variable
from model.lib import ST_RenovateNet


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 2 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def mix_top_for_same_class(self, x, y):
        x_partner = x.detach().clone()
        y = y.unsqueeze(1).repeat(1, x.shape[0] // y.shape[0]).view(-1)
        for sample_id in range(x.shape[0]):
            rand_id = np.random.randint(x.shape[0])
            rand_cnt = 1
            while y[rand_id] != y[sample_id] and sample_id != rand_id and rand_cnt < 5:
                rand_id = np.random.randint(x.shape[0])
                rand_cnt += 1
            if y[rand_id] != y[sample_id]:
                rand_id = sample_id
            x_partner[sample_id] = x[rand_id]
        mixed_x = 0.9 * x + 0.1 * x_partner
        return mixed_x

    def forward(self, x, A=None, alpha=1, get_topology=False, label=None, mix_top=False):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        topology = x1.detach().clone()
        if mix_top:
            x1 = self.mix_top_for_same_class(x1, label)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        if get_topology:
            return x1, topology.mean(1, keepdim=True)
        return x1


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x, get_topology=False, label=None, mix_top=False):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        topology = []
        for i in range(self.num_subset):
            if get_topology:
                z, top = self.convs[i](x, A[i], self.alpha, True, label=label, mix_top=mix_top)
                y = z + y if y is not None else z
                topology.append(top)
            else:
                z = self.convs[i](x, A[i], self.alpha, label=label, mix_top=mix_top)
                y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        if get_topology:
            return y, torch.cat(topology, dim=1).mean(1, keepdim=True)
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:channel//2])
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2], use_pe=False):
        super(TCN_GCN_unit, self).__init__()
        self.use_pe = use_pe
        if self.use_pe:
            self.pos_enc = PositionalEncoding(in_channels, A.shape[1], 64, 'spatial')

        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, get_topology=False, label=None, mix_top=False):
        if self.use_pe:
            x = self.pos_enc(x)
        if get_topology:
            tmp, top = self.gcn1(x, True, label=label, mix_top=mix_top)
            y = self.relu(self.tcn1(tmp) + self.residual(x))
            return y, top
        else:
            y = self.relu(self.tcn1(self.gcn1(x, label=label, mix_top=mix_top)) + self.residual(x))
            return y


def get_attn_map_s(x, e_lambda=1e-4):
        NM, C, T, V = x.size()
        num = V * T - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / num + e_lambda)) + 0.5
        att_map = torch.sigmoid(y)
        att_map_s = att_map.mean(dim=[1, 2])
        # N * M, V
        return att_map_s

class Model(nn.Module):
    def build_basic_blocks(self):
        A = self.graph.A  # 3,25,25
        self.l1 = TCN_GCN_unit(self.in_channels, self.base_channel, A, residual=False, adaptive=self.adaptive)
        self.l2 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l3 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l4 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l5 = TCN_GCN_unit(self.base_channel, self.base_channel * 2, A, stride=2, adaptive=self.adaptive)
        self.l6 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 2, A, adaptive=self.adaptive)
        self.l7 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 2, A, adaptive=self.adaptive)
        self.l8 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 4, A, stride=2, adaptive=self.adaptive)
        self.l9 = TCN_GCN_unit(self.base_channel * 4, self.base_channel * 4, A, adaptive=self.adaptive)
        self.l10 = TCN_GCN_unit(self.base_channel * 4, self.base_channel * 4, A, adaptive=self.adaptive)

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = ST_RenovateNet(self.base_channel, self.num_frame, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_mid = ST_RenovateNet(self.base_channel * 2, self.num_frame // 2, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_high = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_fin = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")

    def __init__(self,
                 # Base Params
                 num_class=60, num_point=25, num_frame=64, num_person=2, graph=None, graph_args=dict(), in_channels=2,
                 base_channel=64, drop_out=0, adaptive=True,
                 # Module Params
                 cl_mode='ST-Multi-Level', multi_cl_weights=[1, 1, 1, 1], cl_version='V0', pred_threshold=0, use_p_map=True,
                 ):
        super(Model, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        self.num_person = num_person
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        self.in_channels = in_channels
        self.base_channel = base_channel
        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x
        self.adaptive = adaptive
        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.build_basic_blocks()

        if self.cl_mode is not None:
            self.build_cl_blocks()

        self.fc = nn.Linear(self.base_channel * 4, self.num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def get_hidden_feat(self, x, pooling=True, raw=False):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # First stage
        x = self.l1(x)

        # Second stage
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        # Third stage
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)

        # Forth stage
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)

        if raw:
            return x

        if pooling:
            return x.mean(3).mean(1)
        else:
            return x.mean(1)

    def get_ST_Multi_Level_cl_output(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                  cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return logits, cl_loss

    def forward(self, x, label=None, get_cl_loss=False, get_hidden_feat=False, **kwargs):

        if get_hidden_feat:
            return self.get_hidden_feat(x)

        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        feat_low = x.clone()

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        feat_mid = x.clone()

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()

        x = self.l9(x)
        x = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T*V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        if get_cl_loss and self.cl_mode == "ST-Multi-Level":
            return self.get_ST_Multi_Level_cl_output(x, feat_low, feat_mid, feat_high, feat_fin, label)

        return self.fc(x)