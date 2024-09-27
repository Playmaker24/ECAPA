
import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SE(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SE, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv2d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Res2Net(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Res2Net, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=kernel_size, dilation=dilation, groups=8, padding=num_pad))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SE(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out

class PreTDNN(nn.Module):
    def __init__(self, inplanes=1, planes=128):
        super(PreTDNN, self).__init__()

        self.conv1  = nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm2d(planes)
        self.layer1 = Res2Net(128, 128, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Res2Net(128, 128, kernel_size=3, dilation=3, scale=8)

        self.layer3 = nn.Conv2d(2*128, 32, kernel_size=1)


    def forward(self, x):
        x = x.unsqueeze(1)
        # print("0", x.shape)
        x = self.conv1(x) # torch.Size([10, 1, 80, 200])
        # print("1",x.shape)
        x = self.relu(x) # torch.Size([10, 64, 80, 200])
        # print("2",x.shape)
        x = self.bn1(x) # torch.Size([10, 64, 80, 200])
        # print("3",x.shape)

        x1 = self.layer1(x) # torch.Size([10, 64, 80, 200])
        # print("4",x1.shape)
        x2 = self.layer2(x+x1) # torch.Size([10, 64, 80, 200])
        # print("5",x2.shape)


        x = self.layer3(torch.cat((x1,x2),dim=1)) # torch.Size([10, 32, 80, 200])
        # print("7",x.shape)
        x = self.relu(x) # torch.Size([10, 32, 80, 200])
        # print("8",x.shape)
        x = torch.flatten(x, start_dim=1, end_dim=2) # torch.Size([10, 2560, 200])
        # print("9", x.shape)
        return x

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class ECAPA_TDNN(nn.Module):

    def __init__(self, C):
        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.specaug = FbankAug()  # Spec augmentation
        self.pre_tdnn = PreTDNN()
        self.conv1 = nn.Conv1d(2560, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug):
        # print('0', x.shape)
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.pre_tdnn(x)
        # print('1', x.shape)
        x = self.conv1(x)
        # print('2', x.shape)
        x = self.relu(x)
        # print('3', x.shape)
        x = self.bn1(x)
        # print('4', x.shape)

        x1 = self.layer1(x)
        # print('5', x1.shape)
        x2 = self.layer2(x + x1)
        # print('6', x2.shape)
        x3 = self.layer3(x + x1 + x2)
        # print('7', x3.shape)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        # print('8', x.shape)
        x = self.relu(x)
        # print('9', x.shape)

        t = x.size()[-1]
        # print('10', t)

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        # print('11', global_x.shape)

        w = self.attention(global_x)
        # print('12', w.shape)

        mu = torch.sum(x * w, dim=2)
        # print('13', mu.shape)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        # print('14', sg.shape)

        x = torch.cat((mu, sg), 1)
        # print('15', x.shape)
        x = self.bn5(x)
        # print('16', x.shape)
        x = self.fc6(x)
        # print('17', x.shape)
        x = self.bn6(x)
        # print('18', x.shape)

        return x