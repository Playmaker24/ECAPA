import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class FCNClassifier(nn.Module):

    def __init__(self, n_class):
        super(FCNClassifier, self).__init__()
        self.fc1 = nn.Linear(192, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, n_class)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        #output = F.softmax(x, dim=1)

        if label is not None:

            if not isinstance(label, torch.Tensor):
                label = torch.tensor([label]).cuda()

            loss = self.ce(x, label)
            #acc = accuracy(output.detach(), label.detach(), topk=(1,))[0]
            acc = accuracy(x.detach(), label.detach(), topk=(1,))[0]
            return loss, acc, x

        else:
            return x
