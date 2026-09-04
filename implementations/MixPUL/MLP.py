# -*- coding: utf-8 -*-

#import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.relu1 = nn.ReLU()
#        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size)
        self.relu2 = nn.ReLU()
#        self.drop2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.bn3 = nn.BatchNorm1d(num_features=int(hidden_size / 2))
        self.relu3 = nn.ReLU()
#        self.drop3 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(num_features=int(hidden_size / 4))
#        self.drop4 = nn.Dropout(p=0.5)

        self.fc5 = nn.Linear(int(hidden_size / 4), output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
#        out = self.drop1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
#        out = self.drop2(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
#        out = self.drop3(out)
#
        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu4(out)
#        out = self.drop4(out)

        out = self.fc5(out)

        out = F.softmax(out, dim=1)
        return out

if __name__ == '__main__':
    pass

