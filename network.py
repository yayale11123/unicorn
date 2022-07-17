import torch
import torch.nn as nn
import torch.nn.functional as F


# AdaptiveWeightNetwork defines the structure of the Adaptive-Weight network
class AdaptiveWeightNetwork(nn.Module):
    def __init__(self):
        super(AdaptiveWeightNetwork, self).__init__()

        # just　conduct convolution operation in one dim
        self.conv1 = nn.Conv2d(1, 16, (3, 1), stride=(2, 1))
        self.conv2 = nn.Conv2d(16, 32, (3, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(32, 64, (3, 1), stride=(2, 1))
        self.conv4 = nn.Conv2d(64, 32, (3, 1), stride=(2, 1))
        self.conv5 = nn.Conv2d(32, 16, (3, 1), stride=(2, 1))
        self.conv6 = nn.Conv2d(16, 8, (3, 1), stride=(2, 1))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(8)

        self.fc1 = nn.Linear(24, 1)

        # LSTM and FC map the weighted signal to hr
        self.lstm = nn.LSTM(300, 64, num_layers=3, batch_first=True)

        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        raw_signals = torch.squeeze(x, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn1(self.conv5(x)))
        x = F.relu(self.bn4(self.conv6(x)))

        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))

        x = torch.squeeze(x, 2)
        weights = F.softmax(x, dim=1)
        weights = torch.unsqueeze(weights, 1)
        weights = weights.expand(raw_signals.shape)
        # multiply the raw signals with the corresponding weights
        weighted_signals = raw_signals * weights
        weighted_signals = weighted_signals.permute(0, 2, 1)
        weighted_signal = weighted_signals[:, 0, :] + weighted_signals[:, 1, :] + weighted_signals[:, 2, :] + weighted_signals[:, 3, :]
        weighted_signal = torch.unsqueeze(weighted_signal, 1)

        out, (_, _) = self.lstm(weighted_signal)
        out = out[:, -1, :]

        out = F.relu(self.fc2(out))
        hr = self.fc3(out)

        return hr


# HrEstimationNetwork defines the structures of HR estimation network
# For the meaning of input parameters of nn.LSTM, refer to https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
class HrEstimationNetwork(nn.Module):
    def __init__(self):
        super(HrEstimationNetwork, self).__init__()
        self.lstm = nn.LSTM(300, 64, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, (_, _) = self.lstm(x)
        out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
