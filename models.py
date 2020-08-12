import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharCNN(nn.Module):
    def __init__(self, classes=4, num_features=69, dropout=0.4):
        super(CharCNN, self).__init__()
        self._conv1 = nn.Conv1d(num_features, 256, kernel_size=7, stride=1)
        self._pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self._conv2 = nn.Conv1d(256, 256, kernel_size=7, stride=1)

        self._conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1)

        self._conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1)

        self._conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1)

        self._conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1)

        self._fc1 = nn.Linear(8704, 1024)
        self._drop1 = nn.Dropout(p=dropout)

        self._fc2 = nn.Linear(1024, 1024)

        self._fc3 = nn.Linear(1024, classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self._pool1(F.relu(self._conv1(x)))
        x = self._pool1(F.relu(self._conv2(x)))
        x = F.relu(self._conv3(x))
        x = F.relu(self._conv4(x))
        x = F.relu(self._conv5(x))
        x = self._pool1(F.relu(self._conv6(x)))
        x = x.view(x.size(0), -1)
        x = self._drop1(F.relu(self._fc1(x)))
        h = x = self._drop1(F.relu(self._fc2(x)))
        x = self._fc3(x)
        x = self.log_softmax(x)

        return h, x

    # def h_to_logits(self, h):
    #     x = self._fc3(h)
    #     x = self.log_softmax(x)
    #     return x

    # def get_penultimate_hidden(self, inputs):
    #     x = self._pool1(F.relu(self._conv1(inputs)))
    #     x = self._pool1(F.relu(self._conv2(x)))
    #     x = F.relu(self._conv3(x))
    #     x = F.relu(self._conv4(x))
    #     x = F.relu(self._conv5(x))
    #     x = self._pool1(F.relu(self._conv6(x)))
    #     x = x.view(x.size(0), -1)
    #     x = self._drop1(F.relu(self._fc1(x)))
    #     x = self._fc2(x)
    #     return x


class SmallRNN(nn.Module):
    def __init__(self, classes=4, bidirection=False, layernum=1, length=20000, embedding_size=100, hiddensize=100):
        super(SmallRNN, self).__init__()
        self.embd = nn.Embedding(length, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hiddensize, layernum, bidirectional=bidirection)
        self.hiddensize = hiddensize
        numdirections = 1 + bidirection
        self.hsize = numdirections * layernum
        self.linear = nn.Linear(hiddensize * numdirections, classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, returnembd=False):
        embd = self.embd(x)
        if returnembd:
            embd = Variable(embd.data, requires_grad=True).to(device)
            embd.retain_grad()

        h0 = Variable(torch.zeros(self.hsize, embd.size(0), self.hiddensize)).to(device)
        c0 = Variable(torch.zeros(self.hsize, embd.size(0), self.hiddensize)).to(device)
        x = embd.transpose(0, 1)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        h = x = x[-1]
        x = self.linear(x)
        x = self.log_softmax(x)

        return h, x

    # def h_to_logits(self, h):
    #     x = self.linear(h)
    #     x = self.log_softmax(x)
    #     return x


class SmallCharRNN(nn.Module):
    def __init__(self, classes=4, bidirection=False, layernum=1, char_size=69, hiddensize=100):
        super(SmallCharRNN, self).__init__()
        self.lstm = nn.LSTM(char_size, hiddensize, layernum, bidirectional=bidirection)
        self.hiddensize = hiddensize
        numdirections = 1 + bidirection
        self.hsize = numdirections * layernum
        self.linear = nn.Linear(hiddensize * numdirections, classes)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        h0 = Variable(torch.zeros(self.hsize, x.size(0), self.hiddensize)).to(device)
        c0 = Variable(torch.zeros(self.hsize, x.size(0), self.hiddensize)).to(device)
        # for inputs in x:
        x = x.transpose(0, 1)
        x = x.transpose(0, 2)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = x[-1]
        x = self.log_softmax(self.linear(x))
        # x = x[-1].transpose(0,1)
        # x = x.view(x.size(0),-1)
        return x


class WordCNN(nn.Module):
    def __init__(self, classes=4, num_features=100, dropout=0.5, maxword=20000):
        super(WordCNN, self).__init__()
        self.embd = nn.Embedding(maxword, num_features)
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(num_features, 256, kernel_size=7, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=3, stride=3)
        # )
        self._conv1 = nn.Conv1d(num_features, 256, kernel_size=7, stride=1)
        self._pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=7, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=3, stride=3)
        # )
        self._conv2 = nn.Conv1d(256, 256, kernel_size=7, stride=1)

        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )
        self._conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1)

        # self.conv4 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )
        self._conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1)

        # self.conv5 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )
        self._conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1)

        # self.conv6 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=3, stride=3)
        # )
        self._conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1)

        # self.fc1 = nn.Sequential(
        #     nn.Linear(3584, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout)
        # )
        self._fc1 = nn.Linear(3584, 1024)
        self._drop1 = nn.Dropout(p=dropout)

        # self.fc2 = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout)
        # )
        self._fc2 = nn.Linear(1024, 1024)

        self._fc3 = nn.Linear(1024, classes)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.embd(x)
        x = x.transpose(1, 2)
        x = self._pool1(F.relu(self._conv1(x)))
        x = self._pool1(F.relu(self._conv2(x)))
        x = F.relu(self._conv3(x))
        x = F.relu(self._conv4(x))
        x = F.relu(self._conv5(x))
        x = self._pool1(F.relu(self._conv6(x)))
        x = x.view(x.size(0), -1)
        x = self._drop1(F.relu(self._fc1(x)))
        x = self._drop1(F.relu(self._fc2(x)))
        x = self._fc3(x)
        x = self.log_softmax(x)

        return x
