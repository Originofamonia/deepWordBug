"""
    Visualize hidden layer
    https://github.com/gurjaspalbedi/deep-learning-pytorch/blob/master/mnist_data_and_analysis_fully_connected.ipynb
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import os
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
from models import CharCNN, SmallRNN, SmallCharRNN, WordCNN
from attack import generate_char_adv, generate_word_adv
from loaddata import loaddata, loaddatawithtokenize
from dataloader import Chardata, Worddata
from adv_train import get_adv


def get_hidden_layer(net, filename, x, device):
    checkpoint = torch.load(filename)
    net = net.to(device)
    net.load_state_dict(checkpoint['net'])

    net.eval()
    h1 = net.get_h1(x)
    h2 = net.get_h2(x)
    h3 = net.get_h3(x)
    h4 = net.get_h4(x)
    h = net(x)
    logits = net.h_to_logits(h)

    return h1, h2, h3, h4, logits


def draw_pca_plot(x, y):
    plt.figure(figsize=(15, 10))
    pca = PCA(2)
    principal_components = pca.fit_transform(x)
    dataframe = pd.DataFrame(data=y, columns=['digit'])
    principalDf = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, dataframe], axis=1)
    digits = [0, 1, 2, 3]  # , 4, 5, 6, 7, 8, 9]
    colors = ['red', 'green', 'blue', 'purple']  # , 'yellow', 'black', 'orange', 'brown', 'grey', 'navy']
    mean = finalDf.groupby('digit').mean()
    for digit, color in zip(digits, colors):
        indices_to_keep = finalDf['digit'] == digit
        plt.scatter(finalDf.loc[indices_to_keep, 'principal component 1'],
                    finalDf.loc[indices_to_keep, 'principal component 2'],
                    c=color)
        # plt.text(mean.loc[digit, 'principal component 1'], mean.loc[digit, 'principal component 2'], digit,
        #          fontsize=14)
    plt.title("PCA h")
    plt.legend(digits)
    plt.grid()
    plt.show()


def draw_tsne_plot(x, y):
    plt.figure(figsize=(15, 10))
    tsne = TSNE(2)
    tsne = tsne.fit_transform(x)
    dataframe = pd.DataFrame(data=y, columns=['digit'])
    principalDf = pd.DataFrame(data=tsne, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, dataframe], axis=1)
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'black', 'orange', 'brown', 'grey', 'navy']
    mean = finalDf.groupby('digit').mean()
    for digit, color in zip(digits, colors):
        indices_to_keep = finalDf['digit'] == digit
        plt.scatter(finalDf.loc[indices_to_keep, 'principal component 1'],
                    finalDf.loc[indices_to_keep, 'principal component 2'],
                    c=color)
        # plt.text(mean.loc[digit, 'principal component 1'], mean.loc[digit, 'principal component 2'], digit,
        # fontsize=14)
    plt.title("tSNE")
    plt.legend(digits)
    plt.grid()
    plt.show()


def main():
    model_path = './outputs/simplernn_0_clean.dat'
    parser = argparse.ArgumentParser(description='Data')
    parser.add_argument('--data', type=int, default=0, metavar='N',
                        help='data 0 - 7')
    parser.add_argument('--charlength', type=int, default=1014, metavar='N',
                        help='length: default 1014')
    parser.add_argument('--wordlength', type=int, default=500, metavar='N',
                        help='length: default 500')
    parser.add_argument('--model', type=str, default='simplernn', metavar='N',
                        help='model type: LSTM as default')
    parser.add_argument('--space', type=bool, default=False, metavar='B',
                        help='Whether including space in the alphabet')
    parser.add_argument('--trans', type=bool, default=False, metavar='B',
                        help='Not implemented yet, add thesaurus transformation')
    parser.add_argument('--backward', type=int, default=-1, metavar='B',
                        help='Backward direction')
    parser.add_argument('--epochs', type=int, default=10, metavar='B',
                        help='Number of epochs')
    parser.add_argument('--power', type=int, default=25, metavar='N',
                        help='Attack power')
    parser.add_argument('--batchsize', type=int, default=50, metavar='B',
                        help='batch size')
    parser.add_argument('--maxbatches', type=int, default=None, metavar='B',
                        help='maximum batches of adv samples generated')
    parser.add_argument('--scoring', type=str, default='replaceone', metavar='N',
                        help='Scoring function.')
    parser.add_argument('--transformer', type=str, default='homoglyph', metavar='N',
                        help='Transformer function.')
    parser.add_argument('--dictionarysize', type=int, default=20000, metavar='B',
                        help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='B',
                        help='learning rate')
    parser.add_argument('--maxnorm', type=float, default=400, metavar='B',
                        help='learning rate')
    parser.add_argument('--adv_train', type=bool, default=False, help='is adversarial training?')
    parser.add_argument('--hidden_loss', type=bool, default=False, help='add loss on hidden')
    args = parser.parse_args()

    torch.manual_seed(9527)
    torch.cuda.manual_seed_all(9527)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "charcnn":
        args.datatype = "char"
    elif args.model == "simplernn":
        args.datatype = "word"
    elif args.model == "bilstm":
        args.datatype = "word"
    elif args.model == "smallcharrnn":
        args.datatype = "char"
        args.charlength = 300
    elif args.model == "wordcnn":
        args.datatype = "word"

    if args.datatype == "char":
        (train, test, numclass) = loaddata(args.data)
        trainchar = Chardata(train, getidx=True)
        testchar = Chardata(test, getidx=True)
        train_loader = DataLoader(trainchar, batch_size=args.batchsize, num_workers=4, shuffle=True)
        test_loader = DataLoader(testchar, batch_size=args.batchsize, num_workers=4, shuffle=False)
        alphabet = trainchar.alphabet
        maxlength = args.charlength
        word2index = None
    elif args.datatype == "word":
        (train, test, tokenizer, numclass, rawtrain,
         rawtest) = loaddatawithtokenize(args.data, nb_words=args.dictionarysize, datalen=args.wordlength,
                                         withraw=True)
        word2index = tokenizer.word_index
        index2word = tokenizer.index_word
        trainword = Worddata(train, getidx=True, rawdata=rawtrain)
        testword = Worddata(test, getidx=True, rawdata=rawtest)
        train_loader = DataLoader(trainword, batch_size=args.batchsize, num_workers=4, shuffle=True)
        test_loader = DataLoader(testword, batch_size=args.batchsize, num_workers=4, shuffle=False)
        maxlength = args.wordlength
        alphabet = None

    if args.model == "charcnn":
        model = CharCNN(classes=numclass)
    elif args.model == "simplernn":
        model = SmallRNN(classes=numclass)
    elif args.model == "bilstm":
        model = SmallRNN(classes=numclass, bidirection=True)
    elif args.model == "smallcharrnn":
        model = SmallCharRNN(classes=numclass)
    elif args.model == "wordcnn":
        model = WordCNN(classes=numclass)

    model = model.to(device)
    print(model)
    print(args)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs, targets, idx, raw = data
            inputs, targets, idx = inputs.to(device), targets.to(device), idx.to(device)
            indices = torch.tensor([0, 1, 26, 32]).to(device)
            # data[0] = torch.index_select(inputs, 0, indices)
            # data[1] = torch.index_select(targets, 0, indices)
            # data[2] = torch.index_select(idx, 0, indices)
            # data[3] = data[3][slice(0, 1, 26, 32)]
            inputs = torch.index_select(inputs, 0, indices)
            targets = torch.index_select(targets, 0, indices)
            print('2: ', raw[0])
            print('3: ', raw[1])
            print('1: ', raw[26])
            print('0: ', raw[32])
            h = model(inputs)
            h_orig = h.view(h.size()[0], -1)
            y_adv, x_adv = get_adv(args, data, device, model, numclass, word2index, alphabet)
            ah = model(x_adv)
            ah = ah.view(ah.size()[0], -1)
            h = torch.cat((h_orig, ah), 0)
            y = torch.cat((targets, targets), 0)
            draw_pca_plot(h.cpu().detach().numpy(), y)
            # draw_tsne_plot(h.cpu().detach().numpy(), y)


if __name__ == '__main__':
    main()
