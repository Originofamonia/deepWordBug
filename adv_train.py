import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
from loaddata import loaddata, loaddatawithtokenize
from dataloader import Chardata, Worddata
import shutil
from models import CharCNN, SmallRNN, SmallCharRNN, WordCNN
from attack import generate_char_adv, generate_word_adv


def save_checkpoint(state, is_best, filename='checkpoint.dat'):
    torch.save(state, filename + '_checkpoint.dat')
    if is_best:
        shutil.copyfile(filename + '_checkpoint.dat', filename + "_bestmodel.dat")


def main():
    parser = argparse.ArgumentParser(description='Data')
    parser.add_argument('--data', type=int, default=0, metavar='N',
                        help='data 0 - 6')
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
    parser.add_argument('--power', type=int, default=30, metavar='N',
                        help='Attack power')
    parser.add_argument('--batchsize', type=int, default=100, metavar='B',
                        help='batch size')
    parser.add_argument('--maxbatches', type=int, default=None, metavar='B',
                        help='maximum batches of adv samples generated')
    parser.add_argument('--scoring', type=str, default='replaceone', metavar='N',
                        help='Scoring function.')
    parser.add_argument('--transformer', type=str, default='homoglyph', metavar='N',
                        help='Transformer function.')
    parser.add_argument('--dictionarysize', type=int, default=20000, metavar='B',
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='B',
                        help='learning rate')
    parser.add_argument('--maxnorm', type=float, default=400, metavar='B',
                        help='learning rate')
    parser.add_argument('--adv_train', type=bool, default=True, help='is adversarial training?')
    args = parser.parse_args()

    torch.manual_seed(9527)
    torch.cuda.manual_seed_all(9527)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print("Loading data...")
    if args.datatype == "char":
        (train, test, numclass) = loaddata(args.data)
        trainchar = Chardata(train, getidx=True)
        testchar = Chardata(test, getidx=True)
        train_loader = DataLoader(trainchar, batch_size=args.batchsize, num_workers=4, shuffle=False)
        test_loader = DataLoader(testchar, batch_size=args.batchsize, num_workers=4, shuffle=False)
        alphabet = trainchar.alphabet
        maxlength = args.charlength
    elif args.datatype == "word":
        (train, test, tokenizer, numclass, rawtrain,
         rawtest) = loaddatawithtokenize(args.data, nb_words=args.dictionarysize, datalen=args.wordlength,
                                         withraw=True)
        word_index = tokenizer.word_index
        trainword = Worddata(train, getidx=True, rawdata=rawtrain)
        testword = Worddata(test, getidx=True, rawdata=rawtest)
        train_loader = DataLoader(trainword, batch_size=args.batchsize, num_workers=4, shuffle=False)
        test_loader = DataLoader(testword, batch_size=args.batchsize, num_workers=4, shuffle=False)
        maxlength = args.wordlength

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    bestacc = 0
    for epoch in range(1, args.epochs + 1):
        print('Start epoch %d' % epoch)
        model.train()
        for dataid, data in enumerate(train_loader):
            inputs, target, idx, raw = data
            inputs, target = Variable(inputs), Variable(target)
            inputs, target = inputs.to(device), target.to(device)
            if args.adv_train:
                if args.datatype == "char":
                    x_adv = generate_char_adv(model, args, numclass, alphabet, data, device)
                elif args.datatype == "word":
                    index2word = {0: '[PADDING]', 1: '[START]', 2: '[UNKNOWN]', 3: ''}
                    if args.dictionarysize == 20000:
                        for i in word_index:
                            if word_index[i] + 3 < args.dictionarysize:
                                index2word[word_index[i] + 3] = i
                    else:
                        for i in word_index:
                            if word_index[i] + 3 < args.dictionarysize:
                                index2word[word_index[i] + 3] = i
                    x_adv = generate_word_adv(model, args, numclass, data, device, index2word, word_index)
                output = model(x_adv)
            else:
                output = model(inputs)

            loss = F.nll_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = .0
        total_loss = 0
        model.eval()
        for dataid, inputs, target in enumerate(test_loader):
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            loss = F.nll_loss(output, target)
            total_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = correct / len(test_loader.dataset)
        avg_loss = total_loss / len(test_loader.dataset)
        print('Epoch %d : Loss %.4f Accuracy %.5f' % (epoch, avg_loss, acc))
        is_best = acc > bestacc
        if is_best:
            bestacc = acc
        if args.dictionarysize != 20000:
            fname = "models/" + args.model + str(args.dictionarysize) + "_" + str(args.data)
        else:
            fname = "models/" + args.model + "_" + str(args.data)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'bestacc': bestacc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=fname)


if __name__ == '__main__':
    main()
