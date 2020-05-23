import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
from tqdm import tqdm
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
    parser.add_argument('--data', type=int, default=3, metavar='N',
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
    parser.add_argument('--power', type=int, default=30, metavar='N',
                        help='Attack power')
    parser.add_argument('--batchsize', type=int, default=20, metavar='B',
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
    parser.add_argument('--adv_train', type=bool, default=True, help='is adversarial training?')
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
        train_loader = DataLoader(trainchar, batch_size=args.batchsize, num_workers=4, shuffle=False)
        test_loader = DataLoader(testchar, batch_size=args.batchsize, num_workers=4, shuffle=False)
        alphabet = trainchar.alphabet
        maxlength = args.charlength
        word_index = None
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
    iterator = tqdm(train_loader, ncols=0, leave=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_test(alphabet, args, device, iterator, model, numclass, optimizer, test_loader, word_index)


def train_test(alphabet, args, device, iterator, model, numclass, optimizer, test_loader, word_index):
    bestacc = 0
    for epoch in range(1, args.epochs + 1):
        print('Epoch: {}/{}'.format(epoch, args.epochs))
        model.train()
        for dataid, data in enumerate(iterator):
            inputs, target, idx, raw = data
            inputs, target = Variable(inputs), Variable(target)
            inputs, target = inputs.to(device), target.to(device)
            if args.adv_train:
                y_adv, x_adv = get_adv(args, data, device, model, numclass, word_index, alphabet)
                loss = F.nll_loss(y_adv, target)
                if args.hidden_loss:
                    # add a loss on penultimate hidden layer
                    h_loss = get_penultimate_hidden_loss(data, x_adv, device, model)
                    loss += h_loss
                    desc = 'h_loss:' + "{:10.4f}".format(h_loss.item()) + ' loss:' + "{:10.4f}".format(loss.item())
                else:
                    desc = 'loss:' + "{:10.4f}".format(loss.item())
            else:
                output = model(inputs)
                loss = F.nll_loss(output, target)
                desc = 'loss:' + "{:10.4f}".format(loss.item())

            iterator.set_description(desc=desc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        clean_correct = .0
        total_loss = 0
        adv_correct = .0
        total_loss_adv = 0
        model.eval()
        for dataid, data in enumerate(test_loader):
            inputs, target, idx, raw = data
            inputs, target = inputs.to(device), target.to(device)
            # inference on clean test set
            output = model(inputs)
            loss = F.nll_loss(output, target)
            total_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            clean_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            # inference on adv test set
            y_adv, x_adv = get_adv(args, data, device, model, numclass, word_index, alphabet)
            loss_adv = F.nll_loss(y_adv, target)
            total_loss_adv += loss_adv.item()
            pred_adv = y_adv.data.max(1, keepdim=True)[1]
            adv_correct += pred_adv.eq(target.data.view_as(pred_adv)).cpu().sum().item()

        clean_acc = clean_correct / len(test_loader.dataset)
        avg_loss = total_loss / len(test_loader.dataset)
        adv_acc = adv_correct / len(test_loader.dataset)
        adv_loss = total_loss_adv / len(test_loader.dataset)
        print('Epoch %d : clean loss %.4f clean accuracy %.5f; adv loss %.4f adv accuracy %.5f'
              % (epoch, avg_loss, clean_acc, adv_loss, adv_acc))
        is_best = clean_acc > bestacc
        if is_best:
            bestacc = clean_acc
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


def get_adv(args, data, device, model, numclass, word_index=None, alphabet=None):
    if args.datatype == "char":
        x_adv = generate_char_adv(model, args, numclass, data, device, alphabet)
    elif args.datatype == "word":
        index2word = {0: '[PADDING]', 1: '[START]', 2: '[UNKNOWN]', 3: ''}
        for i in word_index:
            if word_index[i] + 3 < args.dictionarysize:
                index2word[word_index[i] + 3] = i
        x_adv = generate_word_adv(model, args, numclass, data, device, index2word, word_index)
    y_adv = model(x_adv)
    return y_adv, x_adv


def get_penultimate_hidden_loss(data, x_adv, device, model):
    inputs, target, idx, raw = data
    inputs, target = inputs.to(device), target.to(device)
    h = model.get_penultimate_hidden(inputs)
    h_adv = model.get_penultimate_hidden(x_adv)
    h_loss = nn.MSELoss()(h_adv, h)
    return h_loss


if __name__ == '__main__':
    main()
