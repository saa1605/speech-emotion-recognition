import numpy as np
np.random.seed(1234)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support

from model import BiModel, Model, MaskedNLLLoss
from dataloader import IEMOCAPDataset

MODEL_PATH = "./models/"


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path,
                        batch_size=32,
                        valid=0.1,
                        num_workers=0,
                        pin_memory=False):
    trainset = IEMOCAPDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def init_cuda(no_cuda):
    cuda = torch.cuda.is_available() and not no_cuda

    if cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    return cuda


def loss_function_init(class_weight, cuda):
    loss_weights = torch.FloatTensor([
        1 / 0.086747,
        1 / 0.144406,
        1 / 0.227883,
        1 / 0.160585,
        1 / 0.127711,
        1 / 0.252668,
    ])

    if class_weight:
        loss_function = MaskedNLLLoss(
            loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()

    return loss_function


def train_model(model,
                tensorboard,
                dataloader,
                epoch,
                cuda,
                optimizer=None,
                class_weight=True,
                train=True):

    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []

    loss_function = loss_function_init(class_weight, cuda)
    assert not train or optimizer != None
    if train:
        model.train()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, _, _, qmask, umask, label =\
                [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        log_prob, _, _, _ = model(textf, qmask,
                                  umask)  # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0, 1).contiguous().view(
            -1,
            log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        loss.backward()
        if tensorboard:
            for param in model.named_parameters():
                writer.add_histogram(param[0], param[1].grad, epoch)
        optimizer.step()
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(
        accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(
        f1_score(labels, preds, sample_weight=masks, average='weighted') * 100,
        2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [
        alphas, alphas_f, alphas_b, vids
    ]


def eval_model(model, dataloader, cuda, class_weight=True):

    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []

    loss_function = loss_function_init(class_weight, cuda)

    model.eval()
    for data in dataloader:
        textf, _, _, qmask, umask, label =\
                [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        log_prob, alpha, alpha_f, alpha_b = model(
            textf, qmask, umask)  # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0, 1).contiguous().view(
            -1,
            log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        alphas += alpha
        alphas_f += alpha_f
        alphas_b += alpha_b
        vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(
        accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(
        f1_score(labels, preds, sample_weight=masks, average='weighted') * 100,
        2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [
        alphas, alphas_f, alphas_b, vids
    ]


def save_model(model,optimizer):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    ts = time.gmtime()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", ts)
    stamp[:10]
    torch.save(model, './models/dialogueRnn_{}.pth'.format(stamp))


def load_model(modelName):
    return torch.load(MODEL_PATH + modelName)
