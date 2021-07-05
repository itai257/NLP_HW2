#import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter
from DataLoad import get_vocabs, PosDataReader
from DataLoad import PosDataset
from Parser import KiperwasserDependencyParser
import time
from chu_liu_edmonds import decode_mst
torch.manual_seed(1)
def loss_function(scores, real):
    nll_loss = nn.NLLLoss(ignore_index=-1)
    log_soft_max = nn.LogSoftmax(dim=1)
    output = nll_loss(log_soft_max(scores), real)
    return output
########################################################
#def evaluate():
#    acc = 0
#    with torch.no_grad():
#        for batch_idx, input_data in enumerate(test_dataloader):
#            words_idx_tensor, pos_idx_tensor, sentence_length = input_data
#            tag_scores = model(words_idx_tensor)
#            tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)
#
#            _, indices = torch.max(tag_scores, 1)
#            acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
#        acc = acc / len(test)
#    return acc


def predict_edges(scores):
    edge_predictions = []
    for sentence_scores in scores:
        score_matrix = sentence_scores.cpu().detach().numpy()
        #score_matrix[:, 0] = float("-inf")
        mst, _ = decode_mst(score_matrix, len(score_matrix), has_labels=False)
        edge_predictions.append(mst)
    return np.array(edge_predictions)

def evaluate(model, data_loader):
    with torch.no_grad():
        acc = num_of_edges = loss = 0
        for i, sentence in enumerate(data_loader):
            words_idx_tensor, pos_idx_tensor, _, true_tree_heads = sentence
            scores = model((words_idx_tensor, pos_idx_tensor, true_tree_heads))
            loss += loss_function(scores, true_tree_heads.to(device)).item()
            score_matrix = scores.squeeze(0).cpu().detach().numpy()
            predictions = mst, _ = decode_mst(score_matrix, len(score_matrix), has_labels=False)
            true_tree_heads = true_tree_heads.to("cpu").numpy()[:, 1:]
            acc += np.sum(true_tree_heads == predictions)
            num_of_edges += len(predictions[0])


    return acc / num_of_edges, loss / len(data_loader)
data_dir = "data/"
path_train = data_dir + "train.labeled"
print("path_train -", path_train)
path_test = data_dir + "test.labeled"
print("path_test -", path_test)
paths_list = [path_train]


word_dict, pos_dict = get_vocabs(paths_list)
train = PosDataset(word_dict, pos_dict, data_dir, 'train', padding=False)
train_dataloader = DataLoader(train, shuffle=True)
test = PosDataset(word_dict, pos_dict, data_dir, 'test', padding=False)
test_dataloader = DataLoader(test, shuffle=False)


EPOCHS = 50
WORD_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 25
LSTM_HIDDEN_DIM = 125
MLP_HIDDEN_DIM = 100
LEARNING_RATE = 0.001
ACCUMULATED_GRAD_STEPS = 50  # This is the actual batch_size, while we officially use batch_size=1

word_vocab_size = len(train.word_idx_mappings)
tag_vocab_size = len(train.pos_idx_mappings)

model = KiperwasserDependencyParser(word_vocab_size, tag_vocab_size,WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM,
                                    LSTM_HIDDEN_DIM, MLP_HIDDEN_DIM)

use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()

device = torch.device("cuda:0" if use_cuda else "cpu")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training start
print("Training Started")
model.train()
accuracy_list = []
loss_list = []
epochs = EPOCHS

for epoch in range(epochs):
    result_file = open('result.txt', 'a+')
    epoch_start_time = time.time()
    acc_list = []  # to keep track of accuracy
    printable_loss = 0  # To keep track of the loss value
    i = 0
    for batch_idx, input_data in enumerate(train_dataloader):
        i += 1
        words_idx_tensor, pos_idx_tensor, sentence_length, true_tree_heads = input_data

        score_matrix = model((words_idx_tensor, pos_idx_tensor, true_tree_heads))

        loss = loss_function(score_matrix, true_tree_heads.to(device))
        printable_loss += loss.item()
        loss.backward()

        #acc = sum(predicted_tree[1:] == true_tree_heads[0].numpy()[1:]) / (len(predicted_tree)-1)
        #acc_list.append(acc.item())
        if i % ACCUMULATED_GRAD_STEPS == 0 and i > 0:
            optimizer.step()
            model.zero_grad()
    printable_loss = printable_loss / len(train)
    loss_list.append(float(printable_loss))
    test_accuracy, test_loss = evaluate(model, test_dataloader)
    train_accuracy, train_loss = evaluate(model, train_dataloader)
    e_interval = i

    time_of_epoch = time.time() - epoch_start_time
    epoch_print = "---\nEpoch {} Completed,\tTrain Loss {}\tTrain Accuracy: {}\t Test Loss {}\tTest Accuracy: {}\t \
    time: {}".format(epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy, time_of_epoch)
    print(epoch_print)

    result_file.write(epoch_print)
    result_file.close()