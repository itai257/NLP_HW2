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

# CUDA_LAUNCH_BLOCKING=1
data_dir = "data/"
path_train = data_dir + "train.labeled"
print("path_train -", path_train)
path_test = data_dir + "test.labeled"
print("path_test -", path_test)
paths_list = [path_train]


word_dict, pos_dict = get_vocabs(paths_list)
train = PosDataset(word_dict, pos_dict, data_dir, 'train', padding=False)
train_dataloader = DataLoader(train, shuffle=True)
#test = PosDataset(word_dict, pos_dict, data_dir, 'test', padding=False)
#test_dataloader = DataLoader(test, shuffle=False)


EPOCHS = 150
WORD_EMBEDDING_DIM = 100
HIDDEN_DIM = 1000
word_vocab_size = len(train.word_idx_mappings)
tag_vocab_size = len(train.pos_idx_mappings)

model = KiperwasserDependencyParser(HIDDEN_DIM, word_vocab_size, tag_vocab_size,
                                    train_dataloader.dataset.word_vectors)

use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()

#device = "cpu"
device = torch.device("cuda:0" if use_cuda else "cpu")



# Define the loss function as the Negative Log Likelihood loss (NLLLoss)
loss_function = nn.NLLLoss(ignore_index=-1)

# We will be using a simple SGD optimizer to minimize the loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
acumulate_grad_steps = 40  # This is the actual batch_size, while we officially use batch_size=1

# Training start
print("Training Started")
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
        model.zero_grad()
        words_idx_tensor, pos_idx_tensor, sentence_length, true_tree_heads = input_data

        predicted_tree, soft_max_score_matrix = model((words_idx_tensor, pos_idx_tensor, true_tree_heads))

        loss = loss_function(soft_max_score_matrix, true_tree_heads[0].to(device))
        #loss = loss / acumulate_grad_steps
        loss.backward()
        acc = sum(predicted_tree == true_tree_heads[0].numpy()) / len(predicted_tree)
        acc_list.append(acc.item())
        #optimizer.step()
        #model.zero_grad()
        #if i % acumulate_grad_steps == 0:
        optimizer.step()

        if i % 500 == 0:
            text = "-------------------\ntagged_tree: {}, real_tree: {}\nlast 500 acc: {}\nloss:{}"\
                .format(predicted_tree, true_tree_heads, sum(acc_list[-500:]) / len(acc_list[-500:])
                        , printable_loss / i)
            print(text)
            result_file.write(text)
        printable_loss += loss.item()
    printable_loss = printable_loss / len(train)
    loss_list.append(float(printable_loss))
    #test_acc = evaluate()
    e_interval = i
    epoch_print = "---\n---\n---\n---\n---\n---\n---\nEpoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}, \
    time:".format(epoch + 1, np.mean(loss_list[-e_interval:]), sum(acc_list[-e_interval:]) / len(acc_list[-e_interval:]),
                  time.time() - epoch_start_time)
    print("epoch_print")

    result_file.write(epoch_print)
    result_file.close()


