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
import Parser
from Parser import KiperwasserDependencyParser
import time
from chu_liu_edmonds import decode_mst
"""

path_train = "data/train.labeled"
#path_test = "data/test.labeled"
#paths_list = [path_train, path_test]


word_dict, pos_dict = get_vocabs([path_train])
#pos_tagger = PosDataReader(path_train, word_dict, pos_dict)

train = PosDataset(word_dict, pos_dict, "data/", 'train', padding=False)


#train_dataloader = DataLoader(train, shuffle=True)
#test = PosDataset(word_dict, pos_dict, data_dir, 'test', padding=False)
#test_dataloader = DataLoader(test, shuffle=False)

"""

########################################################
def evaluate():
    acc = 0
    with torch.no_grad():
        for batch_idx, input_data in enumerate(test_dataloader):
            words_idx_tensor, pos_idx_tensor, sentence_length = input_data
            tag_scores = model(words_idx_tensor)
            tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)

            _, indices = torch.max(tag_scores, 1)
            acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
        acc = acc / len(test)
    return acc

# CUDA_LAUNCH_BLOCKING=1
data_dir = "data/"
path_train = data_dir + "train.labeled"
print("path_train -", path_train)
path_test = data_dir + "test.labeled"
print("path_test -", path_test)

paths_list = [path_train, path_test]
word_dict, pos_dict = get_vocabs(paths_list)
train = PosDataset(word_dict, pos_dict, data_dir, 'train', padding=False)
train_dataloader = DataLoader(train, shuffle=True)
test = PosDataset(word_dict, pos_dict, data_dir, 'test', padding=False)
test_dataloader = DataLoader(test, shuffle=False)


EPOCHS = 15
WORD_EMBEDDING_DIM = 100
HIDDEN_DIM = 1000
word_vocab_size = len(train.word_idx_mappings)
tag_vocab_size = len(train.pos_idx_mappings)

model = KiperwasserDependencyParser(train_dataloader.dataset.word_vectors, HIDDEN_DIM, word_vocab_size, tag_vocab_size)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if use_cuda:
    model.cuda()

# Define the loss function as the Negative Log Likelihood loss (NLLLoss)
loss_function = nn.NLLLoss()

# We will be using a simple SGD optimizer to minimize the loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
acumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

# Training start
print("Training Started")
accuracy_list = []
loss_list = []
epochs = EPOCHS
clip = 1000 # gradient clipping
for epoch in range(epochs):
    epoch_start_time = time.time()
    acc_list = []  # to keep track of accuracy
    printable_loss = 0  # To keep track of the loss value
    i = 0
    for batch_idx, input_data in enumerate(train_dataloader):
        i += 1
        words_idx_tensor, pos_idx_tensor, sentence_length, true_tree_heads = input_data

        soft_max_score_matrix = model((words_idx_tensor, pos_idx_tensor, true_tree_heads))  # changed??

        score_matrix_to_decode = torch.tensor(soft_max_score_matrix).numpy()
        predicted_tree, _ = decode_mst(score_matrix_to_decode, len(true_tree_heads[0]), has_labels=False)

        #true_edges_indices = torch.cat((true_tree_heads, torch.arange(0,len(true_tree_heads[0])).unsqueeze(0)), dim=0)#.permute(1, 0)
        #tagged_tree = tagged_tree.unsqueeze(0) #.permute(0, 2, 1)

        # print("tag_scores shape -", tag_scores.shape)
        # print("pos_idx_tensor shape -", pos_idx_tensor.shape)
        loss = loss_function(soft_max_score_matrix.to(device), true_tree_heads[0].to(device))
        loss = loss / acumulate_grad_steps
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        if i % acumulate_grad_steps == 0:
            optimizer.step()
            model.zero_grad()
            print("-------------------")
            print("tagged_tree: {}, real_tree: {}".format(predicted_tree, true_tree_heads))
        printable_loss += loss.item()
        #_, indices = torch.max(tagged_tree, 1)
        # print("tag_scores shape-", tag_scores.shape)
        # print("indices shape-", indices.shape)
        acc = sum(predicted_tree == true_tree_heads[0].numpy()) / len(predicted_tree)
        acc_list.append(acc.item())
        #acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
    printable_loss = acumulate_grad_steps * (printable_loss / len(train))
    #acc = acc / len(train)
    loss_list.append(float(printable_loss))
    #accuracy_list.append(float(acc))
    #test_acc = evaluate()
    e_interval = i
    #print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}".format(epoch + 1,
    #                                                                              np.mean(loss_list[-e_interval:]),
    #                                                                              np.mean(accuracy_list[-e_interval:]),
    #                                                                              test_acc))
    print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}, time:".format(epoch + 1,
                                                                                  np.mean(loss_list[-e_interval:]),
                                                                                  sum(acc_list) / len(acc_list),
                                                                                  0))
    print(time.time() - epoch_start_time)