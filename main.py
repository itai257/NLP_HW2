import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from DataLoad import get_vocabs, PosDataReader
from DataLoad import PosDataset
from Parser import KDependencyParserBasic
import time
from chu_liu_edmonds import decode_mst
#import matplotlib.pyplot as plt

def loss_function(scores, real):
    nll_loss = nn.NLLLoss(ignore_index=-1)
    log_soft_max = nn.LogSoftmax(dim=1)
    cur_loss = nll_loss(log_soft_max(scores), real)
    return cur_loss

def evaluate(model, data_loader):
    with torch.no_grad():
        accuracy = 0
        words_count = 0
        accumulated_loss = 0
        for i, sentence in enumerate(data_loader):
            words_idx_tensor, pos_idx_tensor, _, true_tree_heads = sentence

            score_matrix = model((words_idx_tensor, pos_idx_tensor, true_tree_heads))
            accumulated_loss += loss_function(score_matrix, true_tree_heads.to(device)).item()

            score_matrix_numpy = score_matrix.squeeze(0).cpu().detach().numpy()
            predicted_tree, _ = decode_mst(score_matrix_numpy, len(score_matrix_numpy), has_labels=False)
            true_tree_heads = true_tree_heads.squeeze(0).cpu().detach().numpy()
            accuracy += np.sum(true_tree_heads[1:] == predicted_tree[1:]) # compare without first word
            words_count += len(predicted_tree)

        accumulated_loss = accumulated_loss / len(data_loader)
        accuracy = accuracy / words_count

    return accuracy, accumulated_loss

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


EPOCHS = 15
WORD_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 25
LSTM_HIDDEN_DIM = 125
MLP_HIDDEN_DIM = 100
LEARNING_RATE = 0.001
ACCUMULATED_GRAD_STEPS = 50  # This is the actual batch_size, while we officially use batch_size=1

word_vocab_size = len(train.word_idx_mappings)
tag_vocab_size = len(train.pos_idx_mappings)

model = KDependencyParserBasic(word_vocab_size, tag_vocab_size, WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM,
                               LSTM_HIDDEN_DIM, MLP_HIDDEN_DIM)

use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()

device = torch.device("cuda:0" if use_cuda else "cpu")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training start
print("Training Started")
model.train()
train_accuracy_list = []
test_accuracy_list = []

train_loss_list = []
test_loss_list = []

loss_list = []
epochs = EPOCHS

for epoch in range(epochs):
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

        if i % ACCUMULATED_GRAD_STEPS == 0 and i > 0:
            optimizer.step()
            model.zero_grad()
    printable_loss = printable_loss / len(train)
    loss_list.append(float(printable_loss))
    print("evaluating epoch:")
    test_accuracy, test_loss = evaluate(model, test_dataloader)
    train_accuracy, train_loss = evaluate(model, train_dataloader)
    test_loss_list.append(test_loss)
    test_accuracy_list.append(test_accuracy)

    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)

    e_interval = i

    time_of_epoch = time.time() - epoch_start_time
    epoch_print = "---\nEpoch {} Completed,\tTrain Loss {}\tTrain Accuracy: {}\t Test Loss {}\tTest Accuracy: {}\t \
    time: {}".format(epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy, time_of_epoch)
    print(epoch_print)


# show graphs:


#plt.plot(test_accuracy_list, c="red", label ="Accuracy")
#plt.xlabel("Epochs")
#plt.ylabel("Value")
#plt.legend()
#plt.show()
#
#plt.plot(test_loss_list, c="blue", label ="Loss")
#plt.xlabel("Epochs")
#plt.ylabel("Value")
#plt.legend()
#plt.show()