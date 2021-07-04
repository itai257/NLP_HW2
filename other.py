import math
from itertools import permutations
from collections import defaultdict, OrderedDict
from datetime import datetime
from time import time

import numpy as np
import torchtext
import torch
from torch import nn, cuda, tensor, zeros, cat
from torch.optim import Adam, Adagrad, AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchtext import vocab

#import matplotlib.pyplot as plt#

from chu_liu_edmonds import decode_mst

ROOT = "_R_"
TRIM_TRAIN_DATASET = 0
HIDDEN_DIM = 200
WORD_EMBEDDING_DIM = 300
POS_EMBEDDING_DIM = 20
EPOCHS = 1000
BATCH_SIZE = 5
LEARNING_RATE = 1e-4
TEST_SAMPLE_SIZE = 5
OOV = "out_of_vocab"


train_path = "data/train.labeled"
test_path = "data/test.labeled"
#model_file_name = "trained.model"
#uas_file_name = "uas.npy"
#loss_file_name = "loss.npy"


def get_vocab(sentences):
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    for sentence in sentences:
        for i in range(len(sentence)):
            word, pos = sentence[i][0], sentence[i][1]
            word_dict[word] += 1
            pos_dict[pos] += 1
    return word_dict, pos_dict


class ParsingDataset(Dataset):
    def __init__(self, train_sentences, all_sentences):
        self.sentences = train_sentences
        self.glove = vocab.GloVe("42B")
        self.word_embedding_dim = self.glove.dim
        self.word_idx = defaultdict(lambda:0, self.glove.stoi)
        _, self.pos_count = get_vocab(all_sentences)

        # # Set word id from vocab
        # self.word_idx = defaultdict(lambda:0)
        # # self.idx_word = dict()
        # self.words_list = [OOV] + list(self.word_count.keys())
        # for i in range(len(self.words_list)):
        #     w = self.words_list[i]
        #     self.word_idx[w] = i
        #     # self.idx_word[i] = w
        #
        # # Set POS id from vocab
        self.pos_idx = defaultdict(lambda:0)
        # # self.idx_pos = dict()
        self.pos_list = list(self.pos_count.keys())
        for i in range(len(self.pos_list)):
            pos = self.pos_list[i]
            self.pos_idx[pos] = i
            # self.idx_pos[i] = pos

        # self.word_vocab_size, self.pos_vocab_size = len(self.words_list), len(self.pos_list)
        self.pos_vocab_size = len(self.pos_list)

    def vectorize_tokens(self, tokens):
        idxs = [self.word_idx[w] for w in tokens]
        return tensor([idxs])

    def vectorize_pos(self, pos):
        pos_vector = torch.tensor([self.pos_idx[p] for p in pos])
        # pos_vector = zeros((len(pos), len(self.pos_idx.keys())), dtype=torch.int32)
        # for i in range(len(pos)):
        # pos_vector[(i, self.pos_idx[pos[i]])] = 1
        return pos_vector

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        s = self.sentences[idx]
        tokens_vector = self.vectorize_tokens([w[0] for w in s])
        pos_vector = self.vectorize_pos([w[1] for w in s])
        # arcs = [(int(s[i][2]), i) for i in range(len(s))]
        arcs = [int(s[i][2]) for i in range(len(s))]
        return (tokens_vector, pos_vector), tensor(arcs)


def extract_sentences(file_path):
    sentences = []
    with open(file_path, 'r') as f:
        cur_sentence = [(ROOT, ROOT, -1)]
        for line in f:
            if line != '\n':
                splitted = line.split('\t')
                cur_sentence.append((splitted[1], splitted[3], splitted[6]))
            else:
                sentences.append(cur_sentence)
                cur_sentence = [(ROOT, ROOT, -1)]
    return sentences

train_sentences = extract_sentences(train_path)
test_sentences = extract_sentences(test_path)

sentences = train_sentences + test_sentences


if TRIM_TRAIN_DATASET > 0:
    train_dataset = ParsingDataset(train_sentences[:TRIM_TRAIN_DATASET], sentences)
else:
    train_dataset = ParsingDataset(train_sentences, sentences)

test_dataset = ParsingDataset(test_sentences, sentences)

# import matplotlib.pyplot as plt
#
# arc_lengths = list()
# for i in range(len(train_dataset)):
#     _, arcs = train_dataset[i]
#     for j in range(len(arcs)):
#         arc_lengths.append(int(abs(j-arcs[j])))
#
#
# plt.figure(figsize=(10,10))
# plt.title("Distance between head and modifier histogram")
# plt.xlabel("Distance")
# plt.ylabel("Number of pairs")
# plt.hist(arc_lengths, bins=max(arc_lengths))
# plt.xlim(0, 10)
# plt.show()
#
# import matplotlib.pyplot as plt
#
# arcs2d = list()
# for i in range(len(train_dataset)):
#     _, arcs = train_dataset[i]
#     for j in range(len(arcs)):
#         arcs2d.append((arcs[j], j))
#
#
# plt.figure(figsize=(10,10))
# plt.title("Head-modifier pairs scatter plot")
# plt.xlabel("Head index")
# plt.ylabel("Modifier index")
# plt.scatter(*zip(*arcs2d), )
# plt.show()


class DependencyParsingNetwork(nn.Module):
    def __init__(self, pos_vocab_size, glove):
        super(DependencyParsingNetwork, self).__init__()

        # EMBEDDING
        self.word_embedding = nn.Embedding.from_pretrained(glove.vectors)
        # self.word_embedding = nn.Embedding(word_vocab_size, WORD_EMBEDDING_DIM, padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_vocab_size, POS_EMBEDDING_DIM, padding_idx=0)

        # RNN
        self.rnn = nn.GRU(input_size=WORD_EMBEDDING_DIM + POS_EMBEDDING_DIM, hidden_size=HIDDEN_DIM,
                            num_layers=2,
                            bidirectional=True, batch_first=True)

        # FC
        self.post_seq = nn.Sequential(
            nn.Linear(2*2*HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

        self.softmax = nn.Softmax(dim=0)

    def forward(self, token_vector, pos_vector):
        _tokens = self.word_embedding(token_vector).squeeze(0)
        _pos = self.pos_embedding(pos_vector).squeeze(0)
        x = cat((_tokens, _pos), dim=1)
        x = x.unsqueeze(0)
        x, hn = self.rnn(x)
        x = x.squeeze(0)
        scores = zeros(x.shape[0], x.shape[0])
        for t1, t2 in permutations(range(len(x)), 2):
            scores[t1][t2] = self.post_seq(cat((x[t1], x[t2])))
        scores = self.softmax(scores)
        return scores

def test_accuracy(model, test_data, sample_size=10):
    with torch.no_grad():
        edges_count = 0
        correct_edges_count = 0
        random_test_idx = torch.randint(len(test_data), (sample_size,))
        for i in random_test_idx:
            (tokens_vector, pos_vector), arcs = test_data[i]
            tokens_vector = tokens_vector.to(device)
            pos_vector = pos_vector.to(device)
            arc = arcs.to(device)
            scores = model(tokens_vector, pos_vector)
            mst, _ = decode_mst(scores.detach().numpy(), scores.shape[0], has_labels=False)
            edges_count += scores.shape[0]
            correct_edges_count += sum(np.equal(mst, arcs))
        accuracy = correct_edges_count / edges_count
        return accuracy


if __name__ == "__main__":


    device = 'cuda' if cuda.is_available() else 'cpu'
    print("Device = ", device)

    # glove = Vocab(train_dataset.word_count)
    # word_embeddings = glove.stoi, glove.itos, glove.vectors

    # glove = vocab.GloVe("42B", dim=WORD_EMBEDDING_DIM)
    model = DependencyParsingNetwork(train_dataset.pos_vocab_size, train_dataset.glove)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    avg_uas_history = list()
    loss_history = list()

    arcs_count = 0
    correct_predictions = 0

    for epoch in range(EPOCHS):
        t0 = time()
        L = 0
        edges_count = 0
        correct_edges_count = 0
        model.zero_grad()
        uas = list()
        random_batch_idx = torch.randint(len(train_dataset), (BATCH_SIZE,))
        for i in random_batch_idx:
            (tokens_vector, pos_vector), arcs = train_dataset[i]
            tokens_vector = tokens_vector.to(device)
            pos_vector = pos_vector.to(device)

            # Forward
            scores = model(tokens_vector, pos_vector)

            log_softmax_scores = F.log_softmax(scores, dim=0)
            loss = -torch.sum(torch.stack([log_softmax_scores[arcs[j]][j] for j in range(len(arcs))]))
            L += loss

            mst, _ = decode_mst(scores.detach().numpy(), scores.shape[0], has_labels=False)
            arcs_count += len(arcs) - 1
            correct_predictions += sum(np.equal(mst[1:], arcs[1:]))
            loss.backward()
            #
            # edges_count += scores.shape[0]
            # uas.append(sum(np.equal(mst[1:], arcs[1:])) / (len(arcs) - 1))

        for param in model.parameters():
            param = param / BATCH_SIZE

        optimizer.step()

        test_acc = test_accuracy(model, test_dataset, TEST_SAMPLE_SIZE)
        uas = correct_predictions / arcs_count

        avg_uas_history.append(float(uas))
        loss_history.append(float(L))

        print("-------------------------")
        print("> Epoch = ", epoch + 1, "/", EPOCHS, "took", time() - t0, "seconds")
        print("> Loss = ", float(L))
        print("> Train Accuracy = ", float(uas))
        print("> Test Accuracy = ", float(test_acc) )
