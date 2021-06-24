import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader
from chu_liu_edmonds import decode_mst


class KiperwasserDependencyParser(nn.Module):
    def __init__(self, *args):
        super(KiperwasserDependencyParser, self).__init__()
        self.word_embedding =  # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.pos_embedding =  # Implement embedding layer for POS tags
        self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.encoder =  # Implement BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
        self.edge_scorer =  # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference
        self.loss_function =  # Implement the loss function described above

    def forward(self, sentence):
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

        # Pass word_idx and pos_idx through their embedding layers

        # Concat both embedding outputs

        # Get Bi-LSTM hidden representation for each word+pos in sentence

        # Get score for each possible edge in the parsing graph, construct score matrix

        # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix

        # Calculate the negative log likelihood loss described above

        return loss, predicted_tree