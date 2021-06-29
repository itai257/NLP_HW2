#import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader
from chu_liu_edmonds import decode_mst
from torch import Tensor


class KiperwasserDependencyParser(nn.Module):
    def __init__(self, *args):
        super(KiperwasserDependencyParser, self).__init__()
        word_embeddings = args[0]
        hidden_dim = args[1]
        word_vocab_size = args[2]
        tag_vocab_size = args[3]
        word_embedding_dim = word_vocab_size
        tag_embedding_dim = tag_vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #word_embedding_dim = args[4]
        #tag_embedding_dim = args[5]

        #?? not sure:
        #word_embedding_dim = hidden_dim
        #tag_embedding_dim = hidden_dim
        emb_dim = word_embedding_dim+tag_embedding_dim
        #
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim) # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.pos_embedding =  nn.Embedding(tag_vocab_size, tag_embedding_dim) # Implement embedding layer for POS tags
        self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.encoder =  nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=False) # Implement BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
        #self.edge_scorer = # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference
        self.loss_function = nn.NLLLoss()  # Implement the loss function described above

    def forward(self, sentence):

        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

        # Pass word_idx and pos_idx through their embedding layers
        word_embeds = self.word_embedding(word_idx_tensor.to(self.device))
        pos_embeds = self.pos_embedding(pos_idx_tensor.to(self.device))

        # Concat both embedding outputs
        embeds = torch.cat((word_embeds, pos_embeds), 2) #[sentence_length, word_embed + pos_embed]

        # Get Bi-LSTM hidden representation for each word+pos in sentence
        lstm_out, _ = self.encoder(embeds.view(embeds.shape[1], 1, -1))

        # Get score for each possible edge in the parsing graph, construct score matrix

        # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix

        # Calculate the negative log likelihood loss described above
        loss = predicted_tree = 0
        return loss, predicted_tree
