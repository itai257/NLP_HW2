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
    def __init__(self, biLSTM_hidden_dim, word_vocab_size, tag_vocab_size, pretrained_words_embeddings = None):
        super(KiperwasserDependencyParser, self).__init__()
        #if pretrained_words_embeddings is not None:
        #    self.word_embedding = nn.Embedding(word_vocab_size, 150)
        #    self.pre_trained_embeddings = nn.Embedding.from_pretrained(pretrained_words_embeddings, freeze=True)
        #    self.word_embedding_dim = 300+150
        #    self.tag_embedding_dim = 50
        #else:
        self.word_embedding_dim = 100
        self.tag_embedding_dim = 25
        self.word_embedding = nn.Embedding(word_vocab_size, self.word_embedding_dim)

        #self.biLSTM_hidden_size = biLSTM_hidden_dim+500
        self.biLSTM_in_size = self.word_embedding_dim + self.tag_embedding_dim
        self.biLSTM_hidden_size = self.biLSTM_in_size
        self.pos_embedding = nn.Embedding(tag_vocab_size, self.tag_embedding_dim)


        # self.device = "cpu"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #
        #self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.encoder = nn.LSTM(input_size=self.biLSTM_in_size, hidden_size=self.biLSTM_hidden_size, num_layers=2, bidirectional=True, batch_first=False) # Implement BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference
        self.hidden_dim_MLP = 100
        self.layer_1 = torch.nn.Linear(self.biLSTM_hidden_size*2*2, self.hidden_dim_MLP)
        self.layer_2 = torch.nn.Linear(self.hidden_dim_MLP, 1)
        self.activation = torch.tanh
        self.soft_max = nn.LogSoftmax(dim=0)
        self.mlp = nn.Sequential(
            torch.nn.Linear(self.biLSTM_hidden_size*2*2, self.hidden_dim_MLP),
            nn.Tanh(),
            nn.Linear(self.hidden_dim_MLP, 1))

    def forward(self, sentence):
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

        # Pass word_idx and pos_idx through their embedding layers
        word_idx_on_device = word_idx_tensor.to(self.device)
        word_embeds = torch.cat((self.word_embedding(word_idx_on_device).squeeze(0), self.pre_trained_embeddings(word_idx_on_device).squeeze(0)), dim = 1)
        pos_embeds = self.pos_embedding(pos_idx_tensor.to(self.device))

        #word_embeds = self.word_embeddings_one_hot[word_idx_tensor]
        #pos_embeds = self.pos_embeddings_one_hot[pos_idx_tensor]

        num_of_words = len(true_tree_heads[0])

        # Concat both embedding outputs
        embeds = torch.cat((word_embeds, pos_embeds.squeeze(0)), 1).to(self.device) #[sentence_length, word_embed + pos_embed]

        # Get Bi-LSTM hidden representation for each word+pos in sentence
        lstm_out, _ = self.encoder(embeds.unsqueeze(1).float())  # -> [num of words in sentence, 1, hidden_dim*2]


        # Get score for each possible edge in the parsing graph, construct score matrix

        #score_matrix_items = torch.zeros(num_of_words, num_of_words, dtype=torch.float32)
        #for h_idx in range(num_of_words):
        #    for m_idx in range(num_of_words):
        #        if h_idx == m_idx:
        #            continue
        #        v_head_modifier = torch.cat((lstm_out[h_idx], lstm_out[m_idx]), 1).to(self.device)
        #        score_matrix_items[h_idx][m_idx] = self.edge_scorer(v_head_modifier)
        #score_matrix = torch.tensor(score_matrix_items, requires_grad=True)
        v_head_modifier_matrix = self.get_all_appended_head_mod(lstm_out)
        score_matrix = self.edge_scorer(v_head_modifier_matrix).view(num_of_words, num_of_words)

        # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix
        soft_max_score_matrix = self.soft_max(score_matrix)

        score_matrix_to_decode = torch.tensor(score_matrix).cpu().numpy()

        predicted_tree, _ = decode_mst(score_matrix_to_decode, len(true_tree_heads[0]), has_labels=False)
        # -- predicted_tree, _ = decode_mst(score_matrix_to_decode, num_of_words, has_labels=False)

        # Calculate the negative log likelihood loss described above

        # -- return torch.from_numpy(predicted_tree), F.softmax(score_matrix, dim=0)
        return predicted_tree, soft_max_score_matrix

    def get_all_appended_head_mod(self, lstm_out):
        X = lstm_out.permute(1, 0, 2).squeeze(0)
        X1 = X.unsqueeze(1)
        Y1 = X.unsqueeze(0)
        X2 = X1.repeat(1, X.shape[0], 1)
        Y2 = Y1.repeat(X.shape[0], 1, 1)
        Z = torch.cat([X2, Y2], -1)
        Z = Z.view(-1, Z.shape[-1]).to(self.device)
        return Z
    def edge_scorer(self, v_head_modifier):
        x = self.mlp(v_head_modifier)
        return x