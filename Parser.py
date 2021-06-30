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
        biLSTM_hidden_size = args[1]
        word_vocab_size = args[2]
        tag_vocab_size = args[3]
        word_embedding_dim = word_vocab_size
        tag_embedding_dim = tag_vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #word_embedding_dim = args[4]
        #tag_embedding_dim = args[5]

        #?? not sure:
        #word_embedding_dim = biLSTM_hidden_size
        #tag_embedding_dim = biLSTM_hidden_size
        biLSTM_in_size = word_embedding_dim+tag_embedding_dim
        #
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim) # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.pos_embedding =  nn.Embedding(tag_vocab_size, tag_embedding_dim) # Implement embedding layer for POS tags
        self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.encoder = nn.LSTM(input_size=biLSTM_in_size, hidden_size=biLSTM_hidden_size, num_layers=2, bidirectional=True, batch_first=False) # Implement BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
        self.edge_scorer = MLP(biLSTM_hidden_size*2*2) # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference
        #self.loss_function = nn.NLLLoss()  # Implement the loss function described above

    def forward(self, sentence):

        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

        # Pass word_idx and pos_idx through their embedding layers
        word_embeds = self.word_embedding(word_idx_tensor.to(self.device))
        pos_embeds = self.pos_embedding(pos_idx_tensor.to(self.device))

        # Concat both embedding outputs
        embeds = torch.cat((word_embeds, pos_embeds), 2) #[sentence_length, word_embed + pos_embed]

        # Get Bi-LSTM hidden representation for each word+pos in sentence
        lstm_out, _ = self.encoder(embeds.view(embeds.shape[1], 1, -1))  # -> [num of words in sentence, 1, hidden_dim*2]


        # Get score for each possible edge in the parsing graph, construct score matrix
        num_of_words = len(lstm_out)
        v_head_modifier_matrix = []
        #score_matrix = np.zeros((num_of_words, num_of_words))
        score_matrix = torch.zeros(num_of_words, num_of_words, dtype=torch.float32)
        for h_idx in range(num_of_words):
            curr_matrix = []
            for m_idx in range(num_of_words):
                v_head_modifier = torch.cat((lstm_out[h_idx], lstm_out[m_idx]), 1)
                curr_matrix.append(v_head_modifier)
                score = self.edge_scorer.forward(v_head_modifier)
                score_matrix[h_idx][m_idx] = score.item()
            v_head_modifier_matrix.append(curr_matrix)

        # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix
        predicted_tree, _ = decode_mst(score_matrix, num_of_words, has_labels=False)

        # Calculate the negative log likelihood loss described above
        #loss = 0
        return torch.from_numpy(predicted_tree), F.softmax(torch.tensor(score_matrix, requires_grad=True), dim = 0)



class MLP(nn.Module):
   def __init__(self, layer1_input_dim):
       super(MLP, self).__init__()
       self.hidden_dim = 100
       self.layer_1 = torch.nn.Linear(layer1_input_dim, self.hidden_dim)
       self.layer_2 = torch.nn.Linear(self.hidden_dim, 1)
       self.activation = torch.tanh

   def forward(self, x):
       x = self.layer_1(x)        # x.size() -> [batch_size, self.hidden_dim]
       x = self.activation(x)     # x.size() -> [batch_size, self.hidden_dim]
       x = self.layer_2(x)        # x.size() -> [batch_size, 1]
       return x
