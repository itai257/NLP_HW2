import torch
import torch.nn as nn
from chu_liu_edmonds import decode_mst

class KDependencyParserBasic(nn.Module):
    def __init__(self, word_vocab_size, tag_vocab_size, word_embed_dim, pos_embed_dim, biLSTM_hidden_dim,
                 mlp_dim, pretrained_words_embeddings=None):
        super(KDependencyParserBasic, self).__init__()

        # init hyper parameters
        self.word_embedding_dim = word_embed_dim
        self.tag_embedding_dim = pos_embed_dim
        self.biLSTM_hidden_dim = biLSTM_hidden_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.biLSTM_in_size = self.word_embedding_dim + self.tag_embedding_dim
        self.biLSTM_hidden_dim = self.biLSTM_in_size
        self.hidden_dim_MLP = mlp_dim

        if pretrained_words_embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_words_embeddings, freeze=True)
        else:
            self.word_embedding = nn.Embedding(word_vocab_size, self.word_embedding_dim)
        self.pos_embedding = nn.Embedding(tag_vocab_size, self.tag_embedding_dim)

        self.encoder = nn.LSTM(input_size=self.biLSTM_in_size, hidden_size=self.biLSTM_hidden_dim, num_layers=2,
                               bidirectional=True)
        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference

        self.mlp_layer_1 = nn.Linear(self.biLSTM_hidden_dim * 2 * 2, self.hidden_dim_MLP)
        self.activation = nn.Tanh()
        self.mlp_layer_2 = nn.Linear(self.hidden_dim_MLP, 1)

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
        v_head_modifier_matrix = self.get_all_appended_head_mod(lstm_out)
        score_matrix = self.mlp_layer_1(v_head_modifier_matrix)
        score_matrix = self.activation(score_matrix)
        score_matrix = self.mlp_layer_2(score_matrix)
        score_matrix_items = score_matrix.view(lstm_out.shape[0], lstm_out.shape[0])

        return score_matrix_items.unsqueeze(0)

    def get_all_appended_head_mod(self, lstm_out):
        X = lstm_out.permute(1, 0, 2).squeeze(0)
        X1 = X.unsqueeze(1)
        Y1 = X.unsqueeze(0)
        X2 = X1.repeat(1, X.shape[0], 1)
        Y2 = Y1.repeat(X.shape[0], 1, 1)
        Z = torch.cat([X2, Y2], -1)
        Z = Z.view(-1, Z.shape[-1]).to(self.device)
        return Z