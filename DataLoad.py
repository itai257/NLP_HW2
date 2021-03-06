from collections import defaultdict
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from collections import Counter
import torch
import numpy as np

# These are not relevant for our POS tagger but might be usefull for HW2
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
ROOT_TOKEN = PAD_TOKEN # this can be used if you are not padding your batches
#ROOT_TOKEN = "<root>" # use this if you are padding your batches and want a special token for ROOT
#SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN]
SPECIAL_TOKENS = [UNKNOWN_TOKEN, ROOT_TOKEN]

def split(string, delimiters):
    """
        Split strings according to delimiters
        :param string: full sentence
        :param delimiters string: characters for spliting
            function splits sentence to words
    """
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack


def get_vocabs(list_of_paths):
    """
        Extract vocabs from given datasets. Return a word2ids and tag2idx.
        :param file_paths: a list with a full path for all corpuses
            Return:
              - word2idx
              - tag2idx
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    alpha = 0.25

    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                if line == "\n":
                    continue
                splited_line = split(line, ('\t', '\n'))
                token = splited_line[1].lower()
                pos_tag = splited_line[3]
                word_dict[token] += 1
                pos_dict[pos_tag] += 1

    return word_dict, pos_dict




class PosDataReader:
    def __init__(self, file, word_dict, pos_dict):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""

        with open(self.file, 'r') as f:
            cur_sentence = [(ROOT_TOKEN, ROOT_TOKEN, -1)]
            for line in f:
                if line == "\n":
                    self.sentences.append(cur_sentence)
                    cur_sentence = [(ROOT_TOKEN, ROOT_TOKEN, -1)]
                    continue

                splited_line = split(line, ('\t', '\n'))
                token = splited_line[1].lower()
                pos_tag = splited_line[3]
                if splited_line[6] != '_':
                    true_head = int(splited_line[6])
                else:
                    true_head = -1
                # perform dropout:
                alpha = 0.25
                if (alpha / (alpha + self.word_dict[token])) > np.random.rand() and token not in SPECIAL_TOKENS:
                    cur_sentence.append((UNKNOWN_TOKEN, pos_tag, true_head))
                else:
                    cur_sentence.append((token, pos_tag, true_head))




    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class PosDataset(Dataset):
    def __init__(self, word_dict, pos_dict, dir_path: str, file_name: str, padding=False):
        super().__init__()
        #self.subset = subset  # One of the following: [train, test]
        self.file = dir_path + file_name
        self.datareader = PosDataReader(self.file, word_dict, pos_dict)
        self.vocab_size = len(self.datareader.word_dict)

        self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
           self.datareader.word_dict)

        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)

        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len, true_tree_heads = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len, true_tree_heads

    @staticmethod
    def init_word_embeddings(word_dict):
        glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}

        for i, pos in enumerate(sorted(pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            j = int(i + len(SPECIAL_TOKENS))
            pos_idx_mappings[str(pos)] = j
            idx_pos_mappings.append(str(pos))
        print("idx_pos_mappings -", idx_pos_mappings)
        print("pos_idx_mappings -", pos_idx_mappings)
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_len_list = list()
        true_tree_heads = list()
        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_idx_list = []
            pos_idx_list = []
            cur_true_tree_heads = []
            for word, pos, true_head in sentence:
                words_idx_list.append(self.word_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))
                cur_true_tree_heads.append(true_head)

            sentence_len = len(words_idx_list)
            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            true_tree_heads.append(torch.tensor(cur_true_tree_heads, dtype=torch.long, requires_grad=False))
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)

        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_len_list,
                                                                     true_tree_heads))}