from collections import defaultdict
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F

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

    for s_t in SPECIAL_TOKENS:
        word_dict[s_t] += 1
        pos_dict[s_t] += 1

    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                if line == "\n":
                    continue
                splited_line = split(line, ('\t', '\n'))
                token = splited_line[1]
                pos_tag = splited_line[3]
                word_dict[token] += 1
                pos_dict[pos_tag] += 1

    #perform dropout:
    words_to_drop = []
    for word in word_dict.keys():
        if alpha / (alpha + word_dict[word]) > 0.2 and word not in SPECIAL_TOKENS:
            words_to_drop.append(word)

    for w in words_to_drop:
        word_dict.pop(w)
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
            cur_sentence = [(ROOT_TOKEN, ROOT_TOKEN)]
            cur_true_heads = ['-1']  # addition
            for line in f:
                if line == "\n":
                    self.sentences.append((cur_sentence, cur_true_heads))
                    cur_sentence = [(ROOT_TOKEN, ROOT_TOKEN)]
                    cur_true_heads = ['-1']  # addition
                    continue

                splited_line = split(line, ('\t', '\n'))
                token = splited_line[1]
                pos_tag = splited_line[3]
                true_head = splited_line[6]
                if token in self.word_dict.keys():
                    cur_sentence.append((token, pos_tag))
                    cur_true_heads.append(true_head)
                else:
                    # for words dropout
                    cur_sentence.append((UNKNOWN_TOKEN, UNKNOWN_TOKEN))
                    cur_true_heads.append('0')




    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class PosDataset(Dataset):
    def __init__(self, word_dict, pos_dict, dir_path: str, subset: str, padding=False):
        super().__init__()
        self.subset = subset  # One of the following: [train, test]
        self.file = dir_path + subset + ".labeled"
        self.datareader = PosDataReader(self.file, word_dict, pos_dict)
        self.vocab_size = len(self.datareader.word_dict)

        ## one hot embeddings:
        #self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
        #    self.datareader.word_dict)

        ## one hot embeddings:
        #self.pos_idx_mappings, self.idx_pos_mappings, self.pos_vectors = self.init_pos_vocab(self.datareader.pos_dict)

        #self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        #self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        #self.word_vector_dim = self.word_vectors.size(-1)
        #self.sentence_lens = [len(s[0]) for s in self.datareader.sentences]
        #self.max_seq_len = max(self.sentence_lens)

        self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
           self.datareader.word_dict)

        self.pos_idx_mappings, self.idx_pos_mappings, self.pos_vectors = self.init_pos_vocab(self.datareader.pos_dict)

        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len, true_tree_heads = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len, true_tree_heads

    @staticmethod
    #def init_word_embeddings(word_dict):
    #    glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
    #    return glove.stoi, glove.itos, glove.vectors

    def init_word_embeddings(word_dict):
        word_idx_mappings = dict()
        idx_word_mappings = []
        word_vectors = []
        i = 0
        for word in word_dict:
            word_idx_mappings[str(word)] = i
            idx_word_mappings.append(str(word))
            word_vectors.append(i)
            i += 1

        return word_idx_mappings, idx_word_mappings, F.one_hot(torch.tensor(word_vectors), num_classes=len(word_dict))

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}
        pos_vectors = sorted([i for i in range(len(SPECIAL_TOKENS))])

        for i, pos in enumerate(sorted(pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            j = int(i + len(SPECIAL_TOKENS))
            pos_idx_mappings[str(pos)] = j
            idx_pos_mappings.append(str(pos))
            pos_vectors.append(j)
        print("idx_pos_mappings -", idx_pos_mappings)
        print("pos_idx_mappings -", pos_idx_mappings)
        return pos_idx_mappings, idx_pos_mappings, F.one_hot(torch.tensor(pos_vectors), num_classes=(len(pos_dict) + len(SPECIAL_TOKENS)))

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_len_list = list()
        true_tree_heads = list()
        for sentence_idx, (sentence, true_heads) in enumerate(self.datareader.sentences):
            words_idx_list = []
            pos_idx_list = []
            for word, pos in sentence:
                words_idx_list.append(self.word_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))

            sentence_len = len(words_idx_list)
            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            true_tree_heads.append(torch.tensor([int(i) for i in true_heads], dtype=torch.long, requires_grad=False))
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