import torch
import pickle
from torch.utils.data import DataLoader

from DataLoad import PosDataset
from chu_liu_edmonds import decode_mst

PATH_MODEL_BASIC = "data/basic/basic_model.pkl"
PATH_MODEL_ADVANCED = "data/advanced/advanced_model.pkl"

PATH_WORD_VOCAB_BASIC = "data/basic/word_vocabulary.pkl"
PATH_POS_VOCAB_BASIC = "data/basic/pos_vocabulary.pkl"

PATH_WORD_VOCAB_ADVANCED = "data/advanced/word_vocabulary.pkl"
PATH_POS_VOCAB_ADVANCED = "data/advanced/pos_vocabulary.pkl"

PATH_COMP_UNLABELED = "data/comp.unlabeled"

PATH_COMP_LABELED_BASIC = "data/comp_m1_203903018.labeled"
PATH_COMP_LABELED_ADVANCED = "data/comp_m2_203903018.labeled"

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

def load_vocabs(word_path, pos_path):
    with open(word_path, 'rb') as input:
        word_dict = pickle.load(input)
    with open(pos_path, 'rb') as input:
        pos_dict = pickle.load(input)
    return word_dict, pos_dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



####################### basic model:
print("starting inference competitions on basic model")
word_dict, pos_dict =  load_vocabs(PATH_WORD_VOCAB_BASIC, PATH_POS_VOCAB_BASIC)
model = torch.load(PATH_MODEL_BASIC, map_location=torch.device(device))
model.eval()

comp = PosDataset(word_dict, pos_dict, 'data/', 'comp.unlabeled', padding=False)
comp_dataloader = DataLoader(comp, shuffle=False)

sentences_predictions = []
for idx, sentence in enumerate(comp_dataloader):
    words_idx_tensor, pos_idx_tensor, _, _ = sentence
    score_matrix = model((words_idx_tensor, pos_idx_tensor, _))
    score_matrix_numpy = score_matrix.squeeze(0).cpu().detach().numpy()
    predicted_tree, _ = decode_mst(score_matrix_numpy, len(score_matrix_numpy), has_labels=False)
    sentences_predictions.append(predicted_tree[1:])

sentences_lines = []
with open(PATH_COMP_UNLABELED) as f:
    cur_sentence_lines = []
    for line in f:
        if line == "\n":
            sentences_lines.append(cur_sentence_lines)
            cur_sentence_lines = []
            continue
        cur_sentence_lines.append(line)

assert len(sentences_lines) == len(sentences_predictions)

sentence_ot_write = []
for sentences_lines, predictions in zip(sentences_lines, sentences_predictions):
    assert len(sentences_lines) == len(predictions)
    for line, pred in zip(sentences_lines, predictions):
        splited_line = split(line, ('\t'))
        splited_line[6] = str(pred)
        sentence_ot_write.append('\t'.join(splited_line))
    sentence_ot_write.append('\n')


with open(PATH_COMP_LABELED_BASIC, 'w+') as output:
    for line in sentence_ot_write:
        output.write(line)

################ advanced model:


print("starting inference competitions on advanced model")
word_dict, pos_dict =  load_vocabs(PATH_WORD_VOCAB_ADVANCED, PATH_POS_VOCAB_ADVANCED)
model = torch.load(PATH_MODEL_ADVANCED, map_location=torch.device(device))
model.eval()

comp = PosDataset(word_dict, pos_dict, 'data/', 'comp.unlabeled', padding=False)
comp_dataloader = DataLoader(comp, shuffle=False)

sentences_predictions = []
for idx, sentence in enumerate(comp_dataloader):
    words_idx_tensor, pos_idx_tensor, _, _ = sentence
    score_matrix = model((words_idx_tensor, pos_idx_tensor, _))
    score_matrix_numpy = score_matrix.squeeze(0).cpu().detach().numpy()
    predicted_tree, _ = decode_mst(score_matrix_numpy, len(score_matrix_numpy), has_labels=False)
    sentences_predictions.append(predicted_tree[1:])

sentences_lines = []
with open(PATH_COMP_UNLABELED) as f:
    cur_sentence_lines = []
    for line in f:
        if line == "\n":
            sentences_lines.append(cur_sentence_lines)
            cur_sentence_lines = []
            continue
        cur_sentence_lines.append(line)

assert len(sentences_lines) == len(sentences_predictions)

sentence_ot_write = []
for sentences_lines,predictions in zip(sentences_lines, sentences_predictions):
    assert len(sentences_lines) == len(predictions)
    for line, pred in zip(sentences_lines, predictions):
        splited_line = split(line, ('\t'))
        splited_line[6] = str(pred)
        sentence_ot_write.append('\t'.join(splited_line))
    sentence_ot_write.append('\n')


with open(PATH_COMP_LABELED_ADVANCED, 'w+') as output:
    for line in sentence_ot_write:
        output.write(line)