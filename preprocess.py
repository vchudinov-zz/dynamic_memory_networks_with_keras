import re
from functools import reduce
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras
import operator
from sklearn.preprocessing import LabelBinarizer

"""
Adapted from https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py
"""
class Data_Processor():
    def __init__(self, glove_path):
        self.glove = self.load_glove(glove_path=glove_path)

    def load_glove(self,glove_path):
        word2vec = {}

        print("==> loading glove")
        with open(glove_path) as f:
            for line in f:
                l = line.split()
                word2vec[l[0]] = [float(x) for x in l[1:]]

        print("==> glove is loaded")
        word2vec["<eos>"] = np.random.uniform(0.0, 1.0, (50))
        return word2vec

    def init_tasks(self, fname):
        # From improved DMN
        print("==> Loading test from %s" % fname)
        tasks = []
        task = None
        for i, line in enumerate(open(fname)):
            id = int(line[0:line.find(' ')])
            if id == 1:
                task = {"C": "", "Q": "", "A": ""}

            line = line.strip()
            line = line.replace('.', ' <eos> ')
            line = line.replace('to the', '') # remove stopword
            #line = line.replace('the', '')  # remove stopword

            line = line[line.find(' ') + 1:]
            if line.find('?') == -1:
                task["C"] += line
            else:
                idx = line.find('?')
                tmp = line[idx + 1:].split('\t')
                task["Q"] = line[:idx].strip().lower().split(" ")
                task["A"] = tmp[1].strip()
                tasks.append(task.copy())
        return tasks

    def preprocess_data(self, task_location):
        tasks = self.init_tasks(task_location)

        answers = [task["A"] for task in tasks]
        questions = [task["Q"] for task in tasks]
        tasks = [task["C"].lower().strip().split(" ") for task in tasks]
        tasks = [[word for word in task if len(word)>0] for task in tasks]
        max_story_len = max([len(task) for task in tasks])
        max_q_len = max([len(q) for q in questions])

        positional_encoding_task = self.positional_encoding(max_story_len, 50)
        positional_encoding_q = self.positional_encoding(max_q_len, 50)
        tasks = [[self.glove[word] for word in task] for task in tasks]
        tasks = [np.pad(task, ((0, max_story_len - len(task)), (0, 0)), 'constant') for task in tasks]
        tasks = [task * positional_encoding_task for task in tasks]

        questions = [[self.glove[word] for word in question] for question in questions]
        questions = [np.pad(q, ((0, max_q_len - len(q)), (0, 0)), 'constant') for q in questions]
        questions = [q * positional_encoding_q for q in questions]
        encoder = LabelBinarizer()
        answers = encoder.fit_transform(answers)

        return max_story_len, (np.array(tasks), np.array(questions), np.array(answers)), None



    def positional_encoding(self, sentence_size, embedding_size):
        """We could have used RNN for parsing sentence but that tends to overfit.
        The simpler choice would be to take sum of embedding but we loose loose positional information.
        Position encoding is described in section 4.1 in "End to End Memory Networks" in more detail (http://arxiv.org/pdf/1503.08895v5.pdf)"""
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size+1
        le = embedding_size+1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        return np.transpose(encoding)




def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    s = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    s = [x if x != "." else "<eos>" for x in s]
    s = [x.lower() for x in s if x != "?"]
    return s


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    #story = []

    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            q = q + ['<eos>']
            a = [a] + ['<eos>']
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None, word_idx=None, emb_dim=0):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    max_fact_len = max([max([len(fact) for fact in story]) for story, _, _ in data])*emb_dim
    data = [vectorize_story(x, word_idx, max_fact_len) for x in data]
    max_story_length = max([len(x) for x, _, _, in data])
    #positional_encoding = get_positional_encoding(max_story_length, len(word_idx["<eos>"]))
    positional_encoding_facts = get_positional_encoding(max_story_length, max_fact_len)
    positional_encoding_q = get_positional_encoding(4, emb_dim-1)

    # Label Set.
    task_labels = sorted(list(set([x[0] for _, _, x in data])))

    xs = pad_sequences([x for x,_,_ in data], maxlen=max_length, padding="post")
    xs = [x * positional_encoding_facts for x in xs]
    xqs = [x for _, x, _ in data]
    xqs = [xq * positional_encoding_q for xq in xqs]
    encoder = LabelBinarizer()
    ys = encoder.fit_transform([x[0] for _,_, x in data])
    query_maxlen = max([len(x) for x in xqs])
    return max_length, np.array(xs), pad_sequences(xqs, maxlen=query_maxlen, padding="post"), np.array(ys)

def vectorize_story(datum, word_idx, max_fact_len):

    story = datum[0]
    question= datum[1]
    label= datum[2]

    facts = []
    for fact in story:
        new_fact = reduce(operator.iconcat, [word_idx[w] for w in fact ], [])
        new_fact = new_fact + [0.0]*(max_fact_len - len(new_fact))
        facts.append(new_fact)
    question=  [word_idx[w] for w in question]
    return facts, question, label


def vectorize_stories(story, word_idx,):

    xs = []
    xqs = []
    ys = []
    task_labels = sorted(list(set([x[0] for _, _, x in data])))
    print(task_labels)


    for story, query, answer in data:
        print(story)
        print(query)
        print(answer)
        x = [word_idx[w] for w in story]
        y = [w for w in story if w not in word_idx.keys()]
        if len(y) > 0:
            print(y)

        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.eye(len(task_labels))[task_labels.index(answer[0])]
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    xs = pad_sequences(xs, maxlen=story_maxlen, padding="post")
    xs = [x * positional_encoding for x in xs]
    return np.array(xs), pad_sequences(xqs, maxlen=query_maxlen, padding="post"), np.array(ys)


def load_dataset( emb_location, babi_location, babi_test_location=None, emb_dim=80):
    print("----- Loading Embeddings.-----")
    word_index = load_embeddings_index(emb_location, emb_dim)
    print("----- Retrieving Stories. -----")

    story_maxlen, stories, questions, labels = get_stories(open(babi_location, 'r'),
                                                           word_idx=word_index,
                                                           emb_dim=emb_dim,
                                                           only_supporting=False)

    print(story_maxlen)
    raise SystemExit
    return  story_maxlen, (stories, questions, labels), None





def create_vector(word, word2vec, word_vector_size, silent=False):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print("utils.py::create_vector => %s is missing" % word)
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    '''
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    '''

    if not word in vocab:
        next_index = len(vocab)
        vocab[word] = float(next_index)
        ivocab[next_index] = word

    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")

    def pe_matrix(self, num_words):
        embedding_size = self.dim

        pe_matrix = np.ones((num_words, embedding_size))

        for j in range(num_words):
            for i in range(embedding_size):
                value = (i + 1. - (embedding_size + 1.) / 2.) * (j + 1. - (num_words + 1.) / 2.)
                pe_matrix[j,i] = float(value)
        pe_matrix = 1. + 4. * pe_matrix / (float(embedding_size) * num_words)

    return pe_matrix