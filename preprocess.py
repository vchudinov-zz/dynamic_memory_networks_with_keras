import re
from functools import reduce
import numpy as np
from keras.preprocessing.sequence import pad_sequences

"""
Adapted from https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py
"""


def get_positional_encoding(max_seq, emb_dim):
    """
    Generates  a positional encoding as described in section
    4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)
    Adapted from https://github.com/domluna/memn2n and
    https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow/
    Args:
        max_seq (int) : the maximum sequence size in texts. Data will be padded to that size
        emb_dim (int) : The size of the embedding vectors.
    Return:
        (np.array) with the positional encoding.
    """

    encoding = np.ones((emb_dim, max_seq), dtype=np.float32)
    ls = max_seq + 1
    le = emb_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / emb_dim / max_seq
    return np.transpose(encoding)


def load_embeddings_index(embeddings_path):
    """
    Takne from the Keras blog.
    Loads a  pre-trained embeddings matrix,
    such as the one from https://nlp.stanford.edu/projects/glove/

    Args:
        embeddings_path (str) : Location of the file holding the embeddings.
    Return:
        (dict) of embeddings
    """

    embeddings_index = {}
    f = open(embeddings_path, 'r')

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embeddings_index["<eos>"] = np.random.rand(100)
    return embeddings_index


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
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
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


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)

    def flatten(data): return reduce(lambda x, y: x + y, data)
    data = [
        (flatten(story),
         q,
         answer) for story,
        q,
        answer in data if not max_length or len(
            flatten(story)) < max_length]

    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    """Short summary.

    Parameters
    ----------
    data : type
        Description of parameter `data`.
    word_idx : type
        Description of parameter `word_idx`.
    story_maxlen : type
        Description of parameter `story_maxlen`.
    query_maxlen : type
        Description of parameter `query_maxlen`.

    Returns
    -------
    type
        Description of returned object.

    """
    xs = []
    xqs = []
    ys = []
    task_labels = sorted(list(set([x for _, _, x in data])))
    positional_encoding = get_positional_encoding(story_maxlen, 100)

    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.eye(len(task_labels))[task_labels.index(answer)]
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    xs = pad_sequences(xs, maxlen=story_maxlen)
    xs = [x * positional_encoding for x in xs]
    return np.array(xs), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)


def load_dataset(babi_location, emb_location):
    # TODO: Add progress bar
    print("----- Loading Embeddings.-----")
    word_index = load_embeddings_index(emb_location)
    print("----- Retrieving Stories. -----")
    stories = get_stories(open(babi_location, 'r'))
    story_maxlen = max(map(len, (x for x, _, _ in stories)))
    query_maxlen = max(map(len, (x for _, x, _ in stories)))

    print("----- Vectorizing Stories -----")
    vectorized_stories = vectorize_stories(
        stories, word_index, story_maxlen, query_maxlen)

    return vectorized_stories[0], vectorized_stories[1], vectorized_stories[2], story_maxlen
