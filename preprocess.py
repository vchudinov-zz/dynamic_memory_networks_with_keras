import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# TODO: Add proper labels.

def get_tokenizer(texts, max_words):

    tokenizer = Tokenizer(num_words=max_words,
                        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                        lower=True,#
                        split=" ",
                        char_level=False)
    tokenizer.fit_on_texts(texts)

    print('Found %s unique tokens.' % len(tokenizer.word_index))
    max_seq = max([len(text.split(' '))for text in texts if "?" not in text])
    max_q = max([len(text.split('\t')[0].split(" "))for text in texts if "?" in text])
    return tokenizer, max_seq, max_q

def tokenize_data(texts, max_seq, max_words, tokenizer=None):
    """
    """
    # TODO: Stream loading of new data.

    if tokenizer is None:
        tokenizer, word_index = get_tokenizer(np.reshape(t, newshape=-1), max_words)
    sequences = [tokenizer.texts_to_sequences(text) for text in texts]
    tokenized_data = [pad_sequences(sequence, maxlen=max_seq) for sequence in sequences]
    return tokenized_data, tokenizer, tokenizer.word_index

def load_embeddings(path_to_embs):
    """
    Stolen from keras blog
    """
    embeddings_index = {}
    f = open(path_to_embs, 'r')

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def generate_embeddings_matrix(word_index, embeddings_index, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, matrix_only=False):
    """
    Generates an embeddings matrix from existing embeddings and a word index.
    From keras blog
    """
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            # If the word is not in the embeddings dict generate a random fake embedding.

            embedding_matrix[i] = np.random.uniform(0.0,1.0,(EMBEDDING_DIM,))
    return embedding_matrix


def get_positional_encoding(max_seq, emb_size):
    """ Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)
        From https://github.com/domluna/memn2n
    """
    encoding = np.ones((emb_size, max_seq), dtype=np.float32)
    ls = max_seq+1
    le = emb_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / emb_size / max_seq
    encoding[:, -1] = 1.0 # TODO maybe remove
    return np.transpose(encoding)

def process_sentences(sentences, tokenizer, emb_matrix, max_seq, positional_encoding=None):

    def embed_sentence(sentence, emb_matrix):
        return [emb_matrix[word_index] for word_index in sentence]

    sentences = tokenizer.texts_to_sequences(sentences)
    sentences = pad_sequences(sentences, max_seq)
    sentences = [embed_sentence(sentence=s, emb_matrix=emb_matrix) for s in sentences]

    if positional_encoding is not None:
        sentences = np.sum(sentences*positional_encoding, axis=2)
    return sentences

# Taken from https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow
def encode_tasks(tasks, tokenizer, task_labels, embeddings_matrix, positional_encoding, max_seq):
    """
    Args:
        tasks (list) of (dict)s : holds all tasks
        tokenizer (keras.preprocessing.text.Tokenizer) - pre-trained tokenizer object used to encode inputs
        task_labels (list) : containing the unique set of answers for all tasks. Used to generate one-hot labels
        embeddings_matrix (np.array) : holds an embeddings_matrix for the dataset
        positional_encoding (np.array) : matrix used for positional_encoding of inputs
        max_seq (int) : the maximum sequence length. Used for padding.

    Return:
        (list) of (dict)s where each dict holds the embedded representations of the tasks in the dataset
    """
    for task in tasks:
        task['C'] = process_sentences(task['C'],  tokenizer, embeddings_matrix, max_seq, positional_encoding=positional_encoding)
        task['Q'] = process_sentences([task['Q']], tokenizer, embeddings_matrix, max_seq)[0]
        task['L'] = np.eye(len(task_labels))[task_labels.index(task["A"])]
        task['A'] = process_sentences([task['A']], tokenizer, embeddings_matrix, max_seq)[0]
    return tasks

def get_tasks(babi_task_location):
    """
    Retrieves Babi tasks into a readable dictionary form.
    Each task is represented as a dicitonary that contains the task facts,
    the task question, answer, the indices of the supporting facts, and the task
    ID.

    Taken from https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow/

    Args:
        babi_task_location (str) : path to a file containing babi tasks
    Returns:
        (list) of (dict)s holding a readable representation of the tasks
    """
    # TODO Add onehot label

    print("==> Loading test from %s" % babi_task_location)
    tasks = []
    task = None
    task_counter = 0
    task_labels = set()

    for i, line in enumerate(open(babi_task_location)):
        id = int(line[0:line.find(' ')])

        if id == 1:
            #  C - text; Q - question, A - answer, S - suporting fact, ID - task number
            task = {"C": [], "Q": "", "A": "", "S": "", "ID": ""}
            counter = 0
            id_map = {}

        line = line.strip()
        line = line.replace('.', ' <EOS>')
        line = line[line.find(' ')+1:]

        # if not a question
        if line.find('?') == -1:
            task["C"].append(line)
            id_map[id] = counter
            counter += 1

        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task_labels.add(task["A"])
            task["L"] = []
            for num in tmp[2].split():
                task["L"].append(id_map[int(num.strip())])
            task["ID"] = task_counter
            task_counter+=1

            tasks.append(task.copy())

    return tasks, list(task_labels)

def load_dataset(path_to_set, path_to_embs, emb_dim, tokenizer=None ):
    """
    Loads a babi dataset.
    Args:
        path_to_set (str) : path to the babi dataset you want to use
        path_to_embs (str) : path to  a file holding pretrained word embeddings
        emb_dim (int) : size of the word embeddings to use
        tokenizer (keras.preprocessing.text.Tokenizer) - tokenizer object used to encode inputs
    Return:
        (list), (list) , (list), (list): the encoded inputs, questions, answers and one-hot labels ready to be used by the model
    """

    embeddings = load_embeddings(path_to_embs=path_to_embs)
    if tokenizer is not None:
        tokenizer, max_seq, max_q = get_tokenizer(open(path_to_set, 'r').readlines(), 10000)
    matrix = generate_embeddings_matrix(tokenizer.word_index, embeddings, emb_dim, max_seq)
    positional_encoding = get_positional_encoding(max_seq, emb_dim)
    tasks, task_labels = get_tasks(babi_task_location=path_to_set)
    tasks = encode_tasks(tasks, task_labels=task_labels, tokenizer=tokenizer, embeddings_matrix=matrix,  max_seq=max_seq, positional_encoding=positional_encoding)
    x = [x['C'] for x in tasks]
    x_q = [x['Q'] for x in tasks]
    y = [x['A'] for x in tasks]
    l = [x['L'] for x in tasks]


    return x, x_q, y, l
