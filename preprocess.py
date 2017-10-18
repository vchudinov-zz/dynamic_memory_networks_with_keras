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

def embed_sentence(sentence, emb_matrix):

    return [emb_matrix[word_index] for word_index in sentence]

def get_positional_encoding(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((EMBEDDING_DIM, MAX_SEQUENCE_LENGTH), dtype=np.float32)
    ls = MAX_SEQUENCE_LENGTH+1
    le = EMBEDDING_DIM+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / EMBEDDING_DIM / MAX_SEQUENCE_LENGTH
    #encoding[:, -1] = 1.0 # TODO maybe remove
    return np.transpose(encoding)

def process_texts(sentences, tokenizer, emb_matrix, max_seq, positional_encoding=None):
    sentences = tokenizer.texts_to_sequences(sentences)
    sentences = pad_sequences(sentences, max_seq)
    sentences = [embed_sentence(sentence=s, emb_matrix=emb_matrix) for s in sentences]

    if positional_encoding is not None:
        sentences = np.sum(sentences*positional_encoding, axis=2)
    return sentences

# Taken from https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow
def encode_tasks(tasks, tokenizer, embeddings_matrix,max_seq, positional_encoding):
    for task in tasks:
        task['C'] = process_texts(task['C'],  tokenizer, embeddings_matrix, max_seq, positional_encoding=positional_encoding)
        print(task["Q"])
        task['Q'] = process_texts([task['Q']], tokenizer, embeddings_matrix, max_seq)[0]
        task['A'] = process_texts([task['A']], tokenizer, embeddings_matrix, max_seq)[0]
    return tasks

def get_tasks(babi_task_location):
    """
    Retrieves Babi tasks into a readable dictionary form.
    Taken from https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow/
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
            #  C - text; Q - question, A - answer, L - label, ID - task number
            task = {"C": [], "Q": "", "A": "", "L": "", "ID": ""}
            counter = 0
            id_map = {}

        line = line.strip()
        line = line.replace('.', ' <EOS>')
        line = line[line.find(' ')+1:]

        # if not a question
        if line.find('?') == -1:

            #line = line.split(' ')

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

    return tasks, task_labels

def load_dataset(path_to_set, path_to_embs, emb_dim, tokenizer=None ):

    embeddings = load_embeddings(path_to_embs=path_to_embs)
    tokenizer, max_seq, max_q = get_tokenizer(open(path_to_set, 'r').readlines(), 10000)
    matrix = generate_embeddings_matrix(tokenizer.word_index, embeddings, emb_dim, max_seq)
    positional_encoding = get_positional_encoding(max_seq, emb_dim)
    tasks, task_labels = get_tasks(babi_task_location=path_to_set)
    tasks = encode_tasks(tasks, tokenizer=tokenizer, embeddings_matrix=matrix,  max_seq=max_seq, positional_encoding=positional_encoding)
    x = [x['C'] for x in tasks]
    x_q = [x['Q'] for x in tasks]
    y = [x['A'] for x in tasks]


    return x, x_q, y
