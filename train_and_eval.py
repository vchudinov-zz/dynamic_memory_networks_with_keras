from dmn import DynamicMemoryNetwork
from preprocess import load_dataset
emb_location = '/home/penguinofdoom/Downloads/glove.6B/glove.6B.100d.txt'
babi_task_location = '/home/penguinofdoom/Downloads/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt'
model_folder = "../dmn"

print("----- Loading Dataset ----")
x_train, q_train, y_train, max_len = load_dataset(
    babi_task_location, emb_location)

epochs = 256
validation_split = 0.15
input_shape = x_train[0].shape
question_shape = q_train[0].shape
num_classes = len(y_train[0])
units = 256
emb_dim = 100
memory_steps = 3
dropout = 0.3
batch_size = 50
emb_dim = x_train[0].shape[-1]

print("----- Dataset Loaded. Compiling Model -----")
dmn_net = DynamicMemoryNetwork(save_folder=model_folder)
dmn_net.build_inference_graph(
    input_shape=input_shape,
    question_shape=question_shape,
    num_classes=num_classes,
    units=units,
    emb_dim=emb_dim,
    batch_size=batch_size,
    memory_steps=memory_steps,
    dropout=dropout)

print("------ Model Compiled. Training -------")
dmn_net.fit(x_train, q_train, y_train, batch_size=32,
            epochs=epochs,
            validation_split=0.15,
            l_rate=1e-3,
            l_decay=0,)
print("----- Model Trained. -----")
