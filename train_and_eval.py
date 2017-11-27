from dmn import DynamicMemoryNetwork
from preprocess import load_dataset
from keras import optimizers

emb_location = '/home/penguinofdoom/Downloads/glove.6B/glove.6B.100d.txt'
#babi_test_task_location = '/home/penguinofdoom/Downloads/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt'
#babi_task_location = '/home/penguinofdoom/Downloads/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt'
babi_test_task_location = '/home/penguinofdoom/Downloads/tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_test.txt'
babi_task_location = '/home/penguinofdoom/Downloads/tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_train.txt'

model_folder = "../babi/task6"

print("----- Loading Dataset ----")
trainset, testset,  max_len = load_dataset(emb_location=emb_location,
                                           babi_location=babi_task_location,
                                           babi_test_location=babi_test_task_location,
                                           emb_dim=80)

epochs = 256
validation_split = 0.1
input_shape = trainset[0][0].shape
question_shape = trainset[1][0].shape
#input_shape = x_test[0].shape
#question_shape = q_test[0].shape
num_classes = len(trainset[2][0])
units = 80
emb_dim = 80
memory_steps = 3
dropout = 0.1
batch_size = 100

print("----- Dataset Loaded. Compiling Model -----")
dmn_net = DynamicMemoryNetwork(save_folder=model_folder)
dmn_net.build_inference_graph(
    input_shape=input_shape,
    question_shape=question_shape,
    num_classes=num_classes,
    units=units,
    batch_size=batch_size,
    memory_steps=memory_steps,
    dropout=dropout)


print("------ Model Compiled. Training -------")
dmn_net.fit(trainset[0], trainset[1], trainset[2],
            epochs=epochs,
            validation_split=validation_split,
            l_rate=0.001,
            l_decay=0,)
print("----- Model Trained. Evaluating -----")
loss, acc = dmn_net.model.evaluate(x=[testset[0], testset[1]],y=testset[2], batch_size=batch_size)
print(f'Test Loss: {loss}, Test Accuracy: {acc}')
