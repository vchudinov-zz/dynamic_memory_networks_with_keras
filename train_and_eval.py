from dmn import DynamicMemoryNetwork
from preprocess import load_dataset
from keras import optimizers
import json
import argparse

parser = argparse.ArgumentParser(description='DMN+ Trainer')
parser.add_argument('--settings_file', type=str,
                    help='path to a json with settings')

settings = parser.parse_args()
settings = json.loads(open(settings['--settings_file'], 'r'))

print("----- Loading Dataset ----")

max_len, trainset, testset = load_dataset(emb_location=settings["path_to_embeddings"],
                                           babi_location=settings["path_to_train_task"],
                                           babi_test_location=settings["path_to_test_task"],
                                           emb_dim=settings["embeddings_size"])

input_shape = trainset[0][0].shape
question_shape = trainset[1][0].shape
num_classes = len(trainset[2][0])

print("----- Dataset Loaded. Compiling Model -----")
dmn_net = DynamicMemoryNetwork(save_folder=model_folder)
dmn_net.build_inference_graph(
    input_shape=input_shape,
    question_shape=question_shape,
    num_classes=num_classes,
    units=settings["hidden_units"],
    batch_size=settings["batch_size"],
    memory_steps=settings["memory_steps"],
    dropout=settings["dropout"])

print("------ Model Compiled. Training -------")

dmn_net.fit(trainset[0], trainset[1], trainset[2],
            epochs=settings["epochs"],
            validation_split=settings["validation_split"],
            l_rate= settings["learning_rate"],
            l_decay=settings["learning_decay"],)

if testset is not None:
    print("----- Model Trained. Evaluating -----")
    loss, acc = dmn_net.model.evaluate(x=[testset[0], testset[1]],y=testset[2], batch_size=settings["batch_size"])
    print(f'Test Loss: {loss}, Test Accuracy: {acc}')
