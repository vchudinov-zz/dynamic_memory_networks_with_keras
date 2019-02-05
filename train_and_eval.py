from dmn import DynamicMemoryNetwork
from preprocess import Data_Processor
import json
import argparse
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
"""
TODO: Batch sizes
TODO: Redo labelling - TODO
TODO: Named entities - TODO
TODO: RNN answer module
"""
parser = argparse.ArgumentParser(description='DMN+ Trainer')
parser.add_argument('--settings_file', type=str,
                    help='path to a json with settings')

settings = parser.parse_args()

settings = json.load(open(settings.settings_file, 'r'))

print("----- Loading Dataset ----")
p = Data_Processor(glove_path=settings["path_to_embeddings"])
max_len, trainset, testset = p.preprocess_data(task_location=settings["path_to_train_task"])

input_shape = trainset[0][0].shape
question_shape = trainset[1][0].shape
num_classes = len(trainset[2][0])

print("----- Dataset Loaded. Compiling Model -----")
dmn_net = DynamicMemoryNetwork(save_folder=settings["model_folder"])
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

#if testset is not None:
 ##   print("----- Model Trained. Evaluating -----")
res = dmn_net.model.evaluate(x=[testset[0], testset[1]],
                                   y=testset[2],
                                   batch_size=settings["batch_size"])

predictions = dmn_net.model.predict([testset[0], testset[1]],batch_size=settings["batch_size"])
predictions = np.argmax(predictions, axis=-1)

lb = preprocessing.LabelBinarizer()
#lb.fit_transform(predictions)
print("Result: ", res)
print(lb.fit_transform(predictions))
print("Sklearn: ", accuracy_score(y_true=testset[2], y_pred=predictions))
#print(f'Test Loss: {loss}, Test Accuracy: {acc}')
#