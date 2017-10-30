from dmn import DynamicMemoryNetwork
from preprocess import load_dataset
import numpy as np
batch_size = 5
emb_dim = 50
emb_location = '/home/penguinofdoom/Downloads/glove.6B/glove.6B.50d.txt'
babi_task_location = '/home/penguinofdoom/Downloads/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt'
model_folder="dmn"
input_units = 16
episodic_memory_units = 16


x_train, q_train, y_train, l_train, classes_train = load_dataset( path_to_set=babi_task_location,
                                                                  embeddings_path=emb_location,
                                                                  emb_dim=emb_dim,
                                                                  tokenizer_path=None
                                                                  )
output_memory_units = len(classes_train)
dmn_net = DynamicMemoryNetwork( model_folder=model_folder,
                                input_units=input_units,
                                memory_units=episodic_memory_units,
                                max_seq=7,
                                output_units=output_memory_units
                                )

dmn_net.build_inference_graph(np.array(x_train[:10]), np.array(q_train[:10]), batch_size=5)
dmn_net.model.compile()
print("Model compiled")
