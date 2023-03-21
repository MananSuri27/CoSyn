from node import Node
import pickle

import os

modes = ["test", "train", "val"]

graphs = []

def generateDGL(mode):
    path = "/speech/sreyan/aaai/members/"+mode

    for filename in os.listdir(path):
        id = filename.split(".")[0]
        graphs.append(Node(id=id, type =mode)[0])


for mode in modes:
    generateDGL(mode)


with open("./data/"+"finetuned_combined_tweet_id_reversed_2"+".pkl", 'wb') as f:
    pickle.dump(graphs,f)
