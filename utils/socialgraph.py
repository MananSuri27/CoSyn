import pickle
from socialnode import Node

social_network = Node()
with open("../data/social_network.pkl", 'wb') as f:
    pickle.dump(social_network,f)