# CoSyn
Implementation of [CoSyn: Detecting Implicit Hate Speech in Online Conversations Using a Context Synergized Hyperbolic Network](https://arxiv.org/abs/2303.03387).

![Model diagram](https://cdn.discordapp.com/attachments/1084345327399731342/1087834553160319106/cosyn.png)

## 🧱 Dependencies and Setup
### 🧰 Dependencies
- torch `1.11.0`
- geoopt `0.1.0` 
- dgl `1.0.0` 
- other required packages in `requirements.txt`

### 🛠️ Setup
```bash
# git clone this repository
git clone https://github.com/MananSuri27/CoSyn
cd CoSyn

# install python dependencies
pip3 install -r requirements.txt
```
We will need to move the code for Hyperbolic Graph Comvolution `models/hgconv.py` to the `dgl` library.
```bash
mv CoSyn/models/hgconv.py path-to-dgl/dgl/python/dgl/nn/pytorch/conv/
```
and correspondingly update the export statememt at `path-to-dgl/dgl/python/dgl/nn/pytorch/conv/__init__.py /`.


## 🔌 Dataset Processing
### 💬 Conversation Trees
**Required Files:**
- Node and edge features stored in `members` and `interactions` directory for each split `train`, `dev`, `test`. For each conversation tree, corresponding to a given parent node id `tweet_id`, there will be a `members/tweet_id.csv` and `interactions/tweet_id.csv` having the node features and edge list respectively.
- A file, `username2id.csv` which maps usernames to an id between [0,n) where n is number of users.
- Post embeddings for each post saved in the `embeds` directory.

**To generate the conversation trees and load them as a pickle file, run the following code:**
```bash
python3 utils/graphs.py
```

### 🌐 Social Graph

**Required Files:**
- User relation matrices stored as `[test/train/val]/matrix/file`, where multiple files can exist in each split, and each file is an adjacency list representation of edges between given users.
- Post embeddings of the last m(=100 in our paper) posts posted by the user in `embeds` directory, referenced by user ID.
- A file, `username2id.csv` which maps usernames to an id between [0,n) where n is number of users.

**To generate the conversation trees and load them as a pickle file, run the following code:**
```bash
python3 utils/socialgraph.py
```

## 🏋️ Training
To run the training script, run the following code:
```bash
python3 main.py
```

The arguments for `main.py` are as follows:   
```
 Arguments:  
  --x-size DIM          Embedding Dimension of Post
  --u-size DIM          Embedding Dimension of User 
  --g-size DIM          Output Dimension of HGCN
  --h-size DIM          Hidden Dimension of CHST
  --c C                 Curvature of Hyperbolic Space
  --batch-size  BS      Batch size
  --data-dir DIR        Directory for data
  --device DEVICE       Device
  --lr LR               Learning rate
  --dropout DROPOUT     Dropout probability
  --epochs EPOCHS       Maximum number of epochs to train for
  --weight-decay WEIGHT_DECAY
                        L2 regularization strength
  --optimizer OPTIMIZER
                        Which optimizer to use
  --patience PATIENCE   Patience for early stopping
  --save                Save computed results
  --save-dir SAVE_DIR   Path to save results
  --min-epochs MIN_EPOCHS
                        Do not early stop before min-epochs
```

