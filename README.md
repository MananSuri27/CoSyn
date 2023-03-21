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


## Dataset Processing

## Training

## Citation

