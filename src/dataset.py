# !pip install gensim==4.2.0
# !pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# !pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# !pip install torch-geometric
import torch
from torch_geometric.data import Data
# !pip install transformers 
from torch.utils.data import Dataset
from transformers import BertTokenizer


# class for bipartite signed graph (need to increment edges differently)
class SignedBipartiteData(Data):
    def __init__(self,  x_s=None, x_t=None, pos_edge_index=None, neg_edge_index =None, 
                 pos_edge_weight =None, neg_edge_weight=None):
        super().__init__()
        self.pos_edge_index = pos_edge_index
        self.neg_edge_index = neg_edge_index
        self.pos_edge_weight = pos_edge_weight
        self.neg_edge_weight = neg_edge_weight
        self.x_s = x_s
        self.x_t = x_t
    def __inc__(self, key, value, *args, **kwargs):
      if key == 'pos_edge_index':
          return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
      elif key == 'neg_edge_index':
          return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
      else:
          return super().__inc__(key, value, *args, **kwargs)


# class for BERT text data
class BertDataset(Dataset):

    def __init__(self, df, bert_path):
        # pos_edges and neg_edges are dict of form {user_id: [entity1, entity2, etc.]}
        # get data
        data = df
        l = ['Republican', 'Brexit', 'BlackLivesMatter', 'climate', 'democrats']
        # BERt tokenizer
        self.tok = BertTokenizer.from_pretrained(bert_path)

       

        # get labels, parent posts, children posts, parent ids, children ids, pos and neg edges
        self.labels = list(data.label)
        # self.labels = list(data.subreddit.apply(lambda x: l.index(x)))
        self.parents = list(data.body_parent.apply(lambda x: self.tok.encode(x)))
        self.children = list(data.body_child.apply(lambda x: self.tok.encode(x)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = self.labels[idx]
        parent = self.parents[idx]
        child = self.children[idx]

        return label, parent, child

# collator (pads batch)
def collate(batch):

    batch_size = len(batch)

    labels = torch.tensor([l for l, _, _  in batch]).float()
    parents = [p for _, p, _ in batch]
    children = [c for _, _, c in batch]

    max_len_p = max(len(p) for p in parents)
    sent_pad_p = torch.zeros((batch_size, max_len_p), dtype=torch.int64)
    masks_pad_p = torch.zeros((batch_size, max_len_p), dtype=torch.int64)
    segs_pad_p = torch.zeros((batch_size, max_len_p), dtype=torch.int64)
    
    max_len_c = max(len(c) for c in children)
    sent_pad_c = torch.zeros((batch_size, max_len_c), dtype=torch.int64)
    masks_pad_c = torch.zeros((batch_size, max_len_c), dtype=torch.int64)
    segs_pad_c = torch.zeros((batch_size, max_len_c), dtype=torch.int64) # change this to ones to do pair classification?

    for i, p in enumerate(parents):
        sent_pad_p[i, :len(p)] = torch.tensor(p)
        masks_pad_p[i, :len(p)] = 1

    for i, c in enumerate(children):
        sent_pad_c[i, :len(c)] = torch.tensor(c)
        masks_pad_c[i, :len(c)] = 1
    
    return labels, sent_pad_p, masks_pad_p, segs_pad_p, sent_pad_c, masks_pad_c, segs_pad_c