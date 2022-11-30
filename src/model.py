# from torch_scatter import scatter_add
# from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Union
import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, PairTensor, OptTensor
# from torch_geometric.nn import SignedConv
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from helper_functions import mean_pool

# def gcn_norm(edge_index, edge_weight=None, num_nodes=None):
#         num_nodes = maybe_num_nodes(edge_index, num_nodes)
#         row, col = edge_index[0], edge_index[1]
#         idx = col
#         deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
#         deg_inv_sqrt = deg.pow_(-0.5)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#         return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class SignedConvWeighted(MessagePassing):
    r"""The signed graph convolutional operator from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper

   Modified from pytorch geometric implementation at 
   https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/signed_conv.html

    """
    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr

        if first_aggr:
            self.lin_pos_l = Linear(in_channels, out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(in_channels, out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)
        else:
            self.lin_pos_l = Linear(2 * in_channels, out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(2 * in_channels, out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_pos_l.reset_parameters()
        self.lin_pos_r.reset_parameters()
        self.lin_neg_l.reset_parameters()
        self.lin_neg_r.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj, pos_edge_weight: OptTensor = None, neg_edge_weight: OptTensor = None):
        """"""
        if pos_edge_weight == None:
          pos_edge_weight = torch.ones((len(pos_edge_index)))
        if neg_edge_weight == None:
          neg_edge_weight = torch.ones((len(neg_edge_index)))

        # pos_edge_index, pos_edge_weight = gcn_norm(pos_edge_index, pos_edge_weight)
        # neg_edge_index, neg_edge_weight = gcn_norm(neg_edge_index, neg_edge_weight)

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor)
        if self.first_aggr:
            
            out_pos = self.propagate(pos_edge_index, edge_weight=pos_edge_weight, x=x, size=None)
            out_pos = self.lin_pos_l(out_pos)
            out_pos = out_pos + self.lin_pos_r(x[1])

            out_neg = self.propagate(neg_edge_index, edge_weight=neg_edge_weight, x=x, size=None)
            out_neg = self.lin_neg_l(out_neg)
            out_neg = out_neg + self.lin_neg_r(x[1])

            return torch.cat([out_pos, out_neg], dim=-1)

        else:
            F_in = self.in_channels

            out_pos1 = self.propagate(pos_edge_index, edge_weight=pos_edge_weight, size=None,
                                      x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_pos2 = self.propagate(neg_edge_index, edge_weight=neg_edge_weight, size=None,
                                      x=(x[0][..., F_in:], x[1][..., F_in:]))
            out_pos = torch.cat([out_pos1, out_pos2], dim=-1)
            out_pos = self.lin_pos_l(out_pos)
            out_pos = out_pos + self.lin_pos_r(x[1][..., :F_in])

            out_neg1 = self.propagate(pos_edge_index, edge_weight=pos_edge_weight, size=None,
                                      x=(x[0][..., F_in:], x[1][..., F_in:]))
            out_neg2 = self.propagate(neg_edge_index, edge_weight=neg_edge_weight,size=None,
                                      x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_neg = torch.cat([out_neg1, out_neg2], dim=-1)
            out_neg = self.lin_neg_l(out_neg)
            out_neg = out_neg + self.lin_neg_r(x[1][..., F_in:])

            return torch.cat([out_pos, out_neg], dim=-1)


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: PairTensor) -> Tensor:
        adj_t = adj_t.set_value(None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, first_aggr={self.first_aggr})')


class BertEntitiesGraph(nn.Module):
    
    def __init__(self, dropout_rate, hidden_size, bert_path):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.bert_path = bert_path

        # encode parent, child with bert 
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.conv1 = SignedConvWeighted(-1, self.hidden_size, first_aggr=True) # 100*25 => 100*50
        self.conv1_ent = SignedConvWeighted(-1, self.hidden_size, first_aggr=True) # have a different set of params for entities 
        
        self.conv2 = SignedConvWeighted(-1, self.hidden_size//2, first_aggr=True) 
        self.conv2_ent =  SignedConvWeighted(-1, self.hidden_size//2, first_aggr=True) 

        self.conv3 = SignedConvWeighted(-1, self.hidden_size//4, first_aggr = True)


        self.lin1 = nn.Linear(((self.hidden_size//4)*4), 3)
        # self.lin2 = nn.Linear(self.hidden_size, 3) # labels = 3

    
    def forward(self, parents, parents_masks, parents_segs, 
                children, children_masks, children_segs, 
                parents_user_feat, parents_entities_feat, 
                children_user_feat, children_entities_feat, 
                parents_pos_e, parents_neg_e,
                parents_pos_weight, parents_neg_weight,
                children_pos_e, children_neg_e,
                children_pos_weight, children_neg_weight): 

      bert_parents = mean_pool(self.bert, parents, parents_masks, parents_segs) # 1* 768 mean pooling
      bert_children= mean_pool(self.bert, children, children_masks, children_segs)
      # bert_parents_pooled, bert_parent_tokens = self.bert(parents, parents_masks, parents_segs) # 1* 768 mean pooling
      # bert_children_pooled, bert_children_tokens = self.bert(children, children_masks, children_segs)

      index = torch.LongTensor([1, 0])
      parents_pos_e_reversed = parents_pos_e[index]
      parents_neg_e_reversed = parents_neg_e[index]
      children_pos_e_reversed = children_pos_e[index]
      children_neg_e_reversed = children_neg_e[index]

      
      # dim = hidden size *2 because friends vector + enemies vector 
      # first conv
      # users < neighbor entities
      parents_conv1 = F.relu(self.conv1(x =(parents_entities_feat, parents_user_feat), 
                                                    pos_edge_index = parents_pos_e, neg_edge_index = parents_neg_e,
                                                    pos_edge_weight = parents_pos_weight, neg_edge_weight = parents_neg_weight))

      # need to do target to source for entities to message pass because bipartite 
      parents_ent_conv1 =  F.relu(self.conv1_ent(x =(parents_conv1, parents_entities_feat), # entities < user neighbors (ah but that has no info since all users same...)
                                                    pos_edge_index = parents_pos_e_reversed, neg_edge_index = parents_neg_e_reversed,
                                                    pos_edge_weight = parents_pos_weight, neg_edge_weight = parents_neg_weight))
     
      # 2d conv (we are actually getting info from entities who got info from users in 1st conv!)
      # so maybe we don't need 3d conv? 
      parents_conv2 = F.relu(self.conv2(x =(parents_ent_conv1, parents_conv1), 
                                                    pos_edge_index = parents_pos_e, neg_edge_index = parents_neg_e,
                                                    pos_edge_weight = parents_pos_weight, neg_edge_weight = parents_neg_weight)) # users < entities < users

      parents_ent_conv2 =  F.relu(self.conv2_ent(x =(parents_conv2, parents_ent_conv1), 
                                           pos_edge_index = parents_pos_e_reversed, neg_edge_index = parents_neg_e_reversed,
                                           pos_edge_weight = parents_pos_weight, neg_edge_weight = parents_neg_weight))
  
      # 3d conv 
      parents_conv3 = self.conv3(x =(parents_ent_conv2, parents_conv2), 
                                                   pos_edge_index = parents_pos_e, neg_edge_index = parents_neg_e,
                                                    pos_edge_weight = parents_pos_weight, neg_edge_weight = parents_neg_weight)
     
      # parents_ent_conv3 =  self.dropout(F.relu(self.conv3(x =(parents_conv3, parents_ent_conv2), 
      #                                          pos_edge_index = parents_pos_e_reversed, neg_edge_index = parents_neg_e_reversed)))

      # print(parents_conv3.shape)
      # children_conv1 = self.dropout(F.relu(self.conv1(x =(children_entities_feat, children_user_feat), 
      #                                               pos_edge_index = children_pos_e, neg_edge_index = children_neg_e)))
      children_conv1 = F.relu(self.conv1(x =(children_entities_feat, children_user_feat), 
                                                    pos_edge_index = children_pos_e, neg_edge_index = children_neg_e,
                                                    pos_edge_weight = children_pos_weight, neg_edge_weight = children_neg_weight))
    
      children_ent_conv1 =  F.relu(self.conv1_ent(x =(children_conv1, children_entities_feat), 
                                                    pos_edge_index = children_pos_e_reversed, neg_edge_index = children_neg_e_reversed,
                                                    pos_edge_weight = children_pos_weight, neg_edge_weight = children_neg_weight))
    
      # 2d conv
      children_conv2 = F.relu(self.conv2(x =(children_ent_conv1, children_conv1), 
                                                    pos_edge_index = children_pos_e, neg_edge_index = children_neg_e,
                                                    pos_edge_weight = children_pos_weight, neg_edge_weight = children_neg_weight))
    
      children_ent_conv2 =  F.relu(self.conv2_ent(x =(children_conv2, children_ent_conv1), 
                                                    pos_edge_index = children_pos_e_reversed, neg_edge_index = children_neg_e_reversed,
                                                    pos_edge_weight = children_pos_weight, neg_edge_weight = children_neg_weight))
 
      # # 3d conv 
      children_conv3 = self.conv3(x =(children_ent_conv2, children_conv2), 
                                                   pos_edge_index = children_pos_e, neg_edge_index = children_neg_e,
                                                    pos_edge_weight = children_pos_weight, neg_edge_weight = children_neg_weight)
   
      # children_ent_conv3 =  self.dropout(F.relu(self.conv3(x =(children_conv3, children_ent_conv2), 
      #                                          pos_edge_index = children_pos_e_reversed, neg_edge_index = children_neg_e_reversed)))


      all_concat = torch.cat((bert_parents, bert_children, parents_conv3, children_conv3), 1)
      # print(F.cosine_similarity(parents_conv1, children_conv1))
      x = self.lin1(all_concat) # linear 1
      # x = self.dropout(F.relu(x)) # relu then dropout
      # x = self.lin2(x)
      # pred = F.log_softmax(x, dim = 1) # no softmax needed with torch CrossEntropy loss
      return x
