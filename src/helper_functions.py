# !pip install -U sentence-transformers
# !python -m spacy download en_core_web_md
from sentence_transformers import SentenceTransformer
from scipy import spatial
import numpy as np
import spacy
import logging
import torch
from gensim.models import Word2Vec

# !python -m spacy download en_core_web_md

nlp = spacy.load("en_core_web_md")
word2vec = Word2Vec.load(word2vec_model_path)
sbert_model = SentenceTransformer('all-mpnet-base-v2')


# function to get pro/con cosine similarity between post embedding and entity
def get_cos_sim(entity, embed, embeddings_dict):
        # create pro/con sentences
        e_cap = entity.capitalize()
        pro = 'I am for ' + e_cap + '.'
        con = 'I am against ' + e_cap + '.'

        # encode pro/con sentences
        if pro in embeddings_dict:
          e_for = embeddings_dict[pro]
        else:
          e_for = sbert_model.encode(pro)
          embeddings_dict[pro] = e_for

        if con in embeddings_dict:
          e_con = embeddings_dict[con]
        else:
          e_con = sbert_model.encode(con)
          embeddings_dict[con] = e_con

        # get cosine sim with pro, cosine sim with con
        cos_for = 1 - spatial.distance.cosine(embed, e_for)
        cos_con = 1 - spatial.distance.cosine(embed, e_con)
        # get difference between pro and con cosine sims
        cos = cos_for - cos_con

        return cos

# function to create a graph of users and their pro/con cosine sim with entities in their posts
def get_pos_neg_edges(df, threshold):

  nlp = spacy.load("en_core_web_md")
  excluded = ['CARDINAL', 'DATE', 'ORDINAL', 'WORK_OF_ART', 'PERCENT', 'QUANTITY', 'MONEY']
  # df = pd.read_csv(df_path)
  e_counts = {}
  users_dict = {u: {} for u in set(df['author_parent'].unique()).union(set(df['author_child'].unique()))}
  embeddings_dict = {}

  # get posts
  for i in range(len(df)):
    print(i)
    parent = df['author_parent'].iloc[i]
 
    parent_doc = nlp(df['body_parent'].iloc[i])
    # get entities
    parent_ents = set([e.text.lower() for e in parent_doc.ents if e.label_ not in excluded])

    child = df['author_child'].iloc[i]
    child_doc = nlp(df['body_child'].iloc[i])
    child_ents = set([e.text.lower() for e in child_doc.ents if e.label_ not in excluded])
    # get embeddings
    if parent_ents != set():
      parent_embed = np.mean(sbert_model.encode([s.text for s in list(parent_doc.sents)]), axis = 0)
    if child_ents != set():
      child_embed = np.mean(sbert_model.encode([s.text for s in list(child_doc.sents)]), axis = 0)

    # get entity counts
    for e in parent_ents.union(child_ents):
      if e not in e_counts:
        e_counts[e] = 1
      else:
        e_counts[e] += 1

    # parent entities cos sim
    for e in list(parent_ents):
      cos = get_cos_sim(e, parent_embed, embeddings_dict)
      if e in users_dict[parent]:
        users_dict[parent][e].append(cos)
      else:
        users_dict[parent][e] = [cos]

    # children entities cos sim 
    for e in list(child_ents):
      cos = get_cos_sim(e, child_embed, embeddings_dict)
      if e in users_dict[child]:
        users_dict[child][e].append(cos)
      else:
        users_dict[child][e] = [cos]

  pos_edges = {}
  neg_edges = {}
  cos_diff = []

  # get mean cosine diff (it's heavily biased towards negative)
  for u in users_dict:
    for e in users_dict[u]:
        cos_diff.append(users_dict[u][e])

  mean_cos_diff = np.mean(cos_diff)

  for u in users_dict:
    for e in users_dict[u]:
      if e_counts[e] > threshold:
        # put average of cosine sims with entity in pos/neg edges dict
        mean_cos = np.mean(users_dict[u][e])
        if mean_cos > mean_cos_diff:
          if u in pos_edges:
            pos_edges[u][e] = mean_cos 
          else:
            pos_edges[u] = {e: mean_cos}
        # neg edges
        else:
          if u in neg_edges:
            neg_edges[u][e] = mean_cos
          else:
            neg_edges[u] =  {e: mean_cos}

      # if u not in pos_edges and u not in neg_edges:
      #   # create positive edge to neutral entity for isolated nodes
      #   pos_edges[u] = {'NEUTRAL_ENTITY': 0} # maybe just add on one side otherwise edge issue

  return pos_edges, neg_edges  


# function to create data list to feed to PyG dataloader using signed bipartite data class
# ! try to turn this into a generator and see if dataloader takes output
def create_graph_data_lists(df, word2vec, pos_edges, neg_edges):
    # get data
    parents_data_list = []
    children_data_list = []

    # get all entities
    ents = [list(pos_edges[u].keys()) for u in pos_edges] + [list(neg_edges[u].keys()) for u in neg_edges]
    entities = list(set([i for sub in ents for i in sub]))

     # get word2vec embeddings as entity features (! issues with keys are due to spacy vs nltk tokenization)
    word2vec_dict = {}
    errors = []
    for idx, e in enumerate(entities):
      original_e = e

      if e != 'NEUTRAL_ENTITY':
        if '/' in e:
          e = ' '.join(e.split('/'))
        else:
          e = re.sub(r'[^\w\s]', '', e).lstrip()
        if e != '':
          try: 
            if len(e.split()) == 1:
              embed =  word2vec.wv[e]
            else:
              # average embeddings when multiword entity
              words =  [re.sub(r'[^\w]\s', '', i.lower()) for i in e.split()]
              mw = np.zeros((len(words), 100))
              for idx, w in enumerate(words):
                mw[idx, :] = word2vec.wv[w]
              embed = np.mean(mw, axis = 0)
            word2vec_dict[e] = embed 
          except Exception:
            logging.error('could not get vector for:'original_e, e)
            errors.append(e)
        else:
          logging.error('could not get vector for:'original_e, e)
          errors.append(e)
    
    # create neutral embedding for isolated nodes
    l = len(word2vec_dict)
    m = np.zeros((l, 100))
    for idx, e in enumerate(list(word2vec_dict.keys())):
      m[idx, :] = word2vec_dict[e]
    neutral_embed = np.mean(m, axis = 0)
    word2vec_dict['NEUTRAL_ENTITY'] = neutral_embed

    # also give neutral embeddings to errors
    for e in errors:
      word2vec_dict[e] = neutral_embed

    parent_ids = [i for i in list(df.author_parent)]
    children_ids = [i for i in list(df.author_child)]
    # parents_ents = [i for i in list(df.parent_ents)]
    # children_ents = [i for i in list(df.child_ents)]

    for idx in range(len(df)): # for each data point
      parent_id = parent_ids[idx]
      child_id = children_ids[idx]
      parent_pos_edges = [[], []]
      parent_pos_weight =  []
      parent_neg_edges = [[], []]
      parent_neg_weight =  []
      child_pos_edges = [[], []]
      child_pos_weight =  []
      child_neg_edges = [[], []]
      child_neg_weight =  []
      # parent_post_ents = parents_ents[idx]
      # child_post_ents = children_ents[idx]
        
      # ids with no edges get neutral entity embedding (in pos edges because can't have an entity in both)
      parent_entities = []
      child_entities = []

      for edge_list in [pos_edges, neg_edges]:
        # get all entities for a parent
        if parent_id in edge_list:
          # parent_entities.extend([e for e in list(edge_list[parent_id].keys()) if e in parent_post_ents]) # no need for set because can't have same entity
          parent_entities.extend(list(edge_list[parent_id].keys())) 
      for edge_list in [pos_edges, neg_edges]:
        # get all entities for child
        if child_id in edge_list:
          # child_entities.extend([e for e in list(edge_list[child_id].keys())if e in child_post_ents])
          child_entities.extend(list(edge_list[child_id].keys()))

      parent_user_feat = torch.zeros((1, 100)) # initialize user and entity feature matrices 
      child_user_feat = torch.zeros((1, 100))
      parent_entities_feat = torch.zeros((len(parent_entities), 100))
      child_entities_feat = torch.zeros((len(child_entities), 100))
      
      for i, e in enumerate(parent_entities):
        # for idx of entity in list, append entity features to list 
        if '/' in e:
          e = ' '.join(e.split('/'))
        else:
          e = re.sub(r'[^\w\s]', '', e).lstrip()
        parent_entities_feat[i, :] = torch.tensor(word2vec_dict[e])

        if parent_id in pos_edges and e in pos_edges[parent_id]:
          # for each parent/child, there is only one index of user (0)
          parent_pos_edges[0].append(i) # source (= entity) edges first
          parent_pos_edges[1].append(0)
          parent_pos_weight.append(np.abs(pos_edges[parent_id][e]))

        if parent_id in neg_edges and e in neg_edges[parent_id]:
          parent_neg_edges[0].append(i)
          parent_neg_edges[1].append(0)
          parent_neg_weight.append(np.abs(neg_edges[parent_id][e]))

      for i, e in enumerate(child_entities):
        # for idx of entity in list, append entity features to list 
        if '/' in e:
          e = ' '.join(e.split('/'))
        else:
          e = re.sub(r'[^\w\s]', '', e).lstrip()
        child_entities_feat[i, :] = torch.tensor(word2vec_dict[e])

        if child_id in pos_edges and e in pos_edges[child_id]:
          child_pos_edges[0].append(i)
          child_pos_edges[1].append(0)
          child_pos_weight.append(np.abs(pos_edges[child_id][e]))

        if child_id in neg_edges and e in neg_edges[child_id]:
          child_neg_edges[0].append(i)
          child_neg_edges[1].append(0)
          child_neg_weight.append(np.abs(neg_edges[child_id][e]))

      parent_graph = SignedBipartiteData(x_s = parent_entities_feat, x_t = parent_user_feat,
                                          pos_edge_index = torch.tensor(parent_pos_edges, dtype=torch.int64), 
                                          neg_edge_index = torch.tensor(parent_neg_edges, dtype=torch.int64),
                                          pos_edge_weight = torch.tensor(parent_pos_weight),
                                          neg_edge_weight = torch.tensor(parent_neg_weight))
      child_graph = SignedBipartiteData(x_s = child_entities_feat, x_t = child_user_feat,
                                      pos_edge_index = torch.tensor(child_pos_edges, dtype=torch.int64), 
                                      neg_edge_index = torch.tensor(child_neg_edges, dtype=torch.int64),
                                      pos_edge_weight = torch.tensor(child_pos_weight),
                                      neg_edge_weight = torch.tensor(child_neg_weight))

      parents_data_list.append(parent_graph)
      children_data_list.append(child_graph)

    return parents_data_list, children_data_list


def mean_pool(bert_model, ids, masks, segs):
  n_tokens_pad = (ids >= 106).float().sum(dim=-1)
  output_bert = bert_model(ids, attention_mask=masks, token_type_ids=segs)[0]
  output_bert_pad = output_bert * (ids >= 106).float().unsqueeze(-1).expand(-1, -1, 768)
  output_bert_pooled = output_bert_pad.sum(dim=1) / n_tokens_pad.unsqueeze(-1).expand(-1, 768)
  return output_bert_pooled

def multi_acc(y_pred, y_true):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc