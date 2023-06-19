from sentence_transformers import SentenceTransformer
from scipy import spatial
import numpy as np
import spacy
import logging
import torch
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import spacy
import logging
import re
from nltk.stem import WordNetLemmatizer
from dataset import SignedBipartiteData
import numpy as np
from scipy import spatial
from ast import literal_eval
import torch
from torch_scatter import scatter_add
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
lemmatizer = WordNetLemmatizer()

# function to get pro/con cosine similarity between post embedding and entity
def get_cos_sim(sbert_model, entity, embed, embeddings_dict):
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
def get_pos_neg_edges(df):
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    embeddings_dict = {}
    nlp = spacy.load("en_core_web_md")
    excluded = ['CARDINAL', 'DATE', 'ORDINAL', 'WORK_OF_ART', 'PERCENT', 'QUANTITY', 'MONEY' ,'FAC', 'TIME', 'LANGUAGE',
              'PRODUCT']
    users_dict = {u: {} for u in set(df['author_parent'].unique()).union(set(df['author_child'].unique()))}
    u = len(users_dict)

    logging.info(f'Number of users: {u}')

    empty_parents = 0
    empty_children = 0

    # get posts
    for i in range(len(df)):
      print(i)
      # get entities
      parent = df['author_parent'].iloc[i]
      parent_doc = nlp(df['body_parent'].iloc[i])
      parent_ents = set([re.sub(r'[^\w\s]', '', e.text).lower() for e in parent_doc.ents if e.label_ not in excluded])

      child = df['author_child'].iloc[i]
      child_doc = nlp(df['body_child'].iloc[i])
      child_ents = set([re.sub(r'[^\w\s]', '', e.text).lower() for e in child_doc.ents if e.label_ not in excluded])

      # get embeddings
      if parent_ents != set():
        # parent_embed = np.mean(sbert_model.encode([s.text for s in list(parent_doc.sents)]), axis = 0)
        parent_embed = sbert_model.encode([s.text for s in list(parent_doc.sents)])
        if len(parent_embed) == 0:
          logging.error('Empty sentences parent')
      else:
        empty_parents+=1
      if child_ents != set():
        # child_embed = np.mean(sbert_model.encode([s.text for s in list(child_doc.sents)]), axis = 0)
        child_embed = sbert_model.encode([s.text for s in list(child_doc.sents)])
        if len(parent_embed) == 0:
          logging.error('Empty sentences child')
      else:
        empty_children+=1

      # parent entities cos sim
      for e in list(parent_ents):
        c = []
        for embed in parent_embed:
          cos_diff = get_cos_sim(sbert_model, e, embed, embeddings_dict)
          if not isinstance(cos_diff, np.floating):
            logging.error('cosine diff is not float')
          c.append(cos_diff)
        cos = np.nanmean(c)
        e = lemmatizer.lemmatize(e)
        if e in users_dict[parent]:
          users_dict[parent][e].append(cos)
        else:
          users_dict[parent][e] = [cos]

      # children entities cos sim 
      for e in list(child_ents):
        c = []
        for embed in child_embed:
          cos_diff = get_cos_sim(sbert_model, e, embed, embeddings_dict)
          if not isinstance(cos_diff, np.floating):
            logging.error('cosine diff is not float')
          c.append(cos_diff)
        cos = np.nanmean(c)
        e = lemmatizer.lemmatize(e)
        if e in users_dict[child]:
          users_dict[child][e].append(cos)
        else:
          users_dict[child][e] = [cos]

    pos_edges = {}
    neg_edges = {}
    all_cos= []

    # get mean cosine diff (it's heavily biased towards negative!)
    for u in users_dict:
      for e in users_dict[u]:
        all_cos.extend(users_dict[u][e])

    mean_cos_diff = np.nanmean(all_cos)
    logging.info(f'Mean cosine diff : {mean_cos_diff}')

    for u in users_dict:
      for e in users_dict[u]:
        # put average of cosine sims with entity in pos/neg edges dict
        mean_cos = np.nanmean(users_dict[u][e])
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

    pos_l = sum(len(pos_edges[u]) for u in pos_edges) 
    neg_l = sum(len(neg_edges[u]) for u in neg_edges) 
    logging.info(f'Extracted {pos_l} positive edges and {neg_l} negative edges')

    return pos_edges, neg_edges 


# function to create data list to feed to PyG dataloader using signed bipartite data class
# ! try to turn this into a generator and see if dataloader takes output?
def create_graph_data_lists(df, embeddings_dict, embed_size, pos_edges, neg_edges, edge_weights=True):
  logging.info(f'Using embeddings with size {embed_size}')
  pos_l = sum(len(pos_edges[u]) for u in pos_edges) 
  neg_l = sum(len(neg_edges[u]) for u in neg_edges)
  users = set(list(pos_edges.keys()) + list(neg_edges.keys()))
  ents = [list(pos_edges[u].keys()) for u in pos_edges] + [list(neg_edges[u].keys()) for u in neg_edges]
  entities = sorted(list(set([i for sub in ents for i in sub])))
  print('ALL ENTITIES', entities)

  logging.info(f'Processing {len(users)} users')
  logging.info(f'Processing {len(entities)} entities')
  logging.info(f'Processing {pos_l} positive edges and {neg_l} negative edges')

  # get data
  parents_data_list = []
  children_data_list = []

  # get word2vec embeddings as entity features (! issues with keys are due to spacy vs nltk tokenization)
  voc_dict = {}
  errors = []

  for idx, e in enumerate(entities):
    # voc_dict[e] = np.mean(pipeline(e)[0][1:-1], axis = 0)
    original_e = e
    if '/' in e:
      e = ' '.join(e.split('/'))
    else:
      e = re.sub(r'[^\w\s]', '', e).lstrip() # this leaves spaces inside entity
    if e != '':
      try: 
        if len(e.split()) == 1:
          embed =  embeddings_dict[e]
        else:
          # average embeddings when multiword entity
          words = [lemmatizer.lemmatize(re.sub(r'[^\w\s]', '', i.lower())) for i in e.split()]
          mw = np.zeros((len(words), embed_size))
          for idx, w in enumerate(words):
            mw[idx, :] = embeddings_dict[w]
          embed = np.mean(mw, axis = 0)
        voc_dict[e] = embed 
      except Exception:
        logging.error(f'Could not get embedding for:{original_e},{e}')
        errors.append(e)
    else:
      logging.error(f'Empty string for: {original_e}{e}')
      errors.append(e)
  
  # create neutral embedding
  l = len(voc_dict)
  m = np.zeros((l, embed_size))
  for idx, e in enumerate(list(voc_dict.keys())):
    m[idx, :] = voc_dict[e]
  neutral_embed = np.mean(m, axis = 0)
  voc_dict['NEUTRAL_ENTITY'] = neutral_embed
  logging.info(f'Neutral embedding shape : {neutral_embed.shape}')

  # give neutral embeddings to errors
  print('ERRORS:', errors)
  for e in errors:
    voc_dict[e] = neutral_embed

  parent_ids = list(df.author_parent)
  children_ids = list(df.author_child)
  parents_ents = list(df.parent_ents)
  children_ents = list(df.child_ents)

  for idx in range(len(df)): # for each data point
    # parent_post = df['body_parent'].iloc[idx]
    # parent_words = [re.sub(r'[^\w]', '', i.lower()) for i in parent_post.split()]
    # child_post = df['body_child'].iloc[idx]
    # child_words = [re.sub(r'[^\w]', '', i.lower()) for i in child_post.split()]
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
        parent_entities.extend(sorted(list(edge_list[parent_id].keys())))
    for edge_list in [pos_edges, neg_edges]:
      # get all entities for child
      if child_id in edge_list:
        child_entities.extend(sorted(list(edge_list[child_id].keys())))

    parent_user_feat = torch.zeros((1, embed_size)) # initialize user and entity feature matrices 
    child_user_feat = torch.zeros((1, embed_size))
    parent_entities_feat = torch.zeros((len(parent_entities), embed_size))
    child_entities_feat = torch.zeros((len(child_entities), embed_size))
    
    for i, e in enumerate(parent_entities):
      # for idx of entity in list, append entity features to list 
      if '/' in e:
        e = ' '.join(e.split('/'))
      else:
        e = re.sub(r'[^\w\s]', '', e).lstrip()
      parent_entities_feat[i, :] = torch.tensor(voc_dict[e])

      if parent_id in pos_edges and e in pos_edges[parent_id]:
        # for each parent/child, there is only one index of user (0)
        parent_pos_edges[0].append(i) # source (= entity) edges first
        parent_pos_edges[1].append(0)
        parent_pos_weight.append(pos_edges[parent_id][e])
      if parent_id in neg_edges and e in neg_edges[parent_id]:
        parent_neg_edges[0].append(i)
        parent_neg_edges[1].append(0)
        parent_neg_weight.append(neg_edges[parent_id][e])
        

    for i, e in enumerate(child_entities):
      # for idx of entity in list, append entity features to list 
      if '/' in e:
        e = ' '.join(e.split('/'))
      else:
        e = re.sub(r'[^\w\s]', '', e).lstrip()
      child_entities_feat[i, :] = torch.tensor(voc_dict[e])

      if child_id in pos_edges and e in pos_edges[child_id]:
        child_pos_edges[0].append(i)
        child_pos_edges[1].append(0)
        child_pos_weight.append(pos_edges[child_id][e])
      if child_id in neg_edges and e in neg_edges[child_id]:
        child_neg_edges[0].append(i)
        child_neg_edges[1].append(0)
        child_neg_weight.append(neg_edges[child_id][e])

    if edge_weights == True:
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
    else:
      parent_graph = SignedBipartiteData(x_s = parent_entities_feat, x_t = parent_user_feat,
                                          pos_edge_index = torch.tensor(parent_pos_edges, dtype=torch.int64), 
                                          neg_edge_index = torch.tensor(parent_neg_edges, dtype=torch.int64),
                                          pos_edge_weight = torch.ones((len(parent_pos_edges))),
                                          neg_edge_weight =  torch.ones((len(parent_neg_edges))))
      child_graph = SignedBipartiteData(x_s = child_entities_feat, x_t = child_user_feat,
                                      pos_edge_index = torch.tensor(child_pos_edges, dtype=torch.int64), 
                                      neg_edge_index = torch.tensor(child_neg_edges, dtype=torch.int64),
                                      pos_edge_weight = torch.ones((len(child_pos_edges))),
                                      neg_edge_weight = torch.ones((len(child_pos_edges))))

    parents_data_list.append(parent_graph)
    children_data_list.append(child_graph)

    return parents_data_list, children_data_list


def mean_pool(bert_model, ids, masks, segs):
    n_tokens_pad = (ids >= 106).float().sum(dim=-1)
    output_bert = bert_model(ids, attention_mask=masks, token_type_ids=segs)[0]
    output_bert_pad = output_bert * (ids >= 106).float().unsqueeze(-1).expand(-1, -1, 768)
    output_bert_pooled = output_bert_pad.sum(dim=1) / n_tokens_pad.unsqueeze(-1).expand(-1, 768)
    return output_bert_pooled


def gcn_norm(edge_index, edge_weight=None, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    idx = col
    deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def get_top_entities_by_freq(pos_edges, neg_edges, threshold):
    d = {}
    for u in pos_edges:
        for e in pos_edges[u]:
            if e in d:
                d[e] +=1
            else:
                d[e] = 1
    for u in neg_edges:
        for e in neg_edges[u]:
            if e in d:
                d[e] +=1
            else:
                d[e] = 1
    ents = [k for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)][:threshold]
    ents = [e for e in ents if len(e.split())==1]
    return ents


def filter_ents(edges, ents):
    edges_filtered = {}
    for u in sorted(list(edges.keys())):
        new_u = {e: cos for e, cos in edges[u].items() if e in ents}
        if new_u != {}:
            edges_filtered[u] = new_u
    return edges_filtered


def normalise_edges(pos_edges, neg_edges):
    print('ORIG EDGES', len(pos_edges), len(neg_edges))
    neg_val = [neg_edges[u][e] for u in neg_edges for e in neg_edges[u]]
    pos_val = [pos_edges[u][e] for u in pos_edges for e in pos_edges[u]]
    all_val = neg_val + pos_val
    m = np.nanmean(all_val)
    print(f'mean of edge weights is {m}')
    neg_edges_filtered = {}
    pos_edges_filtered = {}
    
    # mean centering and swapping 'fake' negative edges back to pos edges 
    for u in sorted(list(neg_edges.keys())): # all values end up positive 
        new_u_neg = {e: np.abs(cos-m) for e, cos in neg_edges[u].items() if cos <= m} # real negatives
        new_u_pos = {e: np.abs(cos-m) for e, cos in neg_edges[u].items() if cos > m} # fake negatives 
        if new_u_neg != {}:
            neg_edges_filtered[u] = new_u_neg
        if u in pos_edges:
            old_u_pos = {e: np.abs(cos-m)  for e, cos in pos_edges[u].items()}
            new_u_pos.update(old_u_pos) # merging the two dict
            pos_edges_filtered[u] = new_u_pos
    for u in sorted(list(pos_edges.keys())):
        if u not in pos_edges_filtered:
            pos_edges_filtered[u] =  {e: np.abs(cos-m) for e, cos in pos_edges[u].items()}
    return pos_edges_filtered, neg_edges_filtered

def filter_by_sim(word2vec, ents, sim):
    filtered = set()
    for e_orig in ents:
        for t in ['brexit', 'blm', 'climate', 'republican', 'democrat']:
            try:
              if len(e_orig.split()) == 1:
                e = re.sub(r'[^\w]', '', e_orig)
                cos = 1 - spatial.distance.cosine(word2vec.wv[e], word2vec.wv[t])
              else:
                e = re.sub(r'[^\w\s]', ' ', e_orig)
                cos = np.mean([1 - spatial.distance.cosine(word2vec.wv[lemmatizer.lemmatize(w)], word2vec.wv[t]) for w in e.split()])
              if cos > sim:
                filtered.add(e)
            except Exception as exp:
              continue
    return list(filtered)

def preprocess_edges_and_datasets(word2vec, df_train, df_valid, train_pos_edges, train_neg_edges, 
                                  eval_pos_edges, eval_neg_edges, threshold, sim,
                                  lemmatise,
                                  filter_ents_dataset,
                                  full_agreement):
  
  # get entities and stats
  ents = get_top_entities_by_freq(train_pos_edges, train_neg_edges, threshold=threshold)
  ents = filter_by_sim(word2vec, ents, sim=sim)
  print(f'number of target entities: {len(ents)}')

  # normalise edges
  train_pos_edges, train_neg_edges = normalise_edges(train_pos_edges, train_neg_edges)
  eval_pos_edges, eval_neg_edges = normalise_edges(eval_pos_edges, eval_neg_edges)

  # filter entities
  train_pos_edges = filter_ents(train_pos_edges, ents) 
  train_neg_edges = filter_ents(train_neg_edges, ents)
  eval_pos_edges = filter_ents(eval_pos_edges, ents)
  eval_neg_edges = filter_ents(eval_neg_edges, ents)

  # ents for dataset filtering by string
  ents_dataset = [f"'{e}'" for e in ents]
  print(sorted(ents_dataset))

  def to_set(x):
    if x!= 'set()':
      return literal_eval(x)
    else:
      return set()

  if lemmatise==True:
    df_train['parent_ents']=df_train['parent_ents'].apply(lambda x: to_set(x))
    df_train['child_ents']=df_train['child_ents'].apply(lambda x: to_set(x))
    df_train['parent_ents']=df_train['parent_ents'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    df_train['child_ents']=df_train['child_ents'].apply(lambda x:  [lemmatizer.lemmatize(i) for i in x])
    df_train['parent_ents']=df_train['parent_ents'].apply(lambda x: str(x))
    df_train['child_ents']=df_train['child_ents'].apply(lambda x: str(x))

    df_valid['parent_ents']=df_valid['parent_ents'].apply(lambda x: to_set(x))
    df_valid['child_ents']=df_valid['child_ents'].apply(lambda x: to_set(x))
    df_valid['parent_ents']=df_valid['parent_ents'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    df_valid['child_ents']=df_valid['child_ents'].apply(lambda x:  [lemmatizer.lemmatize(i) for i in x])
    df_valid['parent_ents']=df_valid['parent_ents'].apply(lambda x: str(x))
    df_valid['child_ents']=df_valid['child_ents'].apply(lambda x: str(x))

  # # subset train and  validation dataset with target entities
  users_train = set(list(train_pos_edges.keys()) + list(train_neg_edges.keys()))
  users_eval = set(list(eval_pos_edges.keys()) + list(eval_neg_edges.keys()))

  if filter_ents_dataset == True:
    df_train_sub = df_train[(df_train['child_ents'].str.contains('|'.join(ents_dataset)))&(df_train['parent_ents'].str.contains('|'.join(ents_dataset)))].reset_index(drop=True)
    df_valid_sub = df_valid[(df_valid['child_ents'].str.contains('|'.join(ents_dataset)))&(df_valid['parent_ents'].str.contains('|'.join(ents_dataset)))].reset_index(drop=True)
  else:
    df_train_sub = df_train[(df_train['author_parent'].isin(users_train))&(df_train['author_child'].isin(users_train))]
    df_valid_sub = df_valid[(df_valid['author_parent'].isin(users_eval))&(df_valid['author_child'].isin(users_eval))]
  
  if full_agreement == True:
    df_train_sub = df_train_sub[df_train_sub['agreement_fraction'] == 1].reset_index(drop=True)
    df_valid_sub = df_valid_sub[df_valid_sub['agreement_fraction'] ==1].reset_index(drop=True)

  # compute class weights
  y = df_train_sub['label'].values
  y2 = df_valid_sub['label'].values
  class_weights=compute_class_weight('balanced', classes = [0, 1, 2] , y = y)
  class_weights2=compute_class_weight('balanced', classes = [0, 1, 2] , y = y2)
  print('WEIGHTS_train', class_weights)
  print('WEIGHTS_test', class_weights2)
  class_weights=torch.tensor(class_weights,dtype=torch.float)
  return df_train_sub, df_valid_sub, train_pos_edges, train_neg_edges, eval_pos_edges, eval_neg_edges, class_weights

def multi_acc(y_pred, y_true):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

