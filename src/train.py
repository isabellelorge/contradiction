import argparse
import logging
import random
import time
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
import numpy as np
from models import STEntConv
import pandas as pd
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGloader

from dataset import BertDataset, collate 
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from datetime import datetime
import os
import json

from helper_functions import create_graph_data_lists, multi_acc, preprocess_edges_and_datasets
from models import STEntConv


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory.')
    parser.add_argument('--trained_path', default=None, type=str, required=True, help='Path for trained model.')
    parser.add_argument('--random_seed', default=123, type=int, required=True, help='Random seed.')
    parser.add_argument('--batch_size', default=16, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=0.000003, type=float, required=True, help='Learning rate.')
    parser.add_argument('--decay', default=1e-5, type=float, required=True, help='Weight decay.')
    parser.add_argument('--n_epochs', default=6, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--dropout1', default=0.1, type=float, required=True, help='Dropout rate 1.')
    parser.add_argument('--dropout2', default=0.8, type=float, required=True, help='Dropout rate 2.')
    parser.add_argument('--hidden_size)', default=300, type=int, required=True, help='Size of hidden layer GCN.')
    parser.add_argument('--bert_path', default='bert-base-cased', type=str, required=True, help='Path of BERT model.')
    parser.add_argument('--word2vec_path', default ='../data/word2vec_lem.model', type=str, required=True, help='Path of word2vec model.')
    parser.add_argument('--subreddit_sim', default=0.5, type=float, required=True, help='Entities similarity threshold subreddits.')
    parser.add_argument('--freq_threshold', default=5000, type=int, required=True, help='Frequency threshold entities.')
    parser.add_argument('--lemmatise', default=True, type=bool, required=True, help='lemmatise entities.')
    parser.add_argument('--model_name', default='all_layers', type=float, required=True, help='model layers to use: all_layers, bert_only, GCN_only')
    parser.add_argument('--full_agreement', default=True, type=float, required=True, help='filter by crowdsourcers agreement on labels.')

    args = parser.parse_args()

    word2vec = Word2Vec.load(args.wor2vec_path)
    random_seed = args.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    dropout_rate1 = args.dropout1
    dropout_rate2 = args.dropout2
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    bert_path = args.bert_path
    weight_decay = args.decay
    lr = args.lr
    n_epochs = args.epochs
    sim = args.sim
    threshold = args.freq_threshold
    lemmatise = True
    filter_ents_dataset = True
    full_agreement = True
    model_name = args.model_name

    logging.info('Load training data...')
    df_train = pd.read_csv('{}/train.csv'.format(args.data_dir))

    logging.info('Load eval data...')
    df_valid = pd.read_csv('{}/dev.csv'.format(args.data_dir))

    with open('{}/train_pos_edges.json'.format(args.data_dir), 'r') as f:
        train_pos_edges = json.load(f)
    with open('{}/train_neg_edges.json'.format(args.data_dir), 'r') as f:
        train_neg_edges = json.load(f)

    with open('{}/dev_pos_edges.json'.format(args.data_dir), 'r') as f:
        eval_pos_edges = json.load(f)
    with open('{}/dev_neg_edges.json'.format(args.data_dir), 'r') as f:
        eval_neg_edges = json.load(f)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # torch.cuda.empty_cache()
    print('SEED', random_seed)
    print('MODEL:', model_name)
    print('SIMILARITY:', sim)
    print('THRESHOLD FREQUENCY:', threshold) # ! need to not use same edge names since modifying them! CAREFUL: this modifies df_train and df_valid

    df_train_sub, df_valid_sub, train_pos_edges_sub, train_neg_edges_sub, eval_pos_edges_sub, eval_neg_edges_sub, class_weights = preprocess_edges_and_datasets(word2vec, df_train, df_valid, 
                                                                                                                train_pos_edges, train_neg_edges, eval_pos_edges, eval_neg_edges, 
                                                                                                                threshold=threshold, sim=sim,
                                                                                                                lemmatise=lemmatise,
                                                                                                                filter_ents_dataset=filter_ents_dataset,
                                                                                                                full_agreement=full_agreement)
  
    class_weights = class_weights.to(device)
    model = STEntConv(dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, 
                    hidden_size=hidden_size, bert_path=bert_path, model_name=model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print('Train classifier...')
    for epoch in range(1, n_epochs + 1):
        print(f'Training epoch {epoch}')
        random_state = epoch
        print(f'Random state for shuffle: {random_state}')

        # shuffle train dataset (different random state each epoch)
        # df_train_sub = preprocess(df_train, train_pos_edges, train_neg_edged, threshold=100)
        df_train_shuffled = df_train_sub.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(df_train_shuffled['body_parent'].head())
        print(f'train dataset length: {len(df_train_shuffled)}')
        print(df_valid_sub['body_parent'].head())
        print(f'valid dataset length: {len(df_valid_sub)}')

        # BERT datasets
        train_dataset = BertDataset(df_train_shuffled, bert_path)
        eval_dataset = BertDataset(df_valid_sub, bert_path)

        # Graph datasets
        train_parents_entities_list, train_children_entities_list = create_graph_data_lists(df=df_train_shuffled, embeddings_type='word2vec', 
                                                        embed_size=100, pos_edges=train_pos_edges_sub, neg_edges=train_neg_edges_sub,edge_weights=True)
        

        eval_parents_entities_list, eval_children_entities_list = create_graph_data_lists(df=df_valid_sub, embeddings_type='word2vec', 
                                                            embed_size=100, pos_edges=eval_pos_edges_sub, neg_edges=eval_neg_edges_sub, edge_weights=True)
        # load and batch train
        train_text_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate)
        train_parents_dataloader = PyGloader(train_parents_entities_list, batch_size=batch_size)
        train_children_dataloader = PyGloader(train_children_entities_list, batch_size=batch_size)

        # load and batch valid
        eval_text_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate) 
        eval_parents_dataloader = PyGloader(eval_parents_entities_list, batch_size=batch_size)
        eval_children_dataloader = PyGloader(eval_children_entities_list, batch_size=batch_size)

        model.train()
        train_loss = 0
        train_acc = 0
        batch_idx = 0

        # train on batch
        for text_batch, parents_batch, children_batch in zip(train_text_dataloader, train_parents_dataloader, train_children_dataloader):
            if batch_idx % 50 == 0 and batch_idx != 0:
                print(f'Processed {(batch_idx * batch_size)} examples...')

            batch_idx += 1

            # get batch data for text and graphs
            labels, parents, parents_masks, parents_segs, children, children_masks, children_segs = text_batch
            parents_pos_edges, parents_neg_edges, parents_pos_weight, parents_neg_weight, parents_entities_feat, parents_user_feat = parents_batch
            children_pos_edges, children_neg_edges, children_pos_weight, children_neg_weight, children_entities_feat, children_user_feat = children_batch

            # to device
            labels = labels.type(torch.LongTensor) 
            labels = labels.to(device)
            parents = parents.to(device)
            parents_masks = parents_masks.to(device)
            parents_segs = parents_segs.to(device)
            children = children.to(device)
            children_masks = children_masks.to(device)
            children_segs = children_segs.to(device)
            parents_user_feat =  parents_user_feat[1].to(device)
            children_user_feat = children_user_feat[1].to(device)
            parents_entities_feat = parents_entities_feat[1].to(device)
            children_entities_feat = children_entities_feat[1].to(device)
            parents_pos_edges =  parents_pos_edges[1].to(device)
            children_pos_edges = children_pos_edges[1].to(device)
            parents_neg_edges = parents_neg_edges[1].to(device)
            children_neg_edges = children_neg_edges[1].to(device)
            parents_pos_weight = parents_pos_weight[1].to(device)
            parents_neg_weight = parents_neg_weight[1].to(device)
            children_pos_weight = children_pos_weight[1].to(device)
            children_neg_weight = children_neg_weight[1].to(device)

            # zero grad
            optimizer.zero_grad()

            # run model
            output = model(parents=parents, parents_masks=parents_masks, parents_segs=parents_segs, 
                        children=children, children_masks=children_masks, children_segs=children_segs, 
                        parents_user_feat=parents_user_feat, parents_entities_feat=parents_entities_feat, 
                        children_user_feat=children_user_feat, children_entities_feat=children_entities_feat, 
                        parents_pos_e=parents_pos_edges, parents_neg_e=parents_neg_edges,
                        children_pos_e=children_pos_edges, children_neg_e=children_neg_edges,
                        parents_pos_weight=parents_pos_weight, parents_neg_weight=parents_neg_weight,
                        children_pos_weight=children_pos_weight, children_neg_weight=children_neg_weight)

            # current metrics
            if batch_idx % 50 == 0 and batch_idx != 0:
                print('LOSS:', train_loss/batch_idx)
                print('ACC:', train_acc/batch_idx)
                print(labels)
                print(torch.argmax(output, dim=1))
            loss = criterion(output, labels)
            acc = multi_acc(output, labels)
            train_loss += loss.item()
            train_acc += acc.item()

            # backward and step
            loss.backward()
            optimizer.step()

            # evaluate after each epoch
            print(f'Evaluate classifier epoch: {epoch}..')
            model.eval()

            y_true = list()
            y_pred = list()

            with torch.no_grad():
                # get valid data
                for text_batch, parents_batch, children_batch in zip(eval_text_dataloader, eval_parents_dataloader, eval_children_dataloader):
                    labels, parents, parents_masks, parents_segs, children, children_masks, children_segs = text_batch
                    parents_pos_edges, parents_neg_edges, parents_pos_weight, parents_neg_weight, parents_entities_feat, parents_user_feat = parents_batch
                    children_pos_edges, children_neg_edges,children_pos_weight, children_neg_weight, children_entities_feat, children_user_feat = children_batch

                    # to device
                    labels = labels.type(torch.LongTensor) 
                    labels = labels.to(device)
                    parents = parents.to(device)
                    parents_masks = parents_masks.to(device)
                    parents_segs = parents_segs.to(device)
                    children = children.to(device)
                    children_masks = children_masks.to(device)
                    children_segs = children_segs.to(device)
                    parents_user_feat =  parents_user_feat[1].to(device)
                    children_user_feat = children_user_feat[1].to(device)
                    parents_entities_feat = parents_entities_feat[1].to(device)
                    children_entities_feat = children_entities_feat[1].to(device)
                    parents_pos_edges =  parents_pos_edges[1].to(device)
                    children_pos_edges = children_pos_edges[1].to(device)
                    parents_neg_edges = parents_neg_edges[1].to(device)
                    children_neg_edges = children_neg_edges[1].to(device)
                    parents_pos_weight = parents_pos_weight[1].to(device)
                    parents_neg_weight = parents_neg_weight[1].to(device)
                    children_pos_weight = children_pos_weight[1].to(device)
                    children_neg_weight = children_neg_weight[1].to(device)

                    output = model(parents=parents, parents_masks=parents_masks, parents_segs=parents_segs, 
                            children=children, children_masks=children_masks, children_segs=children_segs, 
                            parents_user_feat=parents_user_feat, parents_entities_feat=parents_entities_feat, 
                            children_user_feat=children_user_feat, children_entities_feat=children_entities_feat, 
                            parents_pos_e=parents_pos_edges, parents_neg_e=parents_neg_edges,
                            children_pos_e=children_pos_edges, children_neg_e=children_neg_edges,
                            parents_pos_weight=parents_pos_weight, parents_neg_weight=parents_neg_weight,
                            children_pos_weight=children_pos_weight, children_neg_weight=children_neg_weight)
                    
                    y_pred.append(torch.argmax(output, dim =1).cpu().detach().numpy())
                    y_true.append(labels.cpu().detach().numpy())

                print(classification_report(np.concatenate(y_true, axis = None).flatten(), np.concatenate(y_pred, axis = None).flatten()))

                
    torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_path))

if __name__ == '__main__':
    main()
