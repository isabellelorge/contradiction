import argparse
import logging
import random
import time
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import numpy as np
from dataset import *
from model import BertEntitiesGraph
import pandas as pd
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGloader
from helper_functions import create_graph_data_lists, multi_acc
from dataset import BertDataset, collate 
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory.')
    parser.add_argument('--trained_dir', default=None, type=str, required=True, help='Trained model directory.')
    parser.add_argument('--batch_size', default=8, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=0.000003, type=float, required=True, help='Learning rate.')
    parser.add_argument('--decay', default=1e-5, type=float, required=True, help='Weight decay.')
    parser.add_argument('--n_epochs', default=1, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--dropout', default=0.20, type=float, required=True, help='Dropout rate.')
    parser.add_argument('--hidden_size)', default=0.20, type=float, required=True, help='Size of hidden layer.')
    parser.add_argument('--bert_path', default='bert-base-cased', type=str, required=True, help='Path of BERT model.')
    parser.add_argument('--word2vec_path', default='bert-base-cased', type=str, required=True, help='Path of word2vec model.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    args = parser.parse_args()

    word2vec = Word2Vec.load(args.wor2vec_path)

    logging.info('Load training data...')
    df_train = pd.read_csv('{}/train.csv'.format(args.data_dir))

    logging.info('Load dev data...')
    df_valid = pd.read_csv('{}/dev.csv'.format(args.data_dir))
    
    logging.info('Load test data...')
    train_dataset = pd.read_csv('{}/test.csv'.format(args.data_dir))
    import json

    with open('{}/train_pos_edges.json'.format(args.data_dir), 'r') as f:
        train_pos_edges = json.load(f)
    with open('{}/train_neg_edges.json'.format(args.data_dir), 'r') as f:
        train_neg_edges = json.load(f)

    with open('{}/dev_pos_edges.json'.format(args.data_dir), 'r') as f:
        eval_pos_edges = json.load(f)
    with open('{}/dev_neg_edges.json'.format(args.data_dir), 'r') as f:
        eval_neg_edges = json.load(f)
    
    with open('{}/test_pos_edges.json'.format(args.data_dir), 'r') as f:
        test_pos_edges = json.load(f)
    with open('{}/test_neg_edges.json'.format(args.data_dir), 'r') as f:
        test_neg_edges = json.load(f)

    # shuffle
    df_train_shuffled = df_train.sample(frac=1, random_state = 123).reset_index(drop=True)

    train_dataset = BertDataset(df_train_shuffled, args.bert_path)
    eval_dataset = BertDataset(df_valid, args.bert_path)

    train_parents_entities_list, train_children_entities_list = create_graph_data_lists(df=df_train_shuffled, word2vec=word2vec, 
                                                        pos_edges=train_pos_edges, neg_edges=train_neg_edges)
    eval_parents_entities_list, eval_children_entities_list = create_graph_data_lists(df=df_valid, word2vec=word2vec, 
                                                        pos_edges=eval_pos_edges, neg_edges=eval_neg_edges)

    train_text_dataloader = DataLoader(train_dataset, shuffle=False, batch_size = args.batch_size, collate_fn = collate)
    train_parents_dataloader = PyGloader(train_parents_entities_list, batch_size = args.batch_size)
    train_children_dataloader = PyGloader(train_children_entities_list, batch_size = args.batch_size)

    eval_text_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size = args.batch_size, collate_fn = collate) 
    eval_parents_dataloader = PyGloader(eval_parents_entities_list, batch_size = args.batch_size)
    eval_children_dataloader = PyGloader(eval_children_entities_list, batch_size = args.batch_size)


    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
 
    torch.cuda.empty_cache()
    
    y = df_train['label'].values
    class_weights=compute_class_weight('balanced', classes = [0, 1, 2] , y = y)
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    class_weights = class_weights.to(device)

    model = BertEntitiesGraph(args.dropout, args.hidden_size, args.bert_path).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay) # leaving L2 reg to 0 so we can use L1 instead
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logging.info('Training classifier...')
    for epoch in range(1, args.n_epochs + 1):

        train_loss = 0
        train_acc = 0
        model.train()
        batch_idx = 0

        for text_batch, parents_batch, children_batch in zip(train_text_dataloader, train_parents_dataloader, train_children_dataloader):

            if batch_idx % 100 == 0 and batch_idx != 0:
                logging('Processed {} examples...'.format(batch_idx * args.batch_size))
            batch_idx += 1

            labels, parents, parents_masks, parents_segs, children, children_masks, children_segs = text_batch
            parents_pos_edges, parents_neg_edges, parents_pos_weight, parents_neg_weight, parents_entities_feat, parents_user_feat = parents_batch
            children_pos_edges, children_neg_edges,children_pos_weight, children_neg_weight, children_entities_feat, children_user_feat = children_batch

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

            optimizer.zero_grad()

            output = model(parents=parents, parents_masks=parents_masks, parents_segs=parents_segs, 
                        children=children, children_masks=children_masks, children_segs=children_segs, 
                        parents_user_feat=parents_user_feat, parents_entities_feat=parents_entities_feat, 
                        children_user_feat=children_user_feat, children_entities_feat=children_entities_feat, 
                        parents_pos_e=parents_pos_edges, parents_neg_e=parents_neg_edges,
                        children_pos_e=children_pos_edges, children_neg_e=children_neg_edges,
                        parents_pos_weight=parents_pos_weight, parents_neg_weight=parents_neg_weight,
                        children_pos_weight=children_pos_weight, children_neg_weight=children_neg_weight)

            
            if batch_idx % 50 == 0 and batch_idx != 0:
                print('LOSS', train_loss/batch_idx)
                print('ACC', train_acc/batch_idx)
            loss =  criterion(output, labels)
            acc = multi_acc(output, labels)
            train_loss+=loss.item()
            train_acc+=acc.item()

            loss.backward()
            optimizer.step()

        logging(f'Evaluate classifier epoch: {epoch}..')
        model.eval()

        y_true = list()
        y_pred = list()

        with torch.no_grad():
            for text_batch, parents_batch, children_batch in zip(eval_text_dataloader, eval_parents_dataloader, eval_children_dataloader):
                labels, parents, parents_masks, parents_segs, children, children_masks, children_segs = text_batch
                parents_pos_edges, parents_neg_edges, parents_pos_weight, parents_neg_weight, parents_entities_feat, parents_user_feat = parents_batch
                children_pos_edges, children_neg_edges,children_pos_weight, children_neg_weight, children_entities_feat, children_user_feat = children_batch

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

            # print(classification_report(np.array(y_true).flatten(), np.array(y_pred).flatten())) 
    
# torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_dir, filename))

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()