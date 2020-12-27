import torch
from torch import nn
import os,sys,inspect
from utils import SignalDataset_music
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
import time
import random
from sklearn.metrics import average_precision_score

from c2nn.model import *

os.environ['CUDA_VISIBLE_DEVICE']='0,1,2,3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = True

def train_transformer(args,training_set,train_loader,test_set,test_loader):
    model = TransformerModel(time_step=args['time_step'],
                             input_dims=args['modal_lengths'],
                             hidden_size=args['hidden_size'],
                             embed_dim=args['embed_dim'],
                             output_dim=args['output_dim'],
                             num_heads=args['num_heads'],
                             attn_dropout=args['attn_dropout'],
                             relu_dropout=args['relu_dropout'],
                             res_dropout=args['res_dropout'],
                             out_dropout=args['out_dropout'],
                             layers=args['nlevels'],
                             attn_mask=args['attn_mask'])
    if use_cuda:
        model = model.cuda()
        print("use gpu")
    else:
        print("use cpu")

    print("Model size: {0}".format(count_parameters(model)))

    optimizer = getattr(optim, args['optim'])(model.parameters(), lr=args['lr'], weight_decay=1e-7)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    print("point1 train_transformer ok")

    #model = nn.DataParallel(model,device_ids=[0,1,2])
    #print("use multi gpu")

    return train_model(settings,args,training_set,train_loader,test_set,test_loader)


def train_model(settings,args,training_set,train_loader,test_set,test_loader):
    print("point2 train_model begin")
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    model.to(device)
    def train(model, optimizer, criterion):
        print("train begin")
        epoch_loss = 0.0
        batch_size = args['batch_size']
        num_batches = len(training_set) // batch_size
        total_batch_size = 0
        start_time = time.time()
        shape = (args['time_step'], training_set.len, args['output_dim'])
        true_vals = torch.zeros(shape)
        pred_vals = torch.zeros(shape)
        
        model.train()
        print("train1 done")
        print("train_loader",len(train_loader))
        for i_batch, (batch_X, batch_y) in enumerate(train_loader):
            print("i",i_batch)
            model.zero_grad()
            batch_X = batch_X.transpose(0, 1)
            batch_y = batch_y.transpose(0, 1)
            batch_X, batch_y = batch_X.float().to(device=device), batch_y.float().to(device=device)
            preds = model(batch_X)
            true_vals[:, i_batch*batch_size:(i_batch+1)*batch_size, :] = batch_y.detach().cpu()
            pred_vals[:, i_batch*batch_size:(i_batch+1)*batch_size, :] = preds.detach().cpu()
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            optimizer.step()
            total_batch_size += batch_size
            epoch_loss += loss.item() * batch_size
        aps = average_precision_score(true_vals.flatten(), pred_vals.flatten())
        aps = 0
        print("train done")
        return epoch_loss / len(training_set), aps

    def evaluate(model, criterion):
        epoch_loss = 0.0
        batch_size = args['batch_size']
        loader = test_loader
        total_batch_size = 0
        shape = (args['time_step'], test_set.len, args['output_dim']) 
        true_vals = torch.zeros(shape)
        pred_vals = torch.zeros(shape)
        model.eval()
        with torch.no_grad():
            for i_batch, (batch_X, batch_y) in enumerate(loader):
                batch_X = batch_X.transpose(0, 1)
                batch_y = batch_y.transpose(0, 1)
                batch_X, batch_y = batch_X.float().to(device=device), batch_y.float().to(device=device)
                preds = model(batch_X)
                true_vals[:, i_batch*batch_size:(i_batch+1)*batch_size, :] = batch_y.detach().cpu()
                pred_vals[:, i_batch*batch_size:(i_batch+1)*batch_size, :] = preds.detach().cpu()
                loss = criterion(preds, batch_y)
                total_batch_size += batch_size
                epoch_loss += loss.item() * batch_size
            aps = average_precision_score(true_vals.flatten(), pred_vals.flatten())
        return epoch_loss / len(test_set), aps


    print("point3 def ok")
    print("num_epochs",args['num_epochs'])
    for epoch in range(args['num_epochs']):
        start = time.time() 

        train_loss, acc_train = train(model, optimizer, criterion)
        print('Epoch {:2d} | Train Loss {:5.4f} | APS {:5.4f}'.format(epoch, train_loss, acc_train))
        test_loss, acc_test = evaluate(model, criterion)
        scheduler.step(test_loss)
        print("-"*50)
        print('Epoch {:2d} | Test  Loss {:5.4f} | APS {:5.4f}'.format(epoch, test_loss, acc_test))
        print("-"*50)

        end = time.time()
        print("time: %d" % (end - start))


