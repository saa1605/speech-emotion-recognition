import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from IEMOCAP_utils import *
# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100

MODEL_PATH = './models/iemocap_pretrained/'


no_cuda=False           #does not use GPU
base_model='LSTM'       #base recurrent model, must be one of DialogRNN/LSTM/GRU
graph_model=False       #whether to use graph model after recurrent encoding
nodal_attention=False   #whether to use nodal attention in graph model: Equation 4,5,6 in Paper
windowp=10              #context window size for constructing edges in graph model for past utterances')
windowf=10              #context window size for constructing edges in graph model for future utterances')
lr=0.0001               #learning rate
l2=0.00001              #L2 regularization weight
rec_dropout=0.1         #rec_dropout rate
dropout=0.5             #dropout rate
batch_size=32           #batch size
n_epochs=60             #number of epochs
class_weight=False      #use class weights
active_listener=False   #active listener
attention='general'     #Attention type in DialogRNN model
tensorboard=False       #Enables tensorboard log
n_classes  = 6
D_m = 100
D_g = 150
D_p = 150
D_e = 100
D_h = 100
D_a = 100
graph_h = 100



if graph_model:
    seed_everything()
    model = DialogueGCNModel(base_model,
                                D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                n_speakers=2,
                                max_seq_len=110,
                                window_past=windowp,
                                window_future=windowf,
                                n_classes=n_classes,
                                listener_state=active_listener,
                                context_attention=attention,
                                dropout=dropout,
                                nodal_attention=nodal_attention,
                                no_cuda=no_cuda)

    print ('Graph NN with', base_model, 'as base model.')
    name = 'Graph'

else:
    if base_model == 'DialogRNN':
        model = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a, 
                                n_classes=n_classes,
                                listener_state=active_listener,
                                context_attention=attention,
                                dropout_rec=rec_dropout,
                                dropout=dropout)

        print ('Basic Dialog RNN Model.')


    elif base_model == 'GRU':
        model = GRUModel(D_m, D_e, D_h, 
                            n_classes=n_classes, 
                            dropout=dropout)

        print ('Basic GRU Model.')


    elif base_model == 'LSTM':
        model = LSTMModel(D_m, D_e, D_h, 
                            n_classes=n_classes, 
                            dropout=dropout)

        print ('Basic LSTM Model.')

    else:
        print ('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
        raise NotImplementedError

    name = 'Base'

train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                batch_size=batch_size,
                                                                num_workers=0)


if tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()


cuda = init_cuda(no_cuda)

if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
all_fscore, all_acc, all_loss = [], [], []

for e in range(n_epochs):
    start_time = time.time()

    if graph_model:
        train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, train_loader, e, cuda, optimizer, True)
        valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, valid_loader, e, cuda)
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, test_loader, e, cuda)
        all_fscore.append(test_fscore)
    
        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_attn =\
                    test_loss, test_label, test_pred, test_mask, attentions
        # torch.save({'model_state_dict': model.state_dict()}, path + name + base_model + '_' + str(e) + '.pkl')

    else:
        # cuda ,model, dataloader, epoch, optimizer=None, train=False, tensorboard=False ,class_weight=True
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore, _ = eval_base_model(model, valid_loader, cuda)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = eval_base_model(model, test_loader, cuda)
        all_fscore.append(test_fscore)
    
        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_attn =\
                    test_loss, test_label, test_pred, test_mask, attentions
        # torch.save({'model_state_dict': model.state_dict()}, path + name + base_model + '_' + str(e) + '.pkl')
    
    if tensorboard:
        writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
        writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)

    print('epoch {} train_loss {} train_acc {} train_fscore{}'.format(e+1, train_loss, train_acc, train_fscore))
    print('valid_loss {} valid_acc {} val_fscore{}'.format(valid_loss, valid_acc, valid_fscore))
    print('test_loss {} test_acc {} test_fscore {} time {}'.format(test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))



if tensorboard:
    writer.close()

save_model(model,optimizer,name,base_model)

print('Test performance..')
print ('F-Score:', max(all_fscore))
print('Loss {} accuracy {}'.format(best_loss,
                                  round(accuracy_score(best_label,best_pred,sample_weight=best_mask)*100,2)))
print(classification_report(best_label,best_pred,sample_weight=best_mask,digits=4))
print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))