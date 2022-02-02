import logging
logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
import numpy as np
import json 
import random
import os
import sys
import time 
from scipy.io import loadmat
import torch
import torch.nn as nn
from datetime import datetime

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader

from ConfigSpace.read_and_write import json as config_space_json_r_w


from utils import util
from datasets.utils_data import prep_data
from models.SVGe import SVGE_acc, SVGE
from models.DGMG import DGMG_acc, DGMG

import argparse
parser = argparse.ArgumentParser(description='Surrogate-Model-training')
parser.add_argument('--model',                  type=str ,default='SVGE_acc')
parser.add_argument('--name',                   type=str ,default='Train_PP')
parser.add_argument('--data_search_space',      choices=['ENAS', 'NB101', 'NB201'], help= 'which search space for learning autoencoder', default='NB101')
parser.add_argument("--device",                 type=str, default="cuda:1")
parser.add_argument('--path_state_dict',        type=str, help='directory to saved model', default='state_dicts/SVGE_NB101/')
parser.add_argument('--save_interval',          type=int, default=50, help='how many epochs to wait to save model')
parser.add_argument('--seed',                   type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--checkpoint',             type=int, default=300, help='Which checkpoint of trained model to load')
parser.add_argument('--test',                   type=int, default=1, help='if predict on test acc')
parser.add_argument('--sample_amount',          type=int, default=1000, help='fine tuning VAE and surrogate on 1000 training data')

args = parser.parse_args()

seed = args.seed
print(f"Random Seed: {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = args.device

##############################################################################
#
#                              Runfolder
#
##############################################################################
#Create Log Directory
now = datetime.now()
runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
runfolder = f"{args.data_search_space}/{args.model}/{args.sample_amount}/{runfolder}_{args.name}"

FOLDER_EXPERIMENTS = os.path.join(os.getcwd(), 'Experiments/Surrogate/')
log_dir = os.path.join(FOLDER_EXPERIMENTS, runfolder)
os.makedirs(log_dir)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(log_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')



for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main(args):
    ##############################################################################
    #
    #                           Dataset Config
    #
    ##############################################################################

    if args.data_search_space=='ENAS':
        data_config_path='configs/data_configs/ENAS_configspace.json'
    elif args.data_search_space=='NB101':
        data_config_path='configs/data_configs/NB101_configspace.json'
    elif args.data_search_space=='NB201':
        data_config_path='configs/data_configs/NB201_configspace.json'
    else:
        raise TypeError("Unknow Seach Space : {:}".format(args.data_search_space))
    #Get Data specific configs
    data_config = json.load(open(data_config_path, 'r'))

    ##############################################################################
    #
    #                           Model Config
    #
    ##############################################################################
    #Get Model configs
    model_config_path='configs/model_configs/svge_configspace.json'
    model_configspace = config_space_json_r_w.read(open(model_config_path, 'r').read())
    model_config = model_configspace.get_default_configuration().get_dictionary()


    ##############################################################################
    #
    #                              Dataset
    #
    ##############################################################################
    print("Creating Dataset.")
    if args.data_search_space == 'NB201':
        from datasets.NASBench201 import Dataset as Dataset201
        dataset = Dataset201(batch_size=data_config['batch_size'])
        print("Prepare Test Set")
        val_data = dataset.test_data
        val_data=prep_data(val_data,data_config['max_num_nodes'],  NB201=True)
    elif args.data_search_space == 'NB101':
        from datasets.NASBench101 import Dataset as Dataset101
        dataset = Dataset101(batch_size=data_config['batch_size'])
        print("Prepare Test Set")
        val_data = dataset.test_data
        val_data=prep_data(val_data,data_config['max_num_nodes'],  NB101=True)
    elif args.data_search_space == 'ENAS':
        from datasets.ENAS import Dataset as DatasetENAS
        dataset = DatasetENAS(batch_size=data_config['batch_size'])
        print("Prepare Test Set")
        val_data = dataset.test_data
        val_data=prep_data(val_data,data_config['max_num_nodes'])
    else:
        raise TypeError("Unknow Dataset: {:}".format(args.data_search_space))

    # Sample train Data 
    data = dataset.train_data
    random_shuffle = np.random.permutation(range(len(data)))
    train_data = [data[i] for i in random_shuffle[:args.sample_amount]]
    print(f"Dataset size: {len(train_data)}")


    torch.save(random_shuffle[:args.sample_amount], os.path.join(log_dir, 'sampled_train_idx.pth'))
    
    ##############################################################################
    #
    #                              Model
    #
    ##############################################################################
    model=eval(args.model)(model_config=model_config, data_config=data_config).to(device)
    model_dict = model.state_dict()

    path_state_dict=args.path_state_dict
    checkpoint = args.checkpoint
    m = torch.load(os.path.join(path_state_dict, f"model_checkpoint{checkpoint}.obj"), map_location=device)
    m = {k: v for k, v in m.items() if k in model_dict}

    model_dict.update(m)
    model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['regression_learning_rate']) 
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True) 
    alpha= model_config['regression_loss_proportion']
    criterion = nn.MSELoss()

    ##############################################################################
    #
    #                              Training
    #
    ##############################################################################

    budget = model_config['regression_epochs']
    for epoch in range(1, int(budget)+1):
        logging.info('epoch: %s', epoch)
    
        # training
        train_obj, train_results=train(train_data, model, criterion, optimizer, epoch, device, alpha, data_config, log_dir)
        scheduler.step(train_obj)

        if epoch % args.save_interval == 0:
            logger.info('save model checkpoint {}  '.format(epoch))
            model_name = os.path.join(log_dir, 'model_checkpoint{}.obj'.format(epoch))
            torch.save(model.state_dict(), model_name)
            optimizer_name = os.path.join(log_dir, 'optimizer_checkpoint{}.obj'.format(epoch))
            torch.save(optimizer.state_dict(), optimizer_name)
            scheduler_name = os.path.join(log_dir, 'scheduler_checkpoint{}.obj'.format(epoch))
            torch.save(scheduler.state_dict(), scheduler_name)    
        

        # validation  
        if epoch % 5 == 0:              
            valid_obj, valid_results = infer(val_data, model,criterion, optimizer, epoch, device,alpha, data_config, log_dir)

            config_dict = {
                'epochs': epoch,
                'loss':  train_results["rmse"],
                'val_rmse': valid_results['rmse'],
                'kendall_tau':valid_results['kendall_tau'],
                "spearmanr":valid_results['spearmanr'],
            }
            with open(os.path.join(log_dir, 'results.txt'), 'a') as file:
                        json.dump(str(config_dict), file)
                        file.write('\n')

def train(train_data, model,criterion, optimizer, epoch, device,alpha, data_config, log_dir):
    objs = util.AvgrageMeter()
    vae_objs=util.AvgrageMeter()
    acc_objs=util.AvgrageMeter()
    # TRAINING
    preds = []
    targets = []

    model.train()
   
    data_loader = DataLoader(train_data, shuffle=True, num_workers=data_config['num_workers'], pin_memory=True, batch_size=data_config['batch_size'])
    for step, graph_batch in enumerate(data_loader):
        for i in range(len(graph_batch)):
                graph_batch[i].to(device)
        vae_loss, recon_loss, kl_loss, pred =model(graph_batch)
        pred=pred.view(-1)
        if args.test:
            acc_loss= criterion(pred.view(-1), graph_batch[0].test_acc)
        else:
            acc_loss= criterion(pred.view(-1), graph_batch[0].acc)
        loss=alpha*vae_loss +(1-alpha)*acc_loss

        preds.extend((pred.detach().cpu().numpy()))
        if args.test:
            targets.extend(graph_batch[0].test_acc.detach().cpu().numpy())
        else:
            targets.extend(graph_batch[0].acc.detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = graph_batch[0].num_graphs
        objs.update(loss.data.item(), n)
        vae_objs.update(vae_loss.data.item(), n)
        acc_objs.update(acc_loss.data.item(), n)


            
    config_dict = {
            'epoch': epoch,
            'vae_loss':vae_objs.avg,
            'acc_loss': acc_objs.avg,
            'loss':objs.avg,
            }
    
    with open(os.path.join(log_dir, 'loss.txt'), 'a') as file:
        json.dump(str(config_dict), file)
        file.write('\n')


    logging.info('train %03d %.5f', step, objs.avg)
    train_results = util.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
    logging.info('train metrics:  %s', train_results)
    return objs.avg, train_results

def infer(val_data, model, criterion, optimizer, epoch, device,alpha, data_config, log_dir):
    objs = util.AvgrageMeter()
    vae_objs=util.AvgrageMeter()
    acc_objs=util.AvgrageMeter()

    # VALIDATION
    preds = []
    targets = []
    
    model.eval()
    data_loader = DataLoader( val_data, shuffle=False, num_workers=data_config['num_workers'], batch_size=data_config['batch_size'])
    for step, graph_batch in enumerate(data_loader):
        with torch.no_grad():
            for i in range(len(graph_batch)):
                    graph_batch[i].to(device)
            vae_loss, recon_loss, kl_loss, pred = model(graph_batch)
            pred=pred.view(-1)

            if args.test:
                acc_loss= criterion(pred.view(-1), graph_batch[0].test_acc)
            else:
                acc_loss= criterion(pred.view(-1), graph_batch[0].acc)
            loss=alpha*vae_loss +(1-alpha)*acc_loss

            preds.extend((pred.detach().cpu().numpy()))

            if args.test:
                targets.extend(graph_batch[0].test_acc.detach().cpu().numpy())
            else:
                targets.extend(graph_batch[0].acc.detach().cpu().numpy())


        n = graph_batch[0].num_graphs
        objs.update(loss.data.item(), n)
        vae_objs.update(vae_loss.data.item(), n)
        acc_objs.update(acc_loss.data.item(), n)

    config_dict = {
            'epoch': epoch,
            'vae_loss_val':vae_objs.avg,
            'acc_loss_val': acc_objs.avg,
            'loss-val':objs.avg,
            }
    
    with open(os.path.join(log_dir, 'loss.txt'), 'a') as file:
        json.dump(str(config_dict), file)
        file.write('\n')
    
    logging.info('val %03d %.5f', step, objs.avg)
    val_results = util.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
    logging.info('val metrics:  %s', val_results)
    return objs.avg, val_results

if __name__ == '__main__':
    main(args)
