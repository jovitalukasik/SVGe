import logging
logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
import numpy as np
import json ,random, time
import os, shutil, sys
from datetime import datetime

from scipy.io import loadmat
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader

from ConfigSpace.read_and_write import json as config_space_json_r_w

from utils import util
from models.SVGe import SVGE
from models.DGMG import DGMG


import argparse
parser = argparse.ArgumentParser(description='Graphautoencoder-training')
parser.add_argument('--model',                  type=str ,default='SVGE')
parser.add_argument('--name',                   type=str ,default='Train_VAE')
parser.add_argument('--data_search_space',      choices=['ENAS', 'NB101', 'NB201'], help= 'which search space for learning autoencoder', default='NB101')
parser.add_argument("--device",                 type=str, default="cuda:0")
parser.add_argument('--save_interval',          type=int, default=50, help='how many epochs to wait to save model')
parser.add_argument('--seed',                   type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--only_test_mode',         type=int, default=0, help='only testing of trained VAE from path: log_dir')
parser.add_argument('--log_dir',                type=str, default='state_dicts')
parser.add_argument('--test_epoch',             type=int, default='300')
args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


##############################################################################
#
#                              Runfolder
#
##############################################################################
if not args.only_test_mode:
        now = datetime.now()
        runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
        runfolder = f"{runfolder}_{args.name}_{args.data_search_space}"
        FOLDER_EXPERIMENTS = os.path.join(os.getcwd(), 'Experiments/VAE' )
        log_dir = os.path.join(FOLDER_EXPERIMENTS, runfolder)
        os.makedirs(log_dir)

        # save command line input
        cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
        with open(os.path.join(log_dir, 'cmd_input.txt'), 'a') as f:
            f.write(cmd_input)
        print('Command line input: ' + cmd_input + ' is saved.')

        print(f"Experiment folder: {log_dir}")

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
else:
    log_dir = args.log_dir
    
def main(args, log_dir):
    
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

    data_config = json.load(open(data_config_path, 'r'))

    ##############################################################################
    #
    #                           Model Config
    #
    ##############################################################################

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
    elif args.data_search_space == 'NB101':
        from datasets.NASBench101 import Dataset as Dataset101
        dataset = Dataset101(batch_size=data_config['batch_size'])
    elif args.data_search_space == 'ENAS':
        from datasets.ENAS import Dataset as DatasetENAS
        dataset = DatasetENAS(batch_size=data_config['batch_size'])
    else:
        raise TypeError("Unknow Dataset: {:}".format(args.data_search_space))

    print(f"Dataset size: {dataset.length}")

    ##############################################################################
    #
    #                              Model
    #
    ##############################################################################
    model=eval(args.model)(model_config=model_config, data_config=data_config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate']) 
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True) 

    logging.info("param size = %fMB", util.count_parameters_in_MB(model))

    if not args.only_test_mode:
        
        ##############################################################################
        #
        #                              Training
        #
        ##############################################################################
        budget = model_config['epochs']
        for epoch in range(1, int(budget)+1):
            logging.info('epoch: %s', epoch)
        
            # training
            train_obj=train(dataset, model, optimizer, epoch, args.device, log_dir)
            scheduler.step(train_obj)
                            
                
            # Save the model
            if epoch % args.save_interval == 0:
                logger.info('save model checkpoint {}  '.format(epoch))
                model_name = os.path.join(log_dir, 'model_checkpoint{}.obj'.format(epoch))
                torch.save(model.state_dict(), model_name)
                optimizer_name = os.path.join(log_dir, 'optimizer_checkpoint{}.obj'.format(epoch))
                torch.save(optimizer.state_dict(), optimizer_name)
                scheduler_name = os.path.join(log_dir, 'scheduler_checkpoint{}.obj'.format(epoch))
                torch.save(scheduler.state_dict(), scheduler_name)    
            
            config_dict = {
                    'epochs': epoch,
                    'loss': train_obj,
                    }

            
            with open(os.path.join(log_dir, 'results.txt'), 'a') as file:
                    json.dump(str(config_dict), file)
                    file.write('\n')

            ##############################################################################
            #
            #                              Test
            #
            ##############################################################################
            
            if epoch % args.save_interval == 0:
                        
                if args.data_search_space=='ENAS':
                    test(dataset, model, epoch, args.device, log_dir, data_config, DatasetENAS, ENAS=True)
                elif args.data_search_space=='NB101':
                    test(dataset, model, epoch, args.device, log_dir, data_config, Dataset101, NB101=True)
                elif args.data_search_space=='NB201':
                    test(dataset, model, epoch, args.device, log_dir, data_config, Dataset201, NB201=True)
                else:
                    raise TypeError("Unknow Seach Space : {:}".format(args.data_search_space))     
    
    else:
            ##############################################################################
            #
            #                              Only Test
            #
            ##############################################################################

            log_dir = os.path.join(args.log_dir, args.model+'_'+args.data_search_space)
            
            epoch = args.test_epoch
            m = torch.load(os.path.join(log_dir, f"model_checkpoint{epoch}.obj"), map_location=args.device)
            model.load_state_dict(m)


            if args.data_search_space=='ENAS':
                test(dataset, model, epoch, args.device, log_dir, data_config, DatasetENAS, ENAS=True, pretrained=True)
            elif args.data_search_space=='NB101':
                test(dataset, model, epoch, args.device, log_dir, data_config, Dataset101, NB101=True, pretrained=True)
            elif args.data_search_space=='NB201':
                test(dataset, model, epoch, args.device, log_dir, data_config, Dataset201, NB201=True, pretrained=True)
            else:
                raise TypeError("Unknow Seach Space : {:}".format(args.data_search_space))  

def train(data, model, optimizer, epoch, device, log_dir):
    objs = util.AvgrageMeter()
    recon_objs = util.AvgrageMeter()
    kl_objs = util.AvgrageMeter()
    # TRAINING
        
    model.train()
   
    for step, graph_batch in enumerate(data.train_dataloader):
        for i in range(len(graph_batch)):
                graph_batch[i].to(device)
        loss, recon_loss, kl_loss = model(graph_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = graph_batch[0].num_graphs
        objs.update(loss.data.item(), n)
        recon_objs.update(recon_loss.data.item(), n)
        kl_objs.update(kl_loss.data.item(), n)

            
        config_dict = {
             'epoch': epoch,
             'recon_loss':recon_objs.avg,
             'kl_loss': kl_objs.avg,
             'loss':objs.avg,
                }
    
        with open(os.path.join(log_dir, 'loss.txt'), 'a') as file:
            json.dump(str(config_dict), file)
            file.write('\n')
        
        
    logging.info('train %03d %.5f', step, objs.avg)
    
    return objs.avg

def test(data, model, epoch, device, log_dir, data_config, Dataset, ENAS=False, NB101=False, NB201=False, pretrained=False):
    with torch.no_grad():
        model.eval()

        #Reconstruction Accuracy
        logger.info('Run: Test Dataset for Reconstruction Accuracy')
        rec = util.recon_accuracy(data.test_dataloader, model, args.device)             

        logger.info('Reconstruction Accuracy on test set for model {} is {}'.format(model.__class__.__name__,rec))
        
        #Prior Ability, Uniqueness, Novelty
        logger.info('Run: Train Dataset for Validity Tests')

        logger.info('Extract mean and std of latent space ')
        if not pretrained:
            util.save_latent_representations(Dataset, model , device, epoch, log_dir, data_name=data_config['data_name'], ENAS=ENAS, NB101=NB101, NB201=NB201)
        
        data = loadmat(os.path.join(log_dir,  data_config['data_name']+'_latent_epoch{}.mat'.format(epoch)))
        Z_train = data['Z_train'] 

        n_latent_points=1000
        valid, unique, novel = util.prior_validity(Dataset, model, Z_train, n_latent_points,
                                                device, scale_to_train_range=True, ENAS=ENAS, NB101=NB101, NB201=NB201)
        
        model.train()
                                                

    logger.info("For epoch {} of model {} reconstruction accuracy for test graph {}, valid graphs {}, unique graphs {} and novel graphs {} ".format(epoch, model.__class__.__name__, rec, valid, unique, novel ))
    config_dict = {
        'epoch':epoch,
        'reconstruction_acc': rec,
        'valid': valid,
        'unique': unique,
        'novel': novel
    }
    with open(os.path.join(log_dir, 'results_validity.txt'), 'a') as file:
        json.dump(str(config_dict), file)
        file.write('\n')

if __name__ == '__main__':
    main(args, log_dir)


