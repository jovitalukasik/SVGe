import sys
import os
import os.path
import torch 
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sparse_gp import SparseGP
import scipy.stats as sps
import random
import numpy as np
import json 
import time 
import scipy.io
from scipy.io import loadmat
from scipy.stats import pearsonr
sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.insert(0, '../')
from shutil import copy
import pdb

from ConfigSpace.read_and_write import json as config_space_json_r_w
from nasbench import api
from models.SVGe import SVGE
from models.DGMG import DGMG
from datasets.NASBench101 import nasbench, NUM_VERTICES, OPS_by_IDX_NB101

from utils import util 

import argparse

parser = argparse.ArgumentParser(description='Bayesian optimization experiments.')
parser.add_argument('--model_name',         default='SVGE', help='which model to evaluate, SVGE or DGMG')
parser.add_argument('--data_type',          default='NB101', help='graph dataset name: ENAS or NB101')
parser.add_argument('--checkpoint',         type=int, default=300, help="load which epoch's model checkpoint")
parser.add_argument('--log_dir',            default='res/', help='where to save the Bayesian optimization results') ####
parser.add_argument('--saved_log_dir',      default='../state_dicts/SVGE_NB101/', help='directory of saved model and latent data for evaluation')####
parser.add_argument('--BO_rounds',          type=int, default=10, help="how many rounds of BO to perform")
parser.add_argument('--BO_batch_size',      type=int, default=50, help="how many data points to select in each BO round")
parser.add_argument('--sample_dist',        default='uniform',  help='from which distrbiution to sample random points in the latent \
                                            space as candidates to select; uniform or normal')
parser.add_argument('--random_baseline',    action='store_true', default=True, help='whether to include a baseline that randomly selects points \
                                            to compare with Bayesian optimization')
parser.add_argument("--device",             type=str, default="cuda:0")
parser.add_argument('--share',              type=int, default=1, help='training SGP with complete training data or just an amount of args.keep')
parser.add_argument('--keep',               type=int, default=1000, help='training SGP with 1000 data')


args = parser.parse_args()
device = args.device

##############################################################################
#
#                              Runfolder
#
##############################################################################
#Create Log Directory
log_dir='{}_{}/{}/{}'.format(args.log_dir, args.model_name, args.data_type, time.strftime("%Y%m%d-%H%M%S"))


##############################################################################
#
#                           BO Config
#
##############################################################################

'''BO settings'''
BO_rounds = args.BO_rounds
batch_size = args.BO_batch_size
sample_dist = args.sample_dist
random_baseline = args.random_baseline 

# other BO hyperparameters
lr = 0.0005  # the learning rate to train the SGP model
max_iter = 100  # how many epochs to optimize the SGP each time 
decode_attempts=500 #how many times to decode points from the VAE model's latent space to return most common decoded graph 
M = 500 ##inducing points for SGP 

##############################################################################
#
#                           Dataset Config
#
##############################################################################
data_config_path='../configs/data_configs/NB101_configspace.json' ####
data_config = json.load(open(data_config_path, 'r'))

##############################################################################
#
#                           Model Config
#
##############################################################################
#Get Model configs
model_config_path='../configs/model_configs/svge_configspace.json' ####
model_configspace = config_space_json_r_w.read(open(model_config_path, 'r').read())
model_config = model_configspace.get_default_configuration().get_dictionary()

##############################################################################
#
#                              Model
#
##############################################################################

model=eval(args.model_name)(model_config=model_config, data_config=data_config).to(device)
state_dict = torch.load(args.saved_log_dir+ 'model_checkpoint{}.obj'.format(args.checkpoint),  map_location=device)
model.load_state_dict(state_dict)             


##############################################################################
#
#                           Training
#
##############################################################################


for rand_idx in range(1,11):
    if args.share:
        save_dir = '{}/{}/results_share_{}/'.format(log_dir, args.keep, rand_idx)  # where to save the BO results
    else:
        save_dir = '{}/results_{}/'.format(log_dir, rand_idx)  # where to save the BO results

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # backup files
    copy('bo_nb101.py', save_dir) 
    copy('run_bo_NB101.sh', save_dir)


    # load the data
    data = loadmat(args.saved_log_dir + '{}_latent_epoch{}.mat'.format(args.data_type, args.checkpoint))  # load train/test data

    X_train = data['Z_train'] 
    y_valid_train = data['Y_val_train'].reshape((-1,1))
    y_test_train = data['Y_test_train'].reshape((-1,1))
    y_time_train = data['Y_time_train'].reshape((-1,1))

    mean_y_train, std_y_train = np.mean(y_valid_train), np.std(y_valid_train)

    print('Mean, std of y_train is ', mean_y_train, std_y_train)

    if args.share:
        np.random.seed(rand_idx)
        random_shuffle = np.random.permutation(range(len(X_train)))
        keep=args.keep
        X_train=X_train[random_shuffle[:keep]]
        y_valid_train=y_valid_train[random_shuffle[:keep]]
        y_test_train=y_test_train[random_shuffle[:keep]]
        y_time_train=y_time_train[random_shuffle[:keep]]

    print('amount of training data:', len(X_train))

    # set seed
    random_seed = rand_idx
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)



    X_test = data['Z_test']
    y_val_test = data['Y_val_test'].reshape((-1,1))
    y_test_test = data['Y_test_test'].reshape((-1,1))
    y_time_test = data['Y_time_test'].reshape((-1,1))
    print('amount of test data:', len(X_test))

    best_train_score = max(y_valid_train)
    util.save_object((mean_y_train, std_y_train), "{}mean_std_y_train.dat".format(save_dir))

    print("Best train score is: ", best_train_score)

    '''Bayesian optimization begins here'''
    iteration = 0

    best_score = 1e-15
    best_score_test = 1e-15
    best_arc = None
    best_arc_time=1.5e6

    best_random_score = 1e-15
    best_random_score_test = 1e-15
    best_random_arc = None
    best_random_arc_time=1.5e6

    if args.share:
        print("Average pairwise distance between train points = {}".format(np.mean(pdist(X_train))))
        print("Average pairwise distance between test points = {}".format(np.mean(pdist(X_test))))

    if os.path.exists(save_dir + 'Test_RMSE_ll.txt'):
        os.remove(save_dir + 'Test_RMSE_ll.txt')
    if os.path.exists(save_dir + 'best_arc_scores.txt'):
        os.remove(save_dir + 'best_arc_scores.txt')
    while iteration < BO_rounds:

        # We fit the GP
        sgp = SparseGP(X_train, 0 * X_train, y_valid_train, M)
        sgp.train_via_ADAM(X_train, 0 * X_train, y_valid_train, X_test, X_test * 0,  \
               y_val_test  , minibatch_size = 2 * M, max_iterations = max_iter, learning_rate = lr)
        pred, uncert = sgp.predict(X_test, 0 * X_test)

        print("predictions: ", pred.reshape(-1))
        print("real values: ", y_val_test.reshape(-1))
        error = np.sqrt(np.mean((pred - y_val_test)**2))
        testll = np.mean(sps.norm.logpdf(pred - y_val_test, scale = np.sqrt(uncert)))
        print('Test RMSE: ', error)
        print('Test ll: ', testll)
        pearson = float(pearsonr(pred.reshape(-1), y_val_test.reshape(-1))[0])
        print('Pearson r: ', pearson)
        with open(save_dir + 'Test_RMSE_ll.txt', 'a') as test_file:
            test_file.write('Test RMSE: {:.4f}, ll: {:.4f}, Pearson r: {:.4f}\n'.format(error, testll, pearson))

        error_if_predict_mean = np.sqrt(np.mean((np.mean(y_valid_train, 0) - y_val_test)**2))
        print('Test RMSE if predict mean: ', error_if_predict_mean)
        pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_valid_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_valid_train, scale = np.sqrt(uncert)))
        print('Train RMSE: ', error)
        print('Train ll: ', trainll)

        # We pick the next batch of inputs
        
        next_inputs = sgp.batched_greedy_ei(batch_size, np.min(X_train, 0), np.max(X_train, 0), np.mean(X_train, 0), np.std(X_train, 0), sample=sample_dist) 
        valid_arcs_final = util.decode_from_latent_space(torch.FloatTensor(next_inputs).to(device), model, nasbench, decode_attempts, NUM_VERTICES, False, args.data_type)

        if random_baseline:
            #random_inputs = torch.randn(batch_size, nz).cuda()
            if sample_dist == 'uniform':
                random_inputs = np.random.rand(batch_size, model_config['graph_embedding_dim']) * (X_train.max(0)-X_train.min(0)) + X_train.min(0)
            elif sample_dist == 'normal':
                random_inputs = np.random.randn(batch_size, model_config['graph_embedding_dim']) * X_train.std(0) + X_train.mean(0)
            random_inputs = torch.FloatTensor(random_inputs).to(device)
            valid_arcs_random = util.decode_from_latent_space(random_inputs, model, nasbench, decode_attempts, NUM_VERTICES, False, args.data_type)
        
        new_features = next_inputs
        print("Evaluating selected points")
        scores = []
        scores_test = []
        arcs_time=[]
        for i in range(len(valid_arcs_final)):
            arc = valid_arcs_final[ i ] 
            if arc is not None:
                adj_flatten=np.array([int(s) for s in arc[0].split(' ')])
                nodes=([int(s) for s in arc[1].split(' ')])
                n=len(nodes)
                ops = [OPS_by_IDX_NB101[x] for x in nodes]
                matrix=adj_flatten.reshape(n,n)
                cell = api.ModelSpec(matrix=matrix, ops=ops)
                score=0
                score_test=0
                arc_time=0
                data=nasbench.query(cell)
                score=data['validation_accuracy']
                score_test=data['test_accuracy']
                arc_time=data['training_time']
            else:
                score = min(y_valid_train)[ 0 ]
                score_test = y_test_train[y_valid_train.argmin()][0]
                arc_time= y_time_train[y_valid_train.argmin()][0] 
            if score > best_score:
                best_score = score
                best_arc = arc
                best_score_test= score_test
                best_arc_time=arc_time
            scores.append(score)
            scores_test.append(score_test)
            arcs_time.append(arc_time)
            print(i)

        print("Iteration {}'s selected arcs' scores:".format(iteration))
        print(scores, np.mean(scores))
        util.save_object(scores, "{}scores{}.dat".format(save_dir, iteration))
        util.save_object(scores_test, "{}scores_test{}.dat".format(save_dir, iteration))
        util.save_object(arcs_time, "{}arcs_time{}.dat".format(save_dir, iteration))
        util.save_object(valid_arcs_final, "{}valid_arcs_final{}.dat".format(save_dir, iteration))

        if random_baseline:
            print("Evaluating random points")
            random_scores = []
            random_scores_test = []
            random_arcs_time=[]
            for i in range(len(valid_arcs_random)):
                arc = valid_arcs_random[ i ] 
                if arc is not None:
                    adj_flatten=np.array([int(s) for s in arc[0].split(' ')])
                    nodes=([int(s) for s in arc[1].split(' ')])
                    n=len(nodes)
                    ops = [OPS_by_IDX_NB101[x] for x in nodes]
                    matrix=adj_flatten.reshape(n,n)
                    cell = api.ModelSpec(matrix=matrix, ops=ops)
                    score=0
                    score_test=0
                    arc_time=0
                    data=nasbench.query(cell)
                    score=data['validation_accuracy']
                    score_test=data['test_accuracy']
                    arc_time=data['training_time']
                else:
                    score = min(y_valid_train)[ 0 ]
                    score_test = y_test_train[y_valid_train.argmin()][0]
                    arc_time= y_time_train[y_valid_train.argmin()][0] 
                if score > best_random_score:
                    best_random_score = score
                    best_random_arc = arc
                    best_random_score_test= score_test
                    best_random_arc_time = arc_time
                random_scores.append(score)
                random_scores_test.append(score_test)
                random_arcs_time.append(arc_time)
                print(i)

            print("Iteration {}'s selected arcs' scores:".format(iteration))
            print(scores, np.mean(scores))
            print("Iteration {}'s random arcs' scores:".format(iteration))
            print(random_scores, np.mean(random_scores))
            util.save_object(valid_arcs_random, "{}valid_arcs_random{}.dat".format(save_dir, iteration))
            util.save_object(random_scores, "{}random_scores{}.dat".format(save_dir, iteration))
            util.save_object(random_scores_test, "{}random_scores_test{}.dat".format(save_dir, iteration))
            util.save_object(random_arcs_time, "{}arcs_random_time{}.dat".format(save_dir, iteration))

        if len(new_features) > 0:
            X_train = np.concatenate([ X_train, new_features ], 0)
            y_valid_train = np.concatenate([ y_valid_train, np.array(scores)[ :, None ] ], 0)
            y_test_train = np.concatenate([ y_test_train, np.array(scores_test)[ :, None ] ], 0)
            y_time_train = np.concatenate([ y_time_train, np.array(arcs_time)[ :, None ] ], 0)

        print("Current iteration {}'s best score: {}, test score: {} and training time: {}".format(iteration, best_score, best_score_test, best_arc_time))
        if random_baseline:
            print("Current iteration {}'s best random score: {}, random test score: {} and training time {}".format(iteration, best_random_score, best_random_score_test,best_random_arc_time ))
        print("Best train score is: ", best_train_score)
        if best_arc is not None:
            print("Best architecture: ", best_arc)
            with open(save_dir + 'best_arc_scores.txt', 'a') as score_file:
                score_file.write(str(best_arc) + ', {:.4f}'.format(best_score)+ ', {:.4f}'.format(best_score_test) + ', {}\n'.format(best_arc_time) )
            g_best=util.parse_graph_to_nx(best_arc[0], best_arc[1], flat=True)
            util.plot_DAG(g_best, save_dir, 'best_arc_iter_{}'.format(iteration), data_type=args.data_type, pdf=True)
        iteration += 1
        print(iteration)

