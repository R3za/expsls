from dependencies import *

from objectives import *
from datasets import *
from utils import *

from optimizers.exp_step_sgd import *
from optimizers.exp_step_acc_sgd import *
from optimizers.rit_sgd import *


import argparse
import exp_configs

import tqdm
import pandas as pd
import pprint
import math
import itertools
import os, sys
import pylab as plt
import exp_configs
import time
import numpy as np
import shutil

from haven import haven_utils as hu
from haven import haven_chk as hc
from haven import haven_jobs as hj


Lparam={}
def trainval(exp_dict, savedir_base, reset=False):

	# dataset options
	data_dir = './'

	# get experiment directory
	exp_id = hu.hash_dict(exp_dict)
	opt_dict = exp_dict["opt"]
	savedir = os.path.join(savedir_base, exp_id)

	if reset:
	    # delete and backup experiment
	    hc.delete_experiment(savedir, backup_flag=True)

	# create folder and save the experiment dictionary
	os.makedirs(savedir, exist_ok=True)
	hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
	pprint.pprint(exp_dict)
	print('Experiment saved in %s' % savedir)

	score_list_path = os.path.join(savedir, 'score_list.pkl')

	seed = 42 + exp_dict['runs']
	np.random.seed(seed)

	# default values	
	if "is_subsample" not in exp_dict.keys():
		is_subsample = 0
	else:
		is_subsample = exp_dict["is_subsample"]

	if "is_kernelize" not in exp_dict.keys():
		is_kernelize = 0
	else:
		is_kernelize = exp_dict["is_kernelize"]

	if "false_ratio" not in exp_dict.keys():		
		false_ratio = 0		
	else:
		false_ratio = exp_dict["false_ratio"]
        
	if "standardize" not in exp_dict.keys():		
		standardize = False		
	else:
		standardize = exp_dict["standardize"]
	if "remove_strong_convexity" not in exp_dict.keys():		
		remove_strong_convexity = False		
	else:
		remove_strong_convexity = exp_dict["remove_strong_convexity"]
	
    # load the dataset
	if exp_dict["dataset"] == "synthetic":
	    n, d = exp_dict["n_samples"], exp_dict["d"]
	    false_ratio = exp_dict["false_ratio"]	   
	    margin = exp_dict["margin"]	    
	    X, y, X_test, y_test = data_load(data_dir, exp_dict["dataset"],n, d, margin,
										 false_ratio, standardize=standardize, remove_strong_convexity=remove_strong_convexity)

	elif exp_dict["dataset"] == "synthetic_ls" or exp_dict["dataset"] == "synthetic_reg":
		n, d = exp_dict["n_samples"], exp_dict["d"]
		X, y, X_test, y_test = data_load(data_dir, exp_dict["dataset"], n, d,
										 standardize=standardize, remove_strong_convexity=remove_strong_convexity)
	else:
		if is_subsample == 1:
			n = exp_dict["subsampled_n"]
		else:	
			n = 0

		if is_kernelize == 1:
			d = n
		else:
			d = 0

		X, y, X_test, y_test = data_load(data_dir, exp_dict["dataset"] , n, d, false_ratio,
										 is_subsample=is_subsample, is_kernelize=is_kernelize, standardize=standardize,
										 remove_strong_convexity=remove_strong_convexity)
		n = X.shape[0]
	if exp_dict["dataset"] in Lparam.keys():
		Lmax=Lparam[exp_dict["dataset"]]
	else:
		Lmax=param_l(X)
		Lparam[exp_dict["dataset"]]=Lmax

	#set 1/n as reg factor for all exp
	# exp_dict["regularization_factor"]= 0.01 #1./n
	regularization_factor = exp_dict["regularization_factor"]
	if exp_dict["loss_func"] == "logistic_loss":
		closure = make_closure(logistic_loss, regularization_factor)
		Lmax, Lmin = 1./4*Lmax+regularization_factor,regularization_factor
	elif exp_dict["loss_func"] == "squared_hinge_loss":
		closure = make_closure(squared_hinge_loss, regularization_factor)
	elif exp_dict["loss_func"] == "squared_loss":
		closure = make_closure(squared_loss, regularization_factor)
		Lmax, Lmin = Lmax+regularization_factor,regularization_factor
	elif exp_dict["loss_func"] == "huber_loss":
		closure = make_closure(huber_loss, regularization_factor)
	else:
		print("Not a valid loss")

	# check if score list exists 
	if os.path.exists(score_list_path):
		# resume experiment
		score_list = hu.load_pkl(score_list_path)
		s_epoch = score_list[-1]['itr'] + 1
	else:
		# restart experiment
		score_list = []
		s_epoch = 0

	print('Starting experiment at epoch %d' % (s_epoch))

	if opt_dict["name"] == "EXP_SGD":
		score_list = Exp_SGD(score_list, closure=closure, batch_size=exp_dict["batch_size"],
						 max_epoch=exp_dict["max_epoch"],
						 gamma=1./Lmax,
						 D=X, labels=y,
						 is_sls=opt_dict['is_sls'],
						 alpha_t=opt_dict['alpha_t'],
						 D_test=X_test, labels_test=y_test)


	elif opt_dict["name"] == "EXP_ACC_SGD":
		score_list = Exp_ACC_SGD(score_list, closure=closure, batch_size=exp_dict["batch_size"],
						 max_epoch=exp_dict["max_epoch"],
						 gamma=1./(opt_dict["rho"]*Lmax),
						 D=X, labels=y,
						 L=Lmax,
						 rho=opt_dict["rho"],
						 mu=Lmin,
						 alpha_t=opt_dict['alpha_t'],
						 D_test=X_test, labels_test=y_test)

	# def RIT_SGD(score_list, closure, D, labels, batch_size=1, max_epoch=100,
	# 			x0=None, mu=0.1, L_max=0.1, rho=1, verbose=True, D_test=None, labels_test=None, log_idx=1000):
	elif opt_dict["name"] == "RIT_SGD":
		score_list = RIT_SGD(score_list, closure=closure, batch_size=exp_dict["batch_size"],
						 max_epoch=exp_dict["max_epoch"],
						 D=X, labels=y,
						 L_max=Lmax,
						 rho=opt_dict["rho"],
						 mu=Lmin,
						 D_test=X_test, labels_test=y_test)


	else:
		print('Method does not exist')
		return 1/0
    
	save_pkl(score_list_path, score_list)  

	return score_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)
    parser.add_argument('-v', '--view_jupyter', default=None)
    parser.add_argument('-j', '--run_jobs', default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]


    # Run experiments or View them
    # ----------------------------
    if args.run_jobs:
        # launch jobs
        from haven import haven_jobs as hj
        hj.run_exp_list_jobs(exp_list, 
                       savedir_base=args.savedir_base, 
                       workdir=os.path.dirname(os.path.realpath(__file__)))

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval

            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    reset=args.reset)


# exp_mushrooms exp_ijcnn exp_rcv1