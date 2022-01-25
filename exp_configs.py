from haven import haven_utils as hu
import itertools
import numpy as np


def get_benchmark(benchmark,
                  opt_list,
                  batch_size=[1, 100, -1],
                  runs=[0, 1, 2, 3, 4],
                  max_epoch=[50],
                  losses=["logistic_loss", "squared_loss", "squared_hinge_loss"]
                  ):
    if benchmark == "mushrooms":
        return {"dataset": ["mushrooms"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 0.01,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "ijcnn":
        return {"dataset": ["ijcnn"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 0.01,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "a1a":
        return {"dataset": ["a1a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 0.01,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "a2a":
        return {"dataset": ["a2a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 1. / 2300,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "w8a":
        return {"dataset": ["w8a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 1. / 50000,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "covtype":
        return {"dataset": ["covtype"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 1. / 500000,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "phishing":
        return {"dataset": ["phishing"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 1e-4,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "rcv1":
        return {"dataset": ["rcv1"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 0.01,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs,
                "is_subsample": 1,
                "subsampled_n": 10000}

    elif benchmark == "synthetic_interpolation":
        return {"dataset": ["synthetic"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor": 0.01,
                "margin": [0.1],
                "false_ratio": [0, 0.1, 0.2],
                "n_samples": [10000],
                "d": [200],
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}
    elif benchmark == "synthetic_ls":
        return {"dataset": ["synthetic_ls"],
                "loss_func": ["squared_loss"],
                "opt": opt_list,
                "regularization_factor": 0.,
                "n_samples": [10000],
                "d": [20],
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}

    elif benchmark == "synthetic_reg":
        return {"dataset": ["synthetic_reg"],
                "loss_func": ["logistic_loss"],
                "opt": opt_list,
                "regularization_factor": 1. / 10000,
                "n_samples": [10000],
                "d": [20],
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "runs": runs}
    else:
        print("Benchmark unknown")
        return


EXP_GROUPS = {}

#
# # benchmarks_list = ["mushrooms", "ijcnn", "a1a", "a2a", "w8a", "rcv1", "covtype", "phishing"]
# # benchmarks_interpolation_list = ["synthetic_interpolation"]
#

benchmarks_list = ["mushrooms", "ijcnn", "rcv1"]

for benchmark in benchmarks_list:
    EXP_GROUPS["exp_%s" % benchmark] = []

opt_list = []
MAX_EPOCH=10
# MAX_EPOCH = 100

# RUNS = [0]
RUNS=[0,1,2,3,4]


pis = [1]

# SGD
for alphat in ["DECR"]:
    for sls in [False, True]:
        opt_list += [{'name': 'EXP_SGD',
                      'alpha_t': alphat,
                      'is_sls': sls,
                      }]

opt_list += [{'name': 'EXP_SGD',
                      'alpha_t': "CNST",
                      'is_sls': False,
                      }]
#
# # MASG
opt_list += [{'name': 'M_ASG',
              'p': 1,
              'c': 10
              }]

# #ASGD
rhos=[100]
# rho = 1

for rho in rhos:
    for alphat in [ "DECR"]:
        for sls in [False, True]:
            opt_list += [{'name': 'EXP_ACC_SGD',
                          'alpha_t': alphat,
                          'rho': rho,
                          'is_sls': sls
                          }]

    opt_list += [{'name': 'EXP_ACC_SGD',
                          'alpha_t': "CNST",
                          'rho': rho,
                          'is_sls': False
                          }]


# # #RIT
rho=10
# rho = 1
opt_list += [{'name': 'RIT_SGD',
              'rho': rho
              }]
# #


for benchmark in benchmarks_list:
    EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, opt_list,
                                                                             batch_size=[1], max_epoch=[MAX_EPOCH],
                                                                             runs=RUNS,
                                                                             losses=['squared_loss', 'logistic_loss']))
# 'squared_loss',