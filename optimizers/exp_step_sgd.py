from dependencies import *

from utils import *
from datasets import *
from objectives import *
import time

from optimizers.sls import SLS as SLS



def Exp_SGD(score_list, closure, D, labels,  batch_size=1,max_epoch=100, gamma=None, alpha_t="CNST",
         x0=None, is_sls=False, verbose=True, D_test=None, labels_test=None, log_idx=1000):
    """
        SGD with fixed step size for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """

    n = D.shape[0]
    d = D.shape[1]
 
    m = int(n/batch_size)

    T=m*max_epoch
    alpha=1
    if alpha_t!="CNST":
         alpha=(1./T)**(1./T)

    if is_sls:
        gamma=2

    if x0 is None:
        x = np.zeros(d)
        x0 = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    num_grad_evals = 0
    num_func_evals = 0

    loss, full_grad = closure(x, D, labels)

    if verbose:
        output = 'Epoch.: %d, Grad. norm: %.2e' % \
                 (0, np.linalg.norm(full_grad))
        output += ', Func. value: %e' % loss
        output += ', Step size: %e' % gamma
        output += ', Num gradient evaluations/n: %f' % (num_grad_evals / n)
        output += ', Num function evaluations/n: %f' % (num_func_evals / n)
        print(output)

    score_dict = {"itr": 0}
    score_dict["n_func_evals"] = num_func_evals
    score_dict["n_grad_evals"] = num_grad_evals
    score_dict["n_grad_evals_normalized"] = num_grad_evals / n
    score_dict["train_loss"] = loss
    score_dict["grad_norm"] = np.linalg.norm(full_grad)
    score_dict["train_accuracy"] = accuracy(x, D, labels)
    if D_test is not None:
        test_loss = closure(x, D_test, labels_test, backwards=False)
        score_dict["test_loss"] = test_loss
        score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
    score_list += [score_dict]

    t=0
    for k in range(max_epoch):        
        t_start = time.time()


        
        if np.linalg.norm(full_grad) <= 1e-12:
            break
        if np.linalg.norm(full_grad) > 1e10:
            break
        if np.isnan(full_grad).any():
            break
                   
        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):

            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]

            # compute the loss, gradients
            loss, gk = closure(x, Di, labels_i)

            lr=gamma*(alpha**t)
            x -= lr * gk
            num_grad_evals = num_grad_evals + batch_size

            if is_sls:
                gamma,fv=SLS(x+lr*gk,gk,Di,labels_i,gamma,closure)
                num_func_evals+=fv
                # num_grad_evals = num_grad_evals + batch_size
                # lr=lr*alpha**(t+1)

            if (num_grad_evals) % log_idx == 0 or (num_grad_evals) % n== 0:
                t_end = time.time()

                loss, full_grad = closure(x, D, labels)

                if verbose:
                    output = 'Epoch.: %d, Grad. norm: %.2e' % \
                             (int(t*batch_size/n), np.linalg .norm(full_grad))
                    output += ', Func. value: %e' % loss
                    output += ', Step size: %e' % gamma
                    output += ', Num gradient evaluations/%d: %f' % (log_idx,num_grad_evals / log_idx)
                    output += ', Num function evaluations/%d: %f' % (log_idx,num_func_evals / n)
                    print(output)

                score_dict = {"itr": (t+1)}
                score_dict["time"]=t_end-t_start
                score_dict["n_func_evals"] = num_func_evals
                score_dict["n_grad_evals"] = num_grad_evals
                if batch_size==n:
                    score_dict["n_grad_evals_normalized"] = num_grad_evals / n
                else:
                    score_dict["n_grad_evals_normalized"] = num_grad_evals / log_idx

                score_dict["train_loss"] = loss
                score_dict["grad_norm"] = np.linalg.norm(full_grad)
                score_dict["train_accuracy"] = accuracy(x, D, labels)
                if D_test is not None:
                    test_loss = closure(x, D_test, labels_test, backwards=False)
                    score_dict["test_loss"] = test_loss
                    score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
                score_list += [score_dict]
                if np.linalg.norm(full_grad) <= 1e-12:
                    break
                if np.linalg.norm(full_grad) > 1e10:
                    break
                if np.isnan(full_grad).any():
                    break
                t_start=time.time()
            t += 1

    return score_list