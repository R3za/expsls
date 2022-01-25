from dependencies import *

from utils import *
from datasets import *
from objectives import *
import time

# conservative sls
def SLS(x,g1,Di,labels_i,gamma,closure):
  gamma_m=gamma
  j=0
  f1 = closure(x, Di, labels_i, backwards=False)
  g1_normsq = (np.linalg.norm(g1))**2
  func_val=1
  while j<100:
    f2 = closure(x-gamma*g1, Di, labels_i, backwards=False)

    #c=0.5

    if gamma <= (f1-f2)/(0.5*g1_normsq+1e-12):
      break
    j+=1
    gamma=0.7*gamma
  func_val += j
  if j==100:
      gamma=0.7*gamma_m

  return gamma,func_val
