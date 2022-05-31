# Towards Noise-adaptive, Problem-adaptive Stochastic Gradient Descent
https://arxiv.org/abs/2110.11442
## Experiments

Run the experiments using the command below:

``
python trainval.py -e $exp_{BENCHMARK} -sb ${SAVEDIR_BASE} -r 1
``

with the placeholders defined as follows.

**{BENCHMARK}**

Defines the dataset and regularization constant for the experiments

- `mushrooms`, `ijcnn`, `rcv1` 

**{SAVEDIR_BASE}**

Defines the absolute path to where the results will be saved.
