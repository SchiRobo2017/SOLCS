# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.7rc2
# ---

# %%

# coding: utf-8

# %%


class XCSConfig:
    k=2 #最初は3 #ビット長：k+2^k
    N=1000#600 maximam size of the population
    max_iterations = 5000 #500000#3000
    max_experiments = 10
    
    alpha = 0.1 #used for calculating the fitness of a classifier
    beta = 0.2 #learning rate for p
    gamma = 0.71 #discount factor in updating classifier predictions 
    delta = 0.1 #it specifies tha fraction of the mean fittness in [P] below which the finess of a classifier may be considered in its probability of deletion
    myu = 0.04 #it specifies the probability of mutating an allele(対立遺伝子) in the offspring
    nyu = 5  #used for calculating the fitness of a classifier
    chi = 0.8 #the prpbability of applying crossover in the GA
    
    epsilon_0 = 10 #used for calculating the fitness of a classifier
    
    theta_ga = 25 #the GA threshold. GA is applied in a set when the avarage time since the last GA in the set is greater than theta_ga
    theta_del = 20 #deletion threshold
    theta_sub = 20 #subsumption threshold
    theta_mna = 2 #it specifies minimal number of actions that must be present in a maatch set [M] or else covering will occur
    
    p_sharp = 0.33 #0.65 #the probability of using # in attribute in C when covoering
    p_explr = 1.0 #it specifies the probability during action selection of choosing the action uniform randomly
    
    doGASubsumption = True
    doActionSetSubsumption = True


# %%


XCSConfig = XCSConfig()
conf = XCSConfig

