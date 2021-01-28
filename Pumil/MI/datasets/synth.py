#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import MI
from MI import Bag


def synthesize(L, U, T, theta, eta, n = 20, dim = 2, mu_p = 2, var_p = 1, mu_n = -4, var_n = 2):
  """
  Parameters
  ----------
  L     : the number of labeled positive data
  U     : the number of (training) unlabaled data
  T     : the number of test data
  theta : class prior
  eta   : the percentage of positive instances in positive bags
  n     : the number of instances in each bag
  dim   : the dimension of feature space
  
  positive instances are generated from N(mu_p, var_p)
  """
  # make labeled positive set
  labeled_bags = []

  for i in range(L):
    # the number of positive instances in a positive bag
    npos = np.clip(np.random.binomial(n, eta), 1, n)
    # the number of negative instances in a positive bag
    nneg = n - npos

    pins = np.random.normal(mu_p, var_p, size=(npos, dim)).tolist()
    nins = np.random.normal(mu_n, var_n, size=(nneg, dim)).tolist()
    ins = [{'data': x, 'label': 1} for x in pins] + [{'data': x, 'label': -1} for x in nins]
    labeled_bags.append(Bag(ins))

  # make unlabeled set (for training)
  train_unlabeled_bags = []

  # the number of positive bags in training unlabeled set
  trNpos = np.minimum(np.random.binomial(U, theta), U)
  # the number of negative bags in unlabeled set
  trNneg = U - trNpos

  for i in range(trNpos):
    npos = np.clip(np.random.binomial(n, eta), 1, n)
    nneg = n - npos

    pins = np.random.normal(mu_p, var_p, size=(npos, dim)).tolist()
    nins = np.random.normal(mu_n, var_n, size=(nneg, dim)).tolist()
    ins = [{'data': x, 'label': 1} for x in pins] + [{'data': x, 'label': -1} for x in nins]
    bag = Bag(ins)
    train_unlabeled_bags.append(bag)

  for i in range(trNneg):
    nins = np.random.normal(mu_n, var_n, size=(n, dim)).tolist()
    ins = [{'data': x, 'label': -1} for x in nins]
    bag = Bag(ins)
    train_unlabeled_bags.append(bag)

  # make unlabeled set (for test)
  test_bags = []

  # the number of positive bags in training unlabeled set
  teNpos = np.minimum(np.random.binomial(T, theta), T)
  # the number of negative bags in unlabeled set
  teNneg = T - teNpos

  for i in range(teNpos):
    npos = np.clip(np.random.binomial(n, eta), 1, n)
    nneg = n - npos

    pins = np.random.normal(mu_p, var_p, size=(npos, dim)).tolist()
    nins = np.random.normal(mu_n, var_n, size=(nneg, dim)).tolist()
    ins = [{'data': x, 'label': 1} for x in pins] + [{'data': x, 'label': -1} for x in nins]
    bag = Bag(ins)
    test_bags.append(bag)

  for i in range(teNneg):
    nins = np.random.normal(mu_n, var_n, size=(n, dim)).tolist()
    ins = [{'data': x, 'label': -1} for x in nins]
    bag = Bag(ins)
    test_bags.append(bag)

  for bag in train_unlabeled_bags:
    bag.mask()

  train_bags = labeled_bags + train_unlabeled_bags

  metadata = {
      'train_lp': L,
      'train_up': trNpos,
      'train_un': trNneg,
      'test_p'  : teNpos,
      'test_n'  : teNneg,
  }

  random.shuffle(train_bags)
  random.shuffle(test_bags)

  return train_bags, test_bags, metadata
