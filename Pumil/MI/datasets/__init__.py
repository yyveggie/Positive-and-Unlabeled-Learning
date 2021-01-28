#!/usr/bin/env python
# coding: utf-8

from MI.datasets.synth import synthesize
from MI.datasets.trec9 import load_trec9
from MI.datasets.trec9 import dump_trec9

from MI import PU

import random
import copy
import numpy as np


def load_dataset(dataset, cprior, np, nu, dim = None):
  if dataset == 'musk1':
    bags = load_trec9('datasets/musk1.data', 166, 92*10)

    if dim is not None:
      bags = diminish(bags, dim)

    bags_train, bags_test, metadata = PU.prepare(
        bags, cprior,
        L = np, U = nu, T = 200)

  elif dataset == 'musk2':
    bags = load_trec9('datasets/musk2.data', 166, 102*10)

    if dim is not None:
      bags = diminish(bags, dim)

    bags_train, bags_test, metadata = PU.prepare(
        bags, cprior,
        L = np, U = nu, T = 200)

  elif dataset == 'elephant':
    bags = load_trec9('datasets/elephant_10x.data', 230, 200*10)

    if dim is not None:
      bags = diminish(bags, dim)

    bags_train, bags_test, metadata = PU.prepare(
        bags, cprior,
        L = np, U = nu, T = 200)

  elif dataset == 'fox':
    bags = load_trec9('datasets/fox_10x.data', 230, 200*10)

    if dim is not None:
      bags = diminish(bags, dim)

    bags_train, bags_test, metadata = PU.prepare(
        bags, cprior,
        L = np, U = nu, T = 200)

  elif dataset == 'tiger':
    bags = load_trec9('datasets/tiger_10x.data', 230, 200*10)

    if dim is not None:
      bags = diminish(bags, dim)

    bags_train, bags_test, metadata = PU.prepare(
        bags, cprior,
        L = np, U = nu, T = 200)

  elif dataset == 'synth':
    bags_train, bags_test, metadata = synthesize(
        L = np, U = nu, T = 200,
        theta = cprior,
        eta = 0.2,  # the percentage of positive instances in a positive bag
        n = 4)

  n_class = len(list(set([B.y for B in bags_test])))

  if n_class == 2:
    return bags_train, bags_test, metadata
  else:
    # to avoid ROC-AUC error, ensure n_class = 2
    return load_dataset(dataset, cprior)


def augment(bags, ratio):
  N = len(bags)

  for i in np.random.choice(range(N), int(N * (ratio - 1))):
    aug = copy.deepcopy(bags[i])
    aug.add_noise()
    bags.append(aug)

  random.shuffle(bags)
  return bags


def diminish(bags, dim):
  """
  Dimension reduction for the original dataset.
  This method is expected to be used by puMIL, since WKDE (weighted kernel density estimation) does not work well on super-high dimension data.
  """
  ins = np.vstack([B.data() for B in bags])

  from sklearn.decomposition import PCA
  pca = PCA(n_components = dim)
  pca.fit(ins)

  for bag in bags:
    bag.pca_reduction(pca)

  return bags
