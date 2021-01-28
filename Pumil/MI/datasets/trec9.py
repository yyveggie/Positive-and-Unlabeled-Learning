#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import MI
from MI import Bag


def load_trec9(data_file, dim, bag_count):
  """
  Load SVM-light-extended formatted file and convert into the following form:

  [ Bags( [ {'data': x, 'label': y}, ... ] ),
    Bags( [ {'data': x, 'label': y}, ... ] ),
                      :
    Bags( [ {'data': x, 'label': y}, ... ] )]
  """
  bags = []

  with open(data_file) as f:
    for l in f.readlines():
      if l[0] == '#':
        continue

      ss = l.strip().split(' ')
      x = np.zeros(dim)

      for s in ss[1:]:
        i, xi = s.split(':')
        i     = int(i) - 1
        xi    = float(xi)
        x[i]  = xi

      _, bag_id, y = ss[0].split(':')
      bags.append({'x': x, 'y': int(y), 'bag_id': int(bag_id)})

  return [Bag(list(map(
    lambda X: {'data': X['x'], 'label': X['y']},
    list(filter(lambda X: X['bag_id'] == i, bags)))))
    for i in range(bag_count)]


def dump_trec9(data_file, bags):
  """
  Dump SVM-light-extended formatted file.

  0:bag_id:label 1:dim1 2:dim2 3:dim3 ...
  1:bag_id:label 1:dim1 2:dim2 3:dim3 ...
  2:bag_id:label 1:dim1 2:dim2 3:dim3 ...
  ...
  """
  with open(data_file, 'w') as f:
    total_id = 0

    for bag_id, bag in enumerate(bags):
      for inst in bag.instances:
        f.write("{}:{}:{} ".format(total_id, bag_id, inst['label']))
        for i, v in enumerate(inst['data']):
          if v != 0:
            f.write("{}:{} ".format(i, v))
        f.write("\n")
        total_id += 1
