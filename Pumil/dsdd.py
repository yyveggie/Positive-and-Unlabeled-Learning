#!/usr/bin/env python
# coding: utf-8

import time
import itertools
import argparse
import MI
import numpy as np

def train_lsdd(data, args):
  widths = [1.0e-2, 1.0e-4, 1.0e-6]
  regs = [1.0, 1.0e-03, 1.0e-06]

  def train(data, width, reg, measure_time = False):
    if measure_time:
      t_start = time.time()

    model = MI.UU.DSDD.train(data, width, reg, args)
    metadata = {'width': width, 'reg': reg}

    if measure_time:
      t_end = time.time()
      print("#  elapsed time = {}".format(t_end - t_start))

    return model, metadata

  model, metadata = train(data, 1.0e-2, 1.0e-2, measure_time=True)

  return model


DATASETS = [
    'synth',
    'musk1',
    'musk2',
    'fox',
    'elephant',
    'tiger',
    ]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="PU-SKC experiment toolkit")

  parser.add_argument('--dataset',
      action   = 'store',
      required = True,
      type     = str,
      choices  = DATASETS,
      help     = 'multiple instance dataset')

  parser.add_argument('--prior',
      action   = 'store',
      default  = 0.1,
      type     = float,
      metavar  = '[0-1]',
      help     = 'class prior (the ratio of positive data)')

  parser.add_argument('--np',
      action   = 'store',
      default  = 20,
      type     = int,
      help     = 'the number of positive data')

  parser.add_argument('--nu',
      action   = 'store',
      default  = 180,
      type     = int,
      help     = 'the number of unlabeled data')

  parser.add_argument('-v', '--verbose',
      action   = 'store_true',
      default  = False,
      help     = 'verbose output')

  parser.add_argument('--aucplot',
      action   = 'store_true',
      default  = False,
      help     = 'output prediction score and true label for AUC plot')

  args = parser.parse_args()

  print("# {}".format('-'*80))
  print("# *** Experimental Setting ***")
  print("#   model                     : DSDD")
  print("# {}".format('-'*80))

  bags_train, bags_test, metadata = MI.datasets.load_dataset(args.dataset, args.prior, args.np, args.nu)
  clf = train_lsdd(bags_train, args)
  MI.print_evaluation_result(clf, bags_test, args)
