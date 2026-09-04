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

    model = MI.UU.LSDD.train(data, width, reg, args)
    metadata = {'width': width, 'reg': reg}

    if measure_time:
      t_end = time.time()
      print("#  elapsed time = {}".format(t_end - t_start))

    return model, metadata

  # cross validation
  best_param = {}
  best_error = np.inf
  if args.verbose:
    print("# *** Cross Validation ***")
  for width, reg in itertools.product(widths, regs):
    errors = []
    for data_train, data_val in MI.cross_validation(data, 5):
      t = MI.UU.LSDD.LSDD(
          np.vstack(MI.extract_bags(data_train, 1)),
          np.vstack(MI.extract_bags(data_train, 0)),
          width, reg)
      e = MI.UU.LSDD.validation_error(data_val, data_train, width, reg, t)
      errors.append(e)

    error = np.mean(errors)

    if args.verbose:
      print("#  width = {:.3e} / reg = {:.3e} / error = {:.3e}".format(width, reg, error))

    if error < best_error:
      best_error = error
      best_param = {'width': width, 'reg': reg}

  if args.verbose:
    print("# {}".format('-'*80))

  model, metadata = train(data, best_param['width'], best_param['reg'], measure_time=True)

  return model, best_param


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
  print("#   model                     : LSDD")
  print("# {}".format('-'*80))

  bags_train, bags_test, metadata = MI.datasets.load_dataset(args.dataset, args.prior, args.np, args.nu)
  clf, best_param = train_lsdd(bags_train, args)
  print("#  width = {:.3e} / reg = {:.3e}".format(best_param['width'], best_param['reg']))
  MI.print_evaluation_result(clf, bags_test, args)
