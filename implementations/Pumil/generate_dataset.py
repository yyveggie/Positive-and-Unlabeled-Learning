#!/usr/bin/env python
# coding: utf-8

import MI
import random
import numpy

def generate(if_name, of_name, dim, n, scale):
  bags = MI.datasets.load_trec9(if_name, dim, n)
  bags = MI.datasets.augment(bags, scale)
  MI.datasets.dump_trec9(of_name, bags)

if __name__ == "__main__":
  random.seed(0)
  numpy.random.seed(0)
  generate('musk1.data',    'datasets/musk1.data',    166, 92, 10)
  generate('musk2.data',    'datasets/musk2.data',    166, 102, 10)
  generate('elephant.data', 'datasets/elephant.data', 230, 200, 5)
  generate('fox.data',      'datasets/fox.data',      230, 200, 5)
  generate('tiger.data',    'datasets/tiger.data',    230, 200, 5)
