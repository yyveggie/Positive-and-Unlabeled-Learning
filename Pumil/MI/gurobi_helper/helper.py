#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("/home/local/bin/gurobi650/linux64/lib/python3.4_utf32/gurobipy")
import gurobipy

def dot(x, y):
  return gurobipy.quicksum([_x * _y for (_x, _y) in zip(x, y)])

def mvmul(A, x):
  return [dot(_A, x) for _A in A]

def quadform(A, x):
  return dot(x, mvmul(A, x))
