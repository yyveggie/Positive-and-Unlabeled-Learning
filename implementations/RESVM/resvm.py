#! /usr/bin/env python

# Copyright (c) 2014 Marc Claesen
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from copy import deepcopy
from random import shuffle
import sys, traceback, subprocess, itertools

####################################################
# DEFAULT CONFIGURATION & PARAMETER PARSING
####################################################

# Default configuration
defaults = {
    'esvm-prefix': '',
    'esvm-suffix': '',
    'work-dir': '/tmp/',
    'sep': ',',
    'pos': '1',
    'k': 10,
    'gamma': 1.0,
    'coef0': 0.0,
    'degree': 3,
    'kfun': 0,
    'T': 0.5,
    'niter': 1
    }

class Config:
    def __init__(self, defaults):
        self.dict = defaults
    def __str__(self):
        return str(self.dict)
    def __contains__(self, key):
        return key in self.dict
    def __setitem__(self, key, value):
        self.dict[key]=value
    def __getitem__(self, key):
        val = self.dict.get(key)
        if val == None:
            traceback.print_stack()
            sys.exit("\nError: parameter \"" + key + "\" is not configured.")
        return val            
    def update(self, newconfig):
        self.dict.update(newconfig)
    def get(self):
        return self.dict
        
# Read command line configuration and set global config
cfg = Config(defaults)
# http://stackoverflow.com/a/4260304/2148672
cli_args = dict([arg.split('=', 1) if len(arg.split('=', 1)) == 2 else [arg, ''] for arg in sys.argv[1:]])
cfg.update(cli_args)

debug = "debug" in cfg

####################################################
# PRINT HELP
####################################################

if "help" in cfg or "--help" in cfg:

    if "cross-validate" in cfg:
        print(
"""k-fold cross-validation for a specific parameter tuple.

This function will print out the resulting score (higher is better):
    score = recall^2 * num_predictions / num_positive_predictions

Command line arguments:
data        : training data file.
k           : number of folds in cross-validation
folds       : optional file containing predefined cross-validation folds.
pos         : positive label (default 1).
nmodels     : number of base models to include in the ensemble.
npos        : number of positive instances to use in base model training.
nunl        : number of unlabeled instances to use in base model training.
c           : global misclassification penalty in base models.
wpos        : relative positive misclassification penalty in base models.

Kernel parameters of base models:
kfun        : set type of kernel function (default 0)
              0 -- linear: u'*v
              1 -- polynomial: (gamma*u'*v + coef0)^degree
              2 -- radial basis function: exp(-gamma*|u-v|^2)
              3 -- sigmoid: tanh(gamma*u'*v + coef0)
              4 -- precomputed: data file row = <label> <kernel row>
coef0       : set coef0 in kernel function (default 0.0)
degree      : set degree in kernel function (default 3)
gamma       : set gamma in kernel function  (default 1.0)""")

    elif "train" in cfg:
        print(
"""Train an RESVM model with given parameters.

Command line arguments:
data        : training data file.
model       : filename for the resulting model.
pos         : positive label (default 1).
nmodels     : number of base models to include in the ensemble.
npos        : number of positive instances to use in base model training.
nunl        : number of unlabeled instances to use in base model training.
c           : global misclassification penalty in base models.
wpos        : relative positive misclassification penalty in base models.

Kernel parameters of base models:
kfun        : set type of kernel function (default 0)
              0 -- linear: u'*v
              1 -- polynomial: (gamma*u'*v + coef0)^degree
              2 -- radial basis function: exp(-gamma*|u-v|^2)
              3 -- sigmoid: tanh(gamma*u'*v + coef0)
              4 -- precomputed: data file row = <label> <kernel row>
coef0       : set coef0 in kernel function (default 0.0)
degree      : set degree in kernel function (default 3)
gamma       : set gamma in kernel function  (default 1.0)""")

    elif "predict" in cfg:
        print(
"""Predict a given test set with an RESVM model.

The resulting prediction file contains one line per test instance.
Each line contains the predicted label and decision value.

Command line arguments:
data        : test data file.
model       : model file.
predictions : prediction file.""")

    elif "grid-search" in cfg:
        print(
"""Performs a grid-search to find the optimal parameter tuple.
Optionally trains a model using the optimal parameters.

Parameter tuples are evaluated using repeated k-fold cross-validation 
The following score function is used (higher is better):
    score = recall^2 * num_predictions / num_positive_predictions
    
To fix a specific parameter x to a constant value v, use <x>=<v>.
To define a set of grid points for x, use <x>="<v 1> <v 2> ... <v n>"

Command line arguments:
data        : training data file.
k           : number of folds in cross-validation
folds       : optional file containing predefined cross-validation folds.
niter       : number of cross-validation iterations (default 1)
pos         : positive label (default 1).
nmodels     : number of base models to include in the ensemble.
npos        : number of positive instances to use in base model training.
nunl        : number of unlabeled instances to use in base model training.
c           : global misclassification penalty in base models.
wpos        : relative positive misclassification penalty in base models.
full        : produce full output of all parameter tuples (optional flag)
model       : trains model with best parameters and saves to this file.

Kernel parameters of base models:
kfun        : set type of kernel function (default 0)
              0 -- linear: u'*v
              1 -- polynomial: (gamma*u'*v + coef0)^degree
              2 -- radial basis function: exp(-gamma*|u-v|^2)
              3 -- sigmoid: tanh(gamma*u'*v + coef0)
              4 -- precomputed: data file row = <label> <kernel row>
coef0       : set coef0 in kernel function (default 0.0)
degree      : set degree in kernel function (default 3)
gamma       : set gamma in kernel function  (default 1.0)""")
        
    else:
        print(
"""Script for the Robust Ensemble of SVMs (RESVM) method.
For algorithm details, please refer to:
    ftp://ftp.esat.kuleuven.be/pub/SISTA/claesenm/reports/14-22.pdf
For additional information, updates or bug reports, please refer to:
    https://github.com/claesenm/resvm.
If you use this software in research, please cite the associated paper.

This script allows you to perform the following tasks:
train          : train an RESVM model.
predict        : predict with an existing RESVM model.
cross-validate : perform k-fold cross-validation for a parameter tuple.
grid-search    : select an optimal parameter tuple (+optionally train model).

To perform a specific task, please call "./resvm.py <task> options".
An overview of task specific arguments is shown using "./resvm.py help <task>".

Training and testing data files must be provided in LIBSVM format, e.g.
<label> <index 1>:<value 1> <index 2>:<value 2> ... <index n>:<value n>

This script generates intermediate files in a folder of your choosing. 
The EnsembleSVM library is used as a back-end and must be installed.
EnsembleSVM is freely available at: http://esat.kuleuven.be/stadius/ensemblesvm/.

General options related to EnsembleSVM, usable in all tasks listed above:
esvm-prefix : prefix used in all EnsembleSVM executables (default '').
esvm-suffix : suffix used in all EnsembleSVM executables (default '').
work-dir    : working directory to use for intermediate files (default '/tmp/').
noclean     : retain intermediate files (flag, default: remove intermediates).""")
        
    sys.exit(0)

# check if the task is properly set (e.g. "train", "predict" or "cross-validate")
num_tasks = sum(["cross-validate" in cfg, "train" in cfg, "predict" in cfg, "grid-search" in cfg])
if num_tasks > 1:
    sys.exit("Error: multiple tasks specified. Please choose one task: train, predict, cross-validate or grid-search.")
elif num_tasks == 0:
    sys.exit("Error: no task specified. Please choose one task: train, predict, cross-validate or grid-search.")
    
####################################################
# FUNCTIONS TO READ TRAINING LABELS
####################################################

def read_labels(filename, delimiter):
    """Reads the labels from filename with given column delimiter."""
    labels = []
    with open(filename,'r') as f:
        for line in f:
            cols = line.split(delimiter)
            labels.append(cols[0])
    if len(labels)==0:
        sys.exit("Error: " + filename + " is empty.")
    return labels

def read_binary_labels(filename, delimiter, positive):
    """Reads the binary labels from filename with given column delimiter.
    
    All labels not equal to positive_label are treated as negative.
    """
    labels = read_labels(filename, delimiter)
    binary_labels = [x==positive for x in labels]
    sum_labels = sum(binary_labels)    
    if sum_labels == 0:
        sys.exit("Error: " + filename + " contains only negative labels.")
    elif sum_labels == len(binary_labels):
        sys.exit("Error: " + filename + " contains only positive labels.")
    return binary_labels
    
####################################################
# CONFIGURATION OF EnsembleSVM EXECUTABLES
####################################################

def execute(command):
    str_command = [str(x) for x in command]
    if debug:
        print(" ".join(str_command))
    try:
        subprocess.check_output(str_command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e: # FIXME: stderr not read properly
        command = ' '.join(e.cmd)
        traceback.print_stack()
        sys.exit("ERROR running the following EnsembleSVM command: \n" + command + 
            "\n\nRun this command manually to obtain EnsembleSVM's error report.")

esvmtrain = cfg["esvm-prefix"] + "esvm-train" + cfg["esvm-suffix"]
bootstrap = cfg["esvm-prefix"] + "bootstrap" + cfg["esvm-suffix"]
esvmpredict = cfg["esvm-prefix"] + "esvm-predict" + cfg["esvm-suffix"]

####################################################
# PU LEARNING CROSS-VALIDATION METRIC
####################################################

def scorefun(true_labels, predictions):
    num_pos = sum(true_labels)
    p_pred_pos = float(sum(predictions))/len(predictions)
    if p_pred_pos == 0:
        return 0.0
    tp = sum([all(x) for x in zip(true_labels,predictions)])
    return tp*tp/(num_pos*num_pos*p_pred_pos)
    
####################################################
# GENERATE BOOTSTRAP FILE
####################################################

def resvm_bootstrap(cfg, xvalfile="", fold=0):

    # bootstrap
    bootstraplist = [bootstrap,
        "-data",cfg["data"],
        "-labels",cfg["pos"],"-" + cfg["pos"],"-posvall",
        "-nboot",cfg["nmodels"],
        "-npos",cfg["npos"],
        "-nneg",cfg["nunl"]
        ]

    # bootstrap file name
    bootstrapfile = ""
    if len(xvalfile) > 0:
        bootstraplist.extend([
            "-xval",xvalfile,       
            "-xvalfold",str(fold)
            ])
        bootstrapfile = cfg["work-dir"] + "bootstrap-" + str(fold) + ".txt"
    else:
        bootstrapfile = cfg["work-dir"] + "bootstrap.txt"

    # do bootstrap    
    bootstraplist.extend(["-o",bootstrapfile])
    execute(bootstraplist)

    return bootstrapfile

####################################################
# GENERATE RESAMPLES AND TRAIN MODEL
####################################################

def resvm_train(cfg, bootstrapfile="", xvalfile="", fold=0):
    
    # bootstrap if necessary
    do_bootstrap = len(bootstrapfile) == 0
    if do_bootstrap:
        bootstrapfile = resvm_bootstrap(cfg, xvalfile, fold)
    
    # model name
    if "model" not in cfg:
        if len(xvalfile) > 0:
            cfg.update({"model":cfg["work-dir"] + "model-" + str(fold) + ".txt"})
        else:
            cfg.update({"model":cfg["work-dir"] + "model.txt"})
    
    # train
    pospen = float(cfg["c"])*float(cfg["wpos"])*float(cfg["nunl"])/float(cfg["npos"])
    execute([esvmtrain,
        "-data",cfg["data"],
        "-bootstrap",bootstrapfile,
        "-o",cfg["model"],
        "-labels",cfg["pos"],"-" + cfg["pos"],"-posvall",
        "-nmodels",cfg["nmodels"],
        "-pospen",str(pospen),
        "-negpen",cfg["c"],
        "-kfun",cfg["kfun"],
        "-gamma",cfg["gamma"],
        "-degree",cfg["degree"],
        "-coef0",cfg["coef0"]
        ])
        
    # clean up if necessary
    if not "noclean" in cfg and do_bootstrap:
        execute(["rm",bootstrapfile])
        
    return cfg["model"]

####################################################
# PREDICT WITH SPECIFIED MODEL
####################################################
        
def resvm_predict(cfg, xvalfile="", fold=0):
    predictlist = [esvmpredict,
        "-data",cfg["data"],
        "-model",cfg["model"],
        "-base","-labeled"
        ]
        
    predictionfile = cfg["work-dir"] + "predictions.txt"
    if len(xvalfile) > 0:
        predictlist.extend([
            "-xval",xvalfile,
            "-xvalfold",str(fold)
            ])
        predictionfile = cfg["work-dir"] + "predictions-" + str(fold) + ".txt"
    
    predictlist.extend(["-o",predictionfile])
    execute(predictlist)
    
    i = 0
    decision_values = []
    labels = []  
    with open(predictionfile,'r') as f:
        for line in f:
            cols = line.split(" ")
            labels.append(cols[0])
            decision_values.append(float(cols[1]))
            if decision_values[i] == 0:
                decision_values[i] = sum([float(x) for x in cols[2:]])
            elif decision_values[i] == 1:
                decision_values[i] = 1 + sum([float(x) for x in cols[2:]])
            i = i + 1

    if not "noclean" in cfg:
        execute(["rm",predictionfile])
    
    return [labels, decision_values]
    
####################################################
# PERFORM CROSS-VALIDATION FOR GIVEN PARAMETER TUPLE
####################################################

def generate_folds(cfg, binary_labels):
    num_folds = int(cfg["k"])
    xvalfile = cfg["work-dir"] + "xval.txt"
    
    # find indices of positives and negatives
    pos_ind = [x for x in range(len(binary_labels)) if binary_labels[x]]
    neg_ind = [x for x in range(len(binary_labels)) if not binary_labels[x]]
    
    shuffle(pos_ind)
    shuffle(neg_ind)
    
    folds = [0]*len(binary_labels)
    pos_fold_size = len(pos_ind) // num_folds
    neg_fold_size = len(neg_ind) // num_folds
        
    for i in range(num_folds-1):
        for j in pos_ind[i*pos_fold_size:(i+1)*pos_fold_size]:
            folds[j] = i+1
        for j in neg_ind[i*neg_fold_size:(i+1)*neg_fold_size]:
            folds[j] = i+1
    
    for j in pos_ind[(num_folds-1)*pos_fold_size:]:
        folds[j] = num_folds
    for j in neg_ind[(num_folds-1)*neg_fold_size:]:
        folds[j] = num_folds
        
    with open(xvalfile, 'w') as f:
        f.write("\n".join([str(x) for x in folds]))
    
    return xvalfile
    
def cross_validate(cfg, binary_labels=[]):

    xvalfile = ''
    num_folds = 0
    if len(binary_labels)==0:
        binary_labels = read_binary_labels(cfg["data"], " ", cfg["pos"])

    if "folds" in cfg:
        # read existing cross-validation file
        xvalfile=cfg["folds"]

    else:
        xvalfile=generate_folds(cfg, binary_labels)
         
    folds = []
    with open(xvalfile,'r') as f:
        folds = [int(x) for x in f];
    num_folds = max(folds)

    score = 0.0
    for i in range(num_folds):        
        fold=i+1
        true_labels = [binary_labels[x] for x in range(len(binary_labels)) if folds[x]==fold]
        
        xvalcfg = cfg
        xvalcfg.update({"model":cfg["work-dir"] + "model-" + str(fold) + ".txt"})
        
        if "bootstraps" in cfg:
            resvm_train(xvalcfg, xvalfile=xvalfile, fold=fold, bootstrapfile=cfg["bootstraps"][i])    
        else:        
            resvm_train(xvalcfg, xvalfile=xvalfile, fold=fold)
        
        decision_values = resvm_predict(xvalcfg, xvalfile, fold)[1]
        score = score + scorefun(true_labels, [x > 0.5 for x in decision_values])
     
        if not "noclean" in cfg:
            execute(["rm", xvalcfg["model"]])
            
    return score/num_folds
        
####################################################
# PERFORM REQUIRED TASK
####################################################

if "cross-validate" in cfg:

    score = cross_validate(cfg)
    print(score)
        
elif "train" in cfg:
    
    resvm_train(cfg)
    
elif "predict" in cfg:

    labels, decision_values = resvm_predict(cfg)
    threshold = float(cfg["T"])                 # FIXME: use given threshold

    output = open(cfg["predictions"],'w')
    for (label, dv) in zip(labels,decision_values):
        output.write(label + " " + str(dv) + "\n")
    output.close()
    
elif "grid-search" in cfg: 

    binary_labels = read_binary_labels(cfg["data"], " ", cfg["pos"])

    # set tuning parameter grid    
    cfg["nmodels"] = [int(x) for x in str(cfg["nmodels"]).split()]
    cfg["npos"]    = [int(x) for x in str(cfg["npos"]).split()]
    cfg["nunl"]    = [int(x) for x in str(cfg["nunl"]).split()]
    cfg["degree"]  = [int(x) for x in str(cfg["degree"]).split()]
    cfg["c"]       = [float(x) for x in str(cfg["c"]).split()]
    cfg["wpos"]    = [float(x) for x in str(cfg["wpos"]).split()]
    cfg["coef0"]   = [float(x) for x in str(cfg["coef0"]).split()]
    cfg["gamma"]   = [float(x) for x in str(cfg["gamma"]).split()]

    grid = [cfg["nmodels"], cfg["npos"], cfg["nunl"], 
        cfg["c"], cfg["wpos"], 
        cfg["coef0"], cfg["degree"], cfg["gamma"]]        
    
    num_scores = sum(1 for _ in itertools.product(*grid))
    scores = []
        
    for i in range(int(cfg["niter"])):
        scores.append([])
        xvalfile=""
        if "folds" in  cfg:
            xvalfile = cfg["folds"]
        else:
            xvalfile = generate_folds(cfg, binary_labels)
        last_nmodels, last_npos, last_nunl = -1, -1, -1        
        
        for pars in itertools.product(*grid):
        
            nmodels, npos, nunl, c, wpos, coef0, degree, gamma = pars        
            if nmodels != last_nmodels or npos != last_npos or nunl != last_nunl:
                last_nmodels, last_npos, last_nunl = nmodels, npos, nunl
            
            config = deepcopy(cfg)
            config.update({"nmodels":nmodels, "npos":npos, "nunl":nunl, 
                "bootstraps":[""]*int(cfg["k"]),
                "c":c, "wpos": wpos,
                "coef0":coef0, "degree":degree, "gamma":gamma
                })
            for numboot in range(int(cfg["k"])):
                config["bootstraps"][numboot] = resvm_bootstrap(config, xvalfile, numboot+1)
            
            scores[i].append(cross_validate(config))

    zipped = zip(*scores)
    flattened_scores = [sum(i)/int(cfg["niter"]) for i in zipped]
    
    if "full" in cfg:
        if int(cfg["niter"]) == 1:
            print("(nmodels, npos, nunl, c, wpos, coef0, degree, gamma), score")
            for line in zip(itertools.product(*grid), flattened_scores):
                print(", ".join([str(x) for x in itertools.chain(line)]))        
        else:
            title = "(nmodels, npos, nunl, c, wpos, coef0, degree, gamma), overall_score"
            for i in range(int(cfg["niter"])):
                title = title + ", iter_" + str(i+1)
            print(title)
            for line in zip(itertools.product(*grid), flattened_scores, *scores):
                print(", ".join([str(x) for x in itertools.chain(line)]))
                
        print("")
            
    best_score = max(flattened_scores)
    best_score_index = flattened_scores.index(best_score)
    
    i = 0
    for pars in itertools.product(*grid):
        if i==best_score_index:            
            cfg.update({"nmodels":pars[0], "npos":pars[1], "nunl":pars[2], 
                "c": pars[3], "wpos":pars[4],
                "coef0":pars[5], "degree":pars[6], "gamma":pars[7]
                })            
            break
        i = i+1

    parameters={"nmodels":cfg["nmodels"], "npos":cfg["npos"], "nunl":cfg["nunl"],
        "c":cfg["c"], "wpos":cfg["wpos"], "coef0":cfg["coef0"], "degree":cfg["degree"], "gamma":cfg["gamma"]}
        
    print("*** optimal hyperparameters ***")
    for k,v in sorted(parameters.iteritems()):
        print(str(k) + " = " + str(v))
        
    if "model" in cfg:
        resvm_train(cfg)
        
