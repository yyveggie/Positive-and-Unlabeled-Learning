from sarpu.labeling_mechanisms import label_data
from sarpu.paths_and_names import *
from sarpu.experiments import *

import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

data_folder = "../Data/"
results_folder = "../Results/"

data_name = "mushroom_extclustering"
propensity_attributes = [111, 112, 113, 114]
propensity_attributes_signs = [1, 1, 1, 1]
settings = "lr._.lr._.0-111"
labeling_model_type = "simple_0.2_0.8"

labeling = 0
partition = 1

nb_assignments = 5
nb_labelings = 5


relabel_data = False
rerun_experiments = False


pu_methods = ["supervised", "negative", "sar-e", "scar-c", "sar-em", "scar-km2", "scar-tice"]

# Generate PU data with labeling mechanism

labeling_model = label_data(
    data_folder,
    data_name,
    labeling_model_type,
    propensity_attributes,
    propensity_attributes_signs,
    nb_assignments,
    relabel_data=relabel_data
)

# Train and Evaluate models

for pu_method in pu_methods:
    train_and_evaluate(
        data_folder,
        results_folder,
        data_name,
        labeling_model,
        labeling,
        partition,
        settings,
        pu_method,
        rerun_experiments=rerun_experiments
    )

experiment_path = experiment_result_folder_path(
    results_folder,
    data_name,
    labeling_model,
    labeling,
    partition,
    settings
)

# Evaluate setting
# Labeling Mechanism Properties
# Load dataset

x_path = data_path(data_folder,data_name)
y_path = classlabels_path(data_folder,data_name)
s_path = propensity_labeling_path(data_folder, data_name, labeling_model, labeling)
e_path = propensity_scores_path(data_folder, data_name, labeling_model)
x, y, s, e = read_data((x_path,y_path,s_path,e_path))
model_path = experiment_classifier_path(results_folder, data_name, labeling_model, labeling, partition, settings, "supervised")

y_pred = pickle.load(open(model_path, 'rb')).predict_proba(x)

