'''代码来源地址：https://github.com/trokas/pu_learning/blob/master/Experiments.ipynb'''

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.datasets.samples_generator import make_blobs, make_circles
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from baggingPU import BaggingClassifierPU
from pu_learning import spies
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
logging.debug("Initiating logger...")
plt.rcParams['figure.figsize'] = 9, 7
plt.rcParams['font.size'] = 14


### Create the data set BLOBS ###


X, y = make_blobs(
    n_samples = 6000,
    centers = [[1,5], [5,1], [0,0], [6,6]]
)
y = (y > 1).astype(int)  # Convert the original labels [0,1,2,3] to [0,1]

# # Create the data set CIRCLES
# X, y = make_circles(
#     n_samples = 6000, noise = 0.1,
#     shuffle = True, factor = .65
# )

X = pd.DataFrame(X, columns = ['feature1', 'feature2'])
y = pd.Series(y)

# Check the contents of the set
print('%d data points and %d features' % (X.shape))
print('%d positive out of %d total' % (sum(y), len(y)))

# Keep the original targets safe for later
y_orig = y.copy()

# Unlabel a certain number of data points
hidden_size = 2700
y.loc[np.random.choice(y[y == 1].index, replace = False, size = hidden_size)] = 0

# Check the new contents of the set
print('%d positive out of %d total' % (sum(y), len(y)))

# Plot the data set, as the models will see it
plt.scatter(X[y==0].feature1, X[y==0].feature2, c='k', marker='.', linewidth=1, s=10, alpha=0.5, label='Unlabeled')
plt.scatter(X[y==1].feature1, X[y==1].feature2, c='b', marker='o', linewidth=0, s=50, alpha=0.5, label='Positive')
plt.legend()
plt.title('Data set (as seen by the classifiers)')
plt.show()


bc = BaggingClassifierPU(
    DecisionTreeClassifier(),
    n_estimators = 1000,  # 1000 trees as usual
    max_samples = sum(y), # Balance the positives and unlabeled in each bag
    n_jobs = -1           # Use all cores
)
bc.fit(X, y)

# Store the scores assigned by this approach
results = pd.DataFrame({
    'truth'      : y_orig,   # The true labels
    'label'      : y,        # The labels to be shown to models in experiment
}, columns = ['truth', 'label'])
results['output_bag_tree'] = bc.oob_decision_function_[:,1]

# Visualize this approach's results
plt.scatter(
    X[y==0].feature1, X[y==0].feature2,
    c = results[y==0].output_bag_tree, linewidth = 0, s = 50, alpha = 0.5,
    cmap = 'jet_r'
)
plt.colorbar(label='Scores given to unlabeled points')
plt.title(r'Using ${\tt BaggingClassifierPU}$')
plt.show()