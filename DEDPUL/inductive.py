from sklearn.base import BaseEstimator
from scipy.interpolate import interp1d
import numpy as np
import torch


class InductiveDEDPUL(BaseEstimator):
    def __init__(self, model, preds, ratios):
        self.model = model
        preds = np.concatenate([np.array([0]), preds, np.array([1])])
        ratios = np.concatenate([np.array([0]), ratios, np.array([1])])
        self.inter = interp1d(preds, ratios)

    def predict(self, X):
        X = np.array(X)
        preds = self.model(torch.as_tensor(X, dtype=torch.float32)).detach().numpy()
        array = self.interpolate(preds)
        return array

    def predict_proba(self, X):
        return self.predict(X)

    def interpolate(self, preds):
        return self.inter(preds)

    def decision_function(self, X):
        p = self.predict(X)
        return np.log(p / (1 - p) + 10 ** -5).reshape(-1, 1)
