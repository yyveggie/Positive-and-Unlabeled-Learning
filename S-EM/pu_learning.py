'''代码来源网站：https://github.com/trokas/pu_learning/blob/master/pu_learning.py'''


import numpy as np
import logging

__author__ = 'trokas'
__version__ = '0.1'

logger = logging.getLogger(__name__)

class spies:
    """
    PU spies method, based on Liu, Bing, et al. "Partially supervised classification of
    text documents." ICML. Vol. 2. 2002.
    """
    def __init__(self, first_model, second_model):
        """
        Any two models which have methods fit, predict and predict_proba can be passed,
        for example" `spies(XGBClassifier(), XGBClassifier())`
        """
        self.first_model = first_model
        self.second_model = second_model
        
    def fit(self, X, y, spie_rate=0.2, spie_tolerance=0.05):
        """
        Trains models using spies method using training set (X, y).

        Parameters
        ----------
        X : {array-like} of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (1 for positive, 0 for unlabeled).
            
        spie_rate : {float} = 0.2 (default)
            Determines percentage of spies which will be included when training first model.
            
        spie_tolerance : {float} = 0.05 (default)
            Determines tolerated percentage of spies which can come from the first model.
            Using this tolerance threshold is chosen which splits dataset into Likely negative
            and unlabeled groups.

        Returns
        -------
        self : object
            Returns self.
        """
        np.random.seed(42)
        # Step 1. Infuse spies
        spie_mask = np.random.random(y.sum()) < spie_rate
        # Unknown mix + spies
        MS = np.vstack([X[y == 0], X[y == 1][spie_mask]])
        MS_spies = np.hstack([np.zeros((y == 0).sum()), np.ones(spie_mask.sum())])
        # Positive with spies removed
        P = X[y == 1][~spie_mask].values
        # Combo
        MSP = np.vstack([MS, P])
        # Labels
        MSP_y = np.hstack([np.zeros(MS.shape[0]), np.ones(P.shape[0])])
        # Fit first model
        logger.debug('Training first model')
        self.first_model.fit(MSP, MSP_y)
        prob = self.first_model.predict_proba(MS)[:, 1]
        # Find optimal t
        t = 0.001
        while MS_spies[prob <= t].sum()/MS_spies.sum() <= spie_tolerance:
            t += 0.001
        logger.debug('Optimal t is {0:.06}'.format(t))
        logger.debug('Positive group size {1}, captured spies {0:.02%}'.format(
            MS_spies[prob > t].sum()/MS_spies.sum(), (prob > t).sum()))
        logger.debug('Likely negative group size {1}, captured spies {0:.02%}'.format(
            MS_spies[prob <= t].sum()/MS_spies.sum(), (prob <= t).sum()))
        # likely negative group
        N = MS[(MS_spies == 0) & (prob <= t)]
        P = X[y == 1]
        NP = np.vstack([N, P])
        L = np.hstack([np.zeros(N.shape[0]), np.ones(P.shape[0])])
        # Fit second model
        logger.debug('Training second model')
        self.second_model.fit(NP, L)
        
    def predict(self, X):
        """
        Predicts classes for X. Uses second trained model from self.

        Parameters
        ----------
        X : {array-like} of shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        return self.second_model.predict(np.array(X))
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X. Uses second trained model from self.

        Parameters
        ----------
        X : {array-like} of shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples.
        """
        return self.second_model.predict_proba(np.array(X))[:,1]