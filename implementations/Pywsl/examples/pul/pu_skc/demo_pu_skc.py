import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.utils.estimator_checks import check_estimator

from pywsl.pul import pumil_mr
from pywsl.utils.syndata import gen_twonorm_pumil
from pywsl.utils.comcalc import bin_clf_err


def main():
    prior = .5
    x, y, x_t, y_t = gen_twonorm_pumil(n_p=30, n_u=200, 
                                       prior_u=prior, n_t=100)
    print(x)
    print(len(x))
    print(type(x))
    # print(x[0])
    # print(type(x[0]))
    # print(y)
    # print(type(y))
    breakpoint()
    param_grid = {'prior': [prior], 
                  'lam': np.logspace(-3, 1, 5), 
                  'basis': ['minimax']}
    lambda_list = np.logspace(-3, 1, 5)
    clf = GridSearchCV(estimator=pumil_mr.PUMIL_SL(), 
                       param_grid=param_grid,
                       cv=5, n_jobs=-1)
    clf.fit(x, y)
    y_h = clf.predict(x_t)
    print(y_h)
    print(y_t)
    err = 100*bin_clf_err(y_h, y_t, prior)
    print("MR: {}%".format(err))


if __name__ == "__main__":
    main()