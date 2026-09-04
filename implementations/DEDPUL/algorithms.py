import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from pandas import Series, DataFrame
import torch

from keras.callbacks import EarlyStopping

from catboost import CatBoostClassifier

from NN_functions import get_discriminator, all_convolution, init_keras_model, train_NN
from utils import GaussianMixtureNoFit, maximize_log_likelihood, rolling_apply, MonotonizingTrends
from TIcE import tice_wrapper


def estimate_preds_cv(df, target, cv=3, n_networks=1, lr=1e-4, hid_dim=32, n_hid_layers=1,
                      random_state=None, training_mode='standard', alpha=None, l2=1e-4, train_nn_options=None,
                      all_conv=False, bayes=False, bn=True):
    """
    Estimates posterior probability y(x) of belonging to U rather than P (ignoring relative sizes of U and P);
        predictions are the average of an ensemble of n_networks neural networks;
        performs cross-val predictions to cover the whole dataset
    :param df: features, np.array (n_instances, n_features)
    :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
    :param cv: number of folds, int
    :param n_networks: number of neural networks in the ensemble to average results of
    :param lr: learning rate, float
    :param hid_dim: number of neurons in each hidden layer
    :param n_hid_layers: number of hidden layers in each network
    :param random_state: seed, used in data kfold split, default is None
    :param alpha: share of N in U
    :param train_nn_options: parameters for train_NN

    :return: predicted probabilities y(x) of belonging to U rather than P (ignoring relative sizes of U and P)
    """

    if train_nn_options is None:
        train_nn_options = dict()

    preds = np.zeros((n_networks, df.shape[0],))
    means = np.zeros((n_networks, df.shape[0],))
    variances = np.zeros((n_networks, df.shape[0],))

    for i in range(n_networks):
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        for train_index, test_index in kf.split(df, target):
            train_data = df[train_index]
            train_target = target[train_index]
            mix_data = train_data[train_target == 1]
            pos_data = train_data[train_target == 0]
            test_data = df[test_index]
            test_target = target[test_index]

            mix_data_test = test_data[test_target == 1]
            pos_data_test = test_data[test_target == 0]
            if not all_conv:
                discriminator = get_discriminator(inp_dim=df.shape[1], out_dim=1, hid_dim=hid_dim,
                                                  n_hid_layers=n_hid_layers, bayes=bayes, bn=bn)
            else:
                discriminator = all_convolution(hid_dim_full=hid_dim, bayes=bayes, bn=bn)
            d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=l2)

            train_NN(mix_data, pos_data, discriminator, d_optimizer,
                     mix_data_test, pos_data_test, nnre_alpha=alpha,
                     d_scheduler=None, training_mode=training_mode, bayes=bayes, **train_nn_options)
            if bayes:
                pred, mean, var = discriminator(
                    torch.as_tensor(test_data, dtype=torch.float32), return_params=True, sample_noise=False)
                preds[i, test_index], means[i, test_index], variances[i, test_index] = \
                    pred.detach().numpy().flatten(), mean.detach().numpy().flatten(), var.detach().numpy().flatten()
            else:
                preds[i, test_index] = discriminator(
                    torch.as_tensor(test_data, dtype=torch.float32)).detach().numpy().flatten()

    preds = preds.mean(axis=0)
    if bayes:
        means, variances = means.mean(axis=0), variances.mean(axis=0)
        return preds, means, variances
    else:
        return preds


def estimate_preds_cv_keras(data, target, n_networks=1, n_layers=1, n_hid=32, lr=10**-5, random_state=42,
                            cv=3, batch_size=128, n_epochs=500, n_early_stop=10, alpha=None, verbose=False):
    es = EarlyStopping(monitor='val_loss', patience=n_early_stop, verbose=0, restore_best_weights=True)
    preds = np.zeros((n_networks, data.shape[0]))
    for i in range(n_networks):
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(data, target):
            clf = init_keras_model(n_layers=n_layers, n_hid=n_hid, lr=lr)
            clf.fit(data[train_idx], target[train_idx],
                    validation_data=(data[test_idx], target[test_idx]),
                    # class_weight={0: target.mean(), 1: 1 - target.mean()},
                    batch_size=batch_size, epochs=n_epochs, callbacks=[es], verbose=verbose)
            preds[i, test_idx] = clf.predict_proba(data[test_idx]).reshape(-1,)
        if random_state is not None:
            random_state += 1
    preds = preds.mean(axis=0)
    # preds = np.median(preds, axis=0)
    return preds


def estimate_preds_cv_catboost(data, target, random_state=None, n_networks=1, catboost_params=None,
                               cv=3, n_early_stop=10, verbose=False):
    if catboost_params is None:
        catboost_params = {}
    preds = np.zeros((n_networks, data.shape[0]))
    for i in range(n_networks):
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(data, target):
            clf = CatBoostClassifier(**catboost_params,
                                     class_weights=(target.mean(), 1 - target.mean()), random_seed=random_state)
            clf.fit(data[train_idx], target[train_idx],
                    eval_set=(data[test_idx], target[test_idx]),
                    use_best_model=True, verbose=verbose, early_stopping_rounds=n_early_stop)
            preds[i, test_idx] = clf.predict_proba(data[test_idx])[:, 1]
        if random_state is not None:
            random_state += 1
    preds = preds.mean(axis=0)
    # preds = np.median(preds, axis=0)
    return preds


def estimate_preds_cv_sklearn(data, target, model, random_state=None, n_networks=1, params=None, cv=3):
    if params is None:
        params = {}
    preds = np.zeros((n_networks, data.shape[0]))
#     w = np.zeros(target.shape)
#     w[target == 0] = target.mean()
#     w[target == 1] = 1 - target.mean()
    for i in range(n_networks):
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(data, target):
            clf = model(**params, class_weight={0: target.mean(), 1: 1 - target.mean()}, random_state=random_state)
            clf.fit(data[train_idx], target[train_idx])#, sample_weight=w[train_idx])
            preds[i, test_idx] = clf.predict_proba(data[test_idx])[:, 1]
        if random_state is not None:
            random_state += 1
    preds = preds.mean(axis=0)
    # preds = np.median(preds, axis=0)
    return preds


def estimate_diff(preds, target, bw_mix=0.05, bw_pos=0.1, kde_mode='logit', threshold=None, k_neighbours=None,
                  tune=False, MT=True, MT_coef=0.2, decay_MT_coef=False, kde_type='kde',
                  n_gauss_mix=20, n_gauss_pos=10, bins_mix=20, bins_pos=20):
    """
    Estimates densities of predictions y(x) for P and U and ratio between them f_p / f_u for U sample;
        uses kernel density estimation (kde);
        post-processes difference of estimated densities - imposes monotonicity on lower preds
        (so that diff is partly non-decreasing) and applies rolling median to further reduce variance
    :param preds: predictions of NTC y(x), probability of belonging to U rather than P, np.array with shape (n,)
    :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
    :param bw_mix: bandwidth for kde of U
    :param bw_pos: bandwidth for kde of P
    :param kde_mode: 'prob', 'log_prob' or 'logit'; default is 'logit'
    :param monotonicity: monotonicity is imposed on density difference for predictions below this number, float in [0, 1]
    :param k_neighbours: difference is relaxed with median rolling window with size k_neighbours * 2 + 1,
        default = int(preds[target == 1].shape[0] // 10)

    :return: difference of densities f_p / f_u for U sample
    """

    if kde_mode is None:
        kde_mode = 'logit'

    if (threshold is None) or (threshold == 'mid'):
        threshold = preds[target == 1].mean() / 2 + preds[target == 0].mean() / 2
    elif threshold == 'low':
        threshold = preds[target == 0].mean()
    elif threshold == 'high':
        threshold = preds[target == 1].mean()

    if k_neighbours is None:
        k_neighbours = int(preds[target == 1].shape[0] // 20)

    if kde_mode == 'prob':
        kde_inner_fun = lambda x: x
        kde_outer_fun = lambda dens, x: dens(x)
    elif kde_mode == 'log_prob':
        kde_inner_fun = lambda x: np.log(x)
        kde_outer_fun = lambda dens, x: dens(np.log(x)) / (x + 10 ** -5)
    elif kde_mode == 'logit':
        kde_inner_fun = lambda x: np.log(x / (1 - x + 10 ** -5))
        kde_outer_fun = lambda dens, x: dens(np.log(x / (1 - x + 10 ** -5))) / (x * (1 - x) + 10 ** -5)

    if kde_type == 'kde':
        if tune:
            bw_mix = maximize_log_likelihood(preds[target == 1], kde_inner_fun, kde_outer_fun, kde_type=kde_type)
            bw_pos = maximize_log_likelihood(preds[target == 0], kde_inner_fun, kde_outer_fun, kde_type=kde_type)

        kde_mix = gaussian_kde(np.apply_along_axis(kde_inner_fun, 0, preds[target == 1]), bw_mix)
        kde_pos = gaussian_kde(np.apply_along_axis(kde_inner_fun, 0, preds[target == 0]), bw_pos)

    elif kde_type == 'GMM':
        if tune:
            n_gauss_mix = maximize_log_likelihood(preds[target == 1], kde_inner_fun, kde_outer_fun, kde_type=kde_type)
            n_gauss_pos = maximize_log_likelihood(preds[target == 0], kde_inner_fun, kde_outer_fun, kde_type=kde_type)

        GMM_mix = GaussianMixture(n_gauss_mix, covariance_type='spherical').fit(
            np.apply_along_axis(kde_inner_fun, 0, preds[target == 1]).reshape(-1, 1))
        GMM_pos = GaussianMixture(n_gauss_pos, covariance_type='spherical').fit(
            np.apply_along_axis(kde_inner_fun, 0, preds[target == 0]).reshape(-1, 1))

        kde_mix = lambda x: np.exp(GMM_mix.score_samples(x.reshape(-1, 1)))
        kde_pos = lambda x: np.exp(GMM_pos.score_samples(x.reshape(-1, 1)))

    elif kde_type == 'hist':
        if tune:
            bins_mix = maximize_log_likelihood(preds[target == 1], kde_inner_fun, lambda kde, x: kde(x),
                                               kde_type=kde_type)
            bins_pos = maximize_log_likelihood(preds[target == 0], kde_inner_fun, lambda kde, x: kde(x),
                                               kde_type=kde_type)
        bars_mix = np.histogram(preds[target == 1], bins=bins_mix, range=(0, 1), density=True)[0]
        bars_pos = np.histogram(preds[target == 0], bins=bins_pos, range=(0, 1), density=True)[0]

        kde_mix = lambda x: bars_mix[np.clip((x // (1 / bins_mix)).astype(int), 0, bins_mix-1)]
        kde_pos = lambda x: bars_pos[np.clip((x // (1 / bins_pos)).astype(int), 0, bins_pos-1)]
        kde_outer_fun = lambda kde, x: kde(x)

    # sorting to relax and impose monotonicity
    sorted_mixed = np.sort(preds[target == 1])

    diff = np.apply_along_axis(lambda x: kde_outer_fun(kde_pos, x) / (kde_outer_fun(kde_mix, x) + 10 ** -5), axis=0,
                               arr=sorted_mixed)
    diff[diff > 50] = 50
    diff = rolling_apply(diff, 5)
    diff = np.append(
        np.flip(np.maximum.accumulate(np.flip(diff[sorted_mixed <= threshold], axis=0)), axis=0),
        diff[sorted_mixed > threshold])
    diff = rolling_apply(diff, k_neighbours)

    if MT:
        MTrends = MonotonizingTrends(MT_coef=MT_coef)
        diff = np.flip(np.array(MTrends.monotonize_array(np.flip(diff, axis=0), reset=True, decay_MT_coef=decay_MT_coef)), axis=0)

    diff.sort()
    diff = np.flip(diff, axis=0)

    # desorting
    diff = diff[np.argsort(np.argsort(preds[target == 1]))]

    return diff


def estimate_diff_bayes(means, variances, target, threshold=None, k_neighbours=None):
    if threshold == 'mid':
        threshold = means[target == 1].mean() / 2 + means[target == 0].mean() / 2
    elif (threshold == 'low') or (threshold is None):
        threshold = means[target == 0].mean()
    elif threshold == 'high':
        threshold = means[target == 1].mean()

    if k_neighbours is None:
        k_neighbours = int(means[target == 1].shape[0] // 20)

    n_mix = means[target == 1].shape[0]
    GMM_mix = GaussianMixtureNoFit(n_mix, covariance_type='spherical', max_iter=1, n_init=1,
                                   weights_init=np.ones(n_mix) / n_mix,
                                   means_init=means[target == 1].reshape(-1, 1),
                                   precisions_init=1 / np.sqrt(variances[target == 1])).fit(
        means[target == 1].reshape(-1, 1))
    kde_mix = lambda x: np.exp(GMM_mix.score_samples(x))

    n_pos = means[target == 0].shape[0]
    GMM_pos = GaussianMixtureNoFit(n_pos, covariance_type='spherical', max_iter=1, n_init=1,
                                   weights_init=np.ones(n_pos) / n_pos,
                                   means_init=means[target == 0].reshape(-1, 1),
                                   precisions_init=1 / np.sqrt(variances[target == 0])).fit(
        means[target == 0].reshape(-1, 1))
    kde_pos = lambda x: np.exp(GMM_pos.score_samples(x))

    sorted_means = np.sort(means[target == 1])
    # diff = np.array(kde_pos(sorted_means.reshape(-1, 1)) / kde_mix(sorted_means.reshape(-1, 1)))
    diff = np.array([])
    for i in range(int(len(sorted_means) // 1000)):
        current = sorted_means[i * 1000: min((i + 1) * 1000, len(sorted_means))]
        diff = np.append(diff, kde_pos(current.reshape(-1, 1)) / kde_mix(current.reshape(-1, 1)))
    diff[diff > 50] = 50

    diff = rolling_apply(diff, k_neighbours)
    diff = np.append(np.flip(np.maximum.accumulate(np.flip(diff[sorted_means <= threshold], axis=0)), axis=0),
                     diff[sorted_means > threshold])

    diff = diff[np.argsort(np.argsort(means[target == 1]))]

    return diff


def estimate_poster_dedpul(diff, alpha=None, quantile=0.05, alpha_as_mean_poster=False, max_it=100, **kwargs):
    """
    Estimates posteriors and priors alpha (if not provided) of N in U with dedpul method
    :param diff: difference of densities f_p / f_u for the sample U, np.array (n,), output of estimate_diff()
    :param alpha: priors, share of N in U (estimated if None)
    :param quantile: if alpha is None, relaxation of the estimate of alpha;
        here alpha is estimaeted as infinum, and low quantile is its relaxed version;
        share of posteriors probabilities that we allow to be negative (with the following zeroing-out)
    :param kwargs: dummy

    :return: tuple (alpha, poster), e.g. (priors, posteriors) of N in U for the U sample, represented by diff
    """
    if alpha_as_mean_poster and (alpha is not None):
        poster = 1 - diff * (1 - alpha)
        poster[poster < 0] = 0
        cur_alpha = np.mean(poster)
        if cur_alpha < alpha:
            left_border = alpha
            right_border = 1
        else:
            left_border = 0
            right_border = alpha

            poster_zero = 1 - diff
            poster_zero[poster_zero < 0] = 0
            if np.mean(poster_zero) > alpha:
                left_border = -50
                right_border = 0
                # return 0, poster_zero
        it = 0
        try_alpha = cur_alpha
        while (abs(cur_alpha - alpha) > kwargs.get('tol', 10**-5)) and (it < max_it):
            try_alpha = (left_border + (right_border - left_border) / 2)
            poster = 1 - diff * (1 - try_alpha)
            poster[poster < 0] = 0
            cur_alpha = np.mean(poster)
            if cur_alpha > alpha:
                right_border = try_alpha
            else:
                left_border = try_alpha
            it += 1
        alpha = try_alpha
        if it >= max_it:
            print('Exceeded maximal number of iterations in finding mean_poster=alpha')
    else:
        if alpha is None:
            alpha = 1 - 1 / max(np.quantile(diff, 1 - quantile, interpolation='higher'), 1)
        poster = 1 - diff * (1 - alpha)
        poster[poster < 0] = 0
    return alpha, poster


def estimate_poster_en(preds, target, alpha=None, estimator='e1', quantile=0.05, **kwargs):
    """
    Estimates posteriors and priors alpha (if not provided) of N in U with en [Elkan-Noto, 2008] method
    :param preds: predictions of classifier, np.array with shape (n,)
    :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
    :param alpha: priors, share of N in U (estimated if None)
    :param estimator: 'e1' or 'e3' - from [Elkan-Noto, 2008]
    :param quantile: if alpha is None and estimator is 'e3', relaxation of the estimate of alpha;
        share of posteriors probabilities that we allow to be negative (with the following zeroing-out)
    :param kwargs: dummy
    :return: tuple (alpha, poster), e.g. (priors, posteriors) of N in U for the U sample preds[target == 1]
    """
    if alpha is None:
        if estimator == 'e1':
            c = 1 - np.mean(preds[target == 0])
            alpha = 1 - (1 - c) / c
        elif estimator == 'e3':
            # c = np.quantile(1 - preds, 0.95)
            alpha = 1 - min(np.quantile(preds / (1 - preds), quantile, interpolation='lower'), 1)
        # alpha = 1 - (1 - c) / c
        alpha = max(alpha, 0)
    poster = 1 - (1 - alpha) * (1 - preds[target == 1]) / preds[target == 1]
    poster[poster < 0] = 0
    return alpha, poster


def estimate_poster_em(diff=None, preds=None, target=None, mode='dedpul', converge=True, tol=10**-5,
                       max_iterations=1000, nonconverge=True, step=0.001, max_diff=0.05, plot=False, disp=False,
                       alpha=None, alpha_as_mean_poster=True, **kwargs):
    """
    Performs Expectation-Maximization to estimate posteriors and priors alpha (if not provided) of N in U
        with either of 'en' or 'dedpul' methods; both 'converge' and 'nonconverge' are recommended to be set True for
        better estimate
    :param diff: difference of densities f_p/f_u for the sample U, np.array (n,), output of estimate_diff()
    :param preds: predictions of classifier, np.array with shape (n,)
    :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
    :param mode: 'dedpul' or 'en'; if 'dedpul', diff needs to be provided; if 'en', preds and target need to be provided
    :param converge: True or False; True if convergence estimate should be computed
    :param tol: tolerance of error between priors and mean posteriors, indicator of convergence
    :param max_iterations: if exceeded, search of converged alpha stops even if tol is not reached
    :param nonconverge: True or False; True if non-convergence estimate should be computed
    :param step: gap between points of the [0, 1, step] gird to choose best alpha from
    :param max_diff: alpha with difference of mean posteriors and priors bigger than max_diff cannot be chosen;
        an heuristic to choose bigger alpha
    :param plot: True or False, if True - plots ([0, 1, grid], mean posteriors - alpha) and
        ([0, 1, grid], second lag of (mean posteriors - alpha))
    :param disp: True or False, if True - displays if the algorithm didn't converge
    :param alpha: proportions of N in U; is estimated if None
    :return: tuple (alpha, poster), e.g. (priors, posteriors) of N in U for the U sample
    """
    assert converge + nonconverge, "At least one of 'converge' and 'nonconverge' has to be set to 'True'"

    if alpha is not None:
        if mode == 'dedpul':
            alpha, poster = estimate_poster_dedpul(diff, alpha=alpha, alpha_as_mean_poster=alpha_as_mean_poster, tol=tol, **kwargs)
        elif mode == 'en':
            _, poster = estimate_poster_en(preds, target, alpha=alpha, **kwargs)
        return alpha, poster

    # if converge:
    alpha_converge = 0
    for i in range(max_iterations):

        if mode.endswith('dedpul'):
            _, poster_converge = estimate_poster_dedpul(diff, alpha=alpha_converge, **kwargs)
        elif mode == 'en':
            _, poster_converge = estimate_poster_en(preds, target, alpha=alpha_converge, **kwargs)

        mean_poster = np.mean(poster_converge)
        error = mean_poster - alpha_converge

        if np.abs(error) < tol:
            break
        if np.min(poster_converge) > 0:
            break
        alpha_converge = mean_poster

    if disp:
        if i >= max_iterations - 1:
            print('max iterations exceeded')

    # if nonconverge:

    errors = np.array([])
    for alpha_nonconverge in np.arange(0, 1, step):

        if mode.endswith('dedpul'):
            _, poster_nonconverge = estimate_poster_dedpul(diff, alpha=alpha_nonconverge, **kwargs)
        elif mode == 'en':
            _, poster_nonconverge = estimate_poster_en(preds, target, alpha=alpha_nonconverge, **kwargs)
        errors = np.append(errors, np.mean(poster_nonconverge) - alpha_nonconverge)

    idx = np.argmax(np.diff(np.diff(errors))[errors[1: -1] < max_diff])
    alpha_nonconverge = np.arange(0, 1, step)[1: -1][errors[1: -1] < max_diff][idx]

    if plot:
        fig, axs = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(6, 10))
        axs[0].plot(np.arange(0, 1, step), errors)
        axs[1].plot(np.arange(0, 1, step)[1: -1], np.diff(np.diff(errors)))

    # if converge and not nonconverge:
    #     return alpha_converge, poster_converge

    if ((alpha_nonconverge >= alpha_converge) or#converge and nonconverge and
        (((errors < 0).sum() > 1) and (alpha_converge < 1 - step))):
        return alpha_converge, poster_converge

    elif nonconverge:
        if mode == 'dedpul':
            _, poster_nonconverge = estimate_poster_dedpul(diff, alpha=alpha_nonconverge, **kwargs)
        elif mode == 'en':
            _, poster_nonconverge = estimate_poster_en(preds, target, alpha=alpha_nonconverge, **kwargs)

        if disp:
            print('didn\'t converge')
        return alpha_nonconverge, poster_nonconverge
        # return np.mean(poster_nonconverge), poster_nonconverge

    else:
        if disp:
            print('didn\'t converge')
        return None, None


def estimate_poster_cv(df, target, estimator='dedpul', bayes=False, alpha=None, estimate_poster_options=None,
                       estimate_diff_options=None, estimate_preds_cv_options=None, train_nn_options=None):
    """
    Estimates posteriors and priors alpha (if not provided) of N in U; f_u(x) = (1 - alpha) * f_p(x) + alpha * f_n(x)
    :param df: features, np.array (n_instances, n_features)
    :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
    :param estimator: 'dedpul', 'baseline_dedpul', 'random_dedpul ,'en', 'em_en', or 'nnre';
        'ntc_methods' for every estimate but 'nnre'
    :param alpha: share of N in U; is estimated if not provided (nnRE requires it to be provided)
    :param estimate_poster_options: parameters for estimate_poster... functions
    :param estimate_diff_options: parameters for estimate_diff
    :param estimate_preds_cv_options: parameters for estimate_preds_cv
    :param train_nn_options: parameters for train_NN
    :return: if estimator != 'ntc_methods':
        tuple (alpha, poster), e.g. (priors, posteriors) of N in U for the U sample df[target == 1]
        if estimator == 'ntc_methods':
        dictionary with such (alpha, poster) tuples as values and method names as keys
    """

    if isinstance(df, DataFrame):
        df = df.values
    if isinstance(target, Series):
        target = target.values

    if estimator == 'nnre':
        training_mode = 'nnre'
    else:
        training_mode = 'standard'

    if train_nn_options is None:
        train_nn_options = dict()

    if estimate_poster_options is None:
        estimate_poster_options = dict()

    if estimate_diff_options is None:
        estimate_diff_options = dict()

    if estimate_preds_cv_options is None:
        estimate_preds_cv_options = dict()

    # preds = estimate_preds_cv_catboost(df, target, **estimate_preds_cv_options)
    ### uncomment the line above and comment the line below for experiments with catboost instead of neural networks
    preds = estimate_preds_cv(df=df, target=target, alpha=alpha, training_mode=training_mode, bayes=bayes,
                              train_nn_options=train_nn_options, **estimate_preds_cv_options)
    if bayes:
        preds, means, variances = preds
    if estimator in {'dedpul', 'baseline_dedpul', 'ntc_methods'}:
        if bayes:
            diff = estimate_diff_bayes(means, variances, target, **estimate_diff_options)
        else:
            diff = estimate_diff(preds, target, **estimate_diff_options)

    if estimator == 'dedpul':
        alpha, poster = estimate_poster_em(diff=diff, mode='dedpul', alpha=alpha, **estimate_poster_options)

    elif estimator == 'baseline_dedpul':
        alpha, poster = estimate_poster_dedpul(diff=diff, alpha=alpha, **estimate_poster_options)

    elif estimator == 'en':
        alpha, poster = estimate_poster_en(preds, target, alpha=alpha, **estimate_poster_options)

    elif estimator == 'em_en':
        alpha, poster = estimate_poster_em(preds=preds, target=target, mode='en', alpha=alpha, **estimate_poster_options)

    elif estimator == 'nnre':
        poster = preds[target == 1]

    elif estimator == 'ntc_methods':
        res = dict()
        res['dedpul'] = estimate_poster_em(diff=diff, mode='dedpul', alpha=None, **estimate_poster_options)
        res['baseline_dedpul'] = estimate_poster_dedpul(diff=diff, alpha=None, **estimate_poster_options)
        res['e1_en'] = estimate_poster_en(preds, target, alpha=None, estimator='e1', **estimate_poster_options)
        res['e3_en'] = estimate_poster_en(preds, target, alpha=None, estimator='e3', **estimate_poster_options)
        res['em_en'] = estimate_poster_em(preds=preds, target=target, mode='en', alpha=None, **estimate_poster_options)

        res['dedpul_poster'] = estimate_poster_em(diff=diff, mode='dedpul', alpha=alpha, **estimate_poster_options)
        res['baseline_dedpul_poster'] = estimate_poster_dedpul(diff=diff, alpha=alpha, **estimate_poster_options)
        res['e1_en_poster'] = estimate_poster_en(preds, target, alpha=alpha, estimator='e1', **estimate_poster_options)
        res['e3_en_poster'] = estimate_poster_en(preds, target, alpha=alpha, estimator='e3', **estimate_poster_options)
        res['em_en_poster'] = estimate_poster_em(preds=preds, target=target, mode='en', alpha=alpha, **estimate_poster_options)

        return res

    return alpha, poster
