import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, laplace, t
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
# from statsmodels.stats.multitest import multipletests


class GaussianMixtureNoFit(GaussianMixture):
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10, max_components=None):

        self.max_components = max_components

        idx = np.arange(means_init.shape[0])
        if (max_components is not None) and (means_init.shape[0] > max_components):
            n_components = min(n_components, max_components)
            np.random.shuffle(idx)
            idx = idx[:self.max_components]

        weights_init = weights_init[idx]
        weights_init /= weights_init.sum()
        means_init = means_init[idx]
        precisions_init = precisions_init[idx]

        super().__init__(n_components, covariance_type, tol,
                         reg_covar, max_iter, n_init, init_params,
                         weights_init, means_init, precisions_init,
                         random_state, warm_start,
                         verbose, verbose_interval)

        self.weights = weights_init
        self.means = means_init
        self.precisions = precisions_init
        self.weights_ = weights_init
        self.means_ = means_init
        self.precisions_ = precisions_init
        self.precisions_cholesky_ = self.precisions
        self.covariances_ = 1 / (self.precisions ** 2)
        self.covariances = 1 / (self.precisions ** 2)

    def _initialize(self, X, resp):
        pass

    def _m_step(self, X, log_resp):
        pass

    def fit(self, X, y=None):
        pass
        return self


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))


def rolling_apply(diff, k_neighbours):
    s = pd.Series(diff)
    s = np.concatenate((
                        s.iloc[:2*k_neighbours].expanding(center=False).median()[::2].values,
                        s.rolling(k_neighbours*2+1, center=True).median().dropna().values,
                        np.flip(np.flip(s.iloc[-2*k_neighbours:], axis=0).expanding(center=False).median()[::2], axis=0).values
    ))
    return s


def compute_log_likelihood(preds, kde, kde_outer_fun=lambda kde, x: kde(x)):
    likelihood = np.apply_along_axis(lambda x: kde_outer_fun(kde, x), 0, preds)
    return np.log(likelihood).mean()


def maximize_log_likelihood(preds, kde_inner_fun, kde_outer_fun, n_folds=5, kde_type='kde', bw_low=0.01, bw_high=0.4,
                            n_gauss_low=1, n_gauss_high=50, bins_low=20, bins_high=250, n_steps=25):
    kf = KFold(n_folds, shuffle=True)
    idx_best, like_best = 0, 0
    bws = np.exp(np.linspace(np.log(bw_low), np.log(bw_high), n_steps))
    n_gauss = np.linspace(n_gauss_low, n_gauss_high, n_steps).astype(int)
    bins = np.linspace(bins_low, bins_high, n_steps).astype(int)
    for idx, (bw, n_g, bin) in enumerate(zip(bws, n_gauss, bins)):
        like = 0
        for train_idx, test_idx in kf.split(preds):
            if kde_type == 'kde':
                kde = gaussian_kde(np.apply_along_axis(kde_inner_fun, 0, preds[train_idx]), bw)
            elif kde_type == 'GMM':
                GMM = GaussianMixture(n_g, covariance_type='spherical').fit(
                    np.apply_along_axis(kde_inner_fun, 0, preds[train_idx]).reshape(-1, 1))
                kde = lambda x: np.exp(GMM.score_samples(x.reshape(-1, 1)))
            elif kde_type == 'hist':
                bars = np.histogram(preds[train_idx], bins=bin, range=(0, 1), density=True)[0]
                kde = lambda x: bars[np.clip((x // (1 / bin)).astype(int), 0, bin - 1)]
                kde_outer_fun = lambda kde, x: kde(x)

            like += compute_log_likelihood(preds[test_idx], kde, kde_outer_fun)
        if like > like_best:
            like_best, idx_best = like, idx
    if kde_type == 'kde':
        return bws[idx_best]
    elif kde_type == 'GMM':
        return n_gauss[idx_best]
    elif kde_type == 'hist':
        return bins[idx_best]


class MonotonizingTrends:
    def __init__(self, a=None, MT_coef=1):
        self.counter = dict()
        self.array_new = []
        if a is None:
            self.array_old = []
        else:
            self.add_array(a)
        self.MT_coef = MT_coef

    def add_array(self, a):
        if isinstance(a, np.ndarray) or isinstance(a, pd.Series):
            a = a.tolist()
        self.array_old = a

    def reset(self):
        self.counter = dict()
        self.array_old = []
        self.array_new = []

    def get_highest_point(self):
        if self.counter:
            return max(self.counter)
        else:
            return np.NaN

    def add_point_to_counter(self, point):
        if point not in self.counter.keys():
            self.counter[point] = 1

    def change_counter_according_to_point(self, point):
        for key in self.counter.keys():
            if key <= point:
                self.counter[key] += 1
            else:
                self.counter[key] -= self.MT_coef

    def clear_counter(self):
        for key, value in list(self.counter.items()):
            if value <= 0:
                self.counter.pop(key)

    def update_counter_with_point(self, point):
        self.change_counter_according_to_point(point)
        self.clear_counter()
        self.add_point_to_counter(point)

    def monotonize_point(self, point=None):
        if point is None:
            point = self.array_old.pop(0)
        new_point = max(point, self.get_highest_point())
        self.array_new.append(new_point)
        self.update_counter_with_point(point)
        return new_point

    def monotonize_array(self, a=None, reset=False, decay_MT_coef=False):
        if a is not None:
            self.add_array(a)
        decay_by = 0
        if decay_MT_coef:
            decay_by = self.MT_coef / len(a)

        for _ in range(len(self.array_old)):
            self.monotonize_point()
            if decay_MT_coef:
                self.MT_coef -= decay_by

        if not reset:
            return self.array_new
        else:
            array_new = self.array_new[:]
            self.reset()
            return array_new


def reg_to_class(s):
    return (s > s.mean()).astype(int)


def mul_to_bin(s, border=None):
    if border is None:
        border = s.median()
    return (s > border).astype(int)


def dummy_encode(df):
    """
   Auto encodes any dataframe column of type category or object.
   """

    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df


def normalize_col(s):
    std = s.std()
    mean = s.mean()
    if std > 0:
        return (s - mean) / std
    else:
        return s - mean


def normalize_cols(df, columns=None):
    if columns is None:
        columns = df.columns
    for col in columns:
        df[col] = normalize_col(df[col])
    return df


def generate_data(mix_size, pos_size, alpha, delta_mu=2, multiplier_s=1, distribution='normal',
                  random_state=None, dim=10):
    """
    Generates pu data
    :param mix_size: total size of mixed data
    :param pos_size: total size of positive data
    :param alpha: share of negatives in mixed data
    :param delta_mu: center of negative distribution (positive is centered at 0)
    :param multiplier_s: std of negative distribution (positive has std 1)
    :param distribution: either 'normal' or 'laplace'
    :return: tuple (data, target_pu, target_mix)
        data - generated points, np.array shaped as (n_instances, 1)
        target_pu - label of data: 0 if positive, 1 is unlabeled, np.array shaped as (n_instances,)
        target_mix - label of data: 0 if positive in unlabeled, 1 if negative in unlabeled,
            2 if positive, np.array shaped as (n_instances,)
        target_pu == 0 <=> target_mix == 2; target_pu == 1 <=> target_mix == 0 or target_mix == 1
    """

    if distribution == 'normal':
        sampler = np.random.normal
    elif distribution == 'laplace':
        sampler = np.random.laplace
    elif distribution == 'laplace_rw':
        def sampler(mu, s, size=1, dim=dim, random_state=random_state):
            x = np.zeros((size, dim))
            x[:, 0] = np.random.laplace(mu, s, size)
            for i in range(1, dim):
                if random_state is not None:
                    np.random.seed(random_state + 100*i)
                x[:, i] = x[:, i-1] + np.random.laplace(0, s, size)
            np.random.seed(random_state)
            return x

    np.random.seed(random_state)

    mix_data = np.concatenate((sampler(0, 1, int(mix_size * (1 - alpha))),
                               sampler(delta_mu, multiplier_s, int(mix_size * alpha))), axis=0)
    pos_data = sampler(0, 1, pos_size)
    if distribution in {'normal', 'laplace'}:
        mix_data = mix_data.reshape([-1, 1])
        pos_data = pos_data.reshape([-1, 1])

    data = np.concatenate((mix_data, pos_data), axis=0)
    target_mix = np.append(np.zeros((int(mix_size * (1 - alpha)),)), np.ones((int(mix_size * alpha),)))
    target_mix = np.append(target_mix, np.full((pos_size,), 2))

    index = np.arange(data.shape[0])
    np.random.shuffle(index)

    data = data[index]
    target_mix = target_mix[index]
    target_pu = target_mix.copy()
    target_pu[target_pu == 0] = 1
    target_pu[target_pu == 2] = 0

    np.random.seed(None)

    return data, target_pu, target_mix


def estimate_cons_alpha(dmu, ds, alpha, distribution):
    """
    Estimates alpha^* for limited cases of mixtures of normal and laplace distributions where either dmu=0 or ds=1;
    f_p(x) = N(0, 1) or L(0, 1), while f_n(x) = N(mu2, s2) or L(mu2, s2), is specified with parameters

    :param dmu: mu2 - mu1, float
    :param ds: s2 / s1, float
    :param alpha: true mixture proportions alpha, float in [0, 1]
    :param distribution: 'normal' or 'laplace'
    :return: ground truth proportions alpha^*, float in [0, 1]
    """
    if distribution == 'normal':
        pdf = norm.pdf
    elif distribution == 'laplace':
        pdf = laplace.pdf

    p1 = lambda x: pdf(x, 0, 1)
    p2 = lambda x: pdf(x, dmu, ds)
    pm = lambda x: p1(x) * (1 - alpha) + p2(x) * alpha

    if dmu == 0 and ds == 1:
        cons_alpha = 0
    elif distribution == 'laplace' and ds == 1:
        cons_alpha = (1 - pm(-100) / p1(-100)).item()
    elif distribution == 'normal' and ds == 1:
        cons_alpha = alpha
    elif dmu == 0:
        cons_alpha = (1 - pm(0) / p1(0)).item()
    else:
        raise NotImplemented
    return cons_alpha


def estimate_cons_alpha_universal(data_pos, data_mix, dens_pos, dens_mix):

    data = np.concatenate((data_pos, data_mix), axis=0)
    cons_alpha = 1-np.min(np.apply_along_axis(lambda x: dens_mix(x) / dens_pos(x), min(len(data_mix.shape)-1, 1), data))
    return cons_alpha.item()


def estimate_cons_poster_universal(data, dens_pos, dens_mix, cons_alpha):
    poster = np.apply_along_axis(lambda x: (dens_mix(x) - dens_pos(x) * (1 - cons_alpha)) / dens_mix(x), -1, data)
    poster[poster < 0] = 0
    return poster


def estimate_cons_poster(points, dmu, ds, distribution, alpha, cons_alpha=None):
    """
    Similar to estimate_cons_alpha; estimates ground-truth proportions f^*(p \mid x) that correspond to alpha^*
    :param points: sample from one-dimensional normal or laplace distribution
    :param dmu: mu2 - mu1, float
    :param ds: s2 / s1, float
    :param alpha: true mixture proportions alpha, float in [0, 1]
    :param distribution: 'normal' or 'laplace'
    :param cons_alpha: alpha^*; output of estimate_cons_alpha
    :return: ground-truth proportions f^*(p \mid x) that correspond to alpha^*
    """
    if distribution == 'normal':
        pdf = norm.pdf
    elif distribution == 'laplace':
        pdf = laplace.pdf

    p1 = lambda x: pdf(x, 0, 1)
    p2 = lambda x: pdf(x, dmu, ds)
    pm = lambda x: p1(x) * (1 - alpha) + p2(x) * alpha

    if cons_alpha is None:
        cons_alpha = estimate_cons_alpha(dmu, ds, alpha, distribution)

    cons_poster = np.apply_along_axis(lambda x: (pm(x) - p1(x) * (1 - cons_alpha)) / pm(x), -1, points)
    return cons_poster


def roc_auc_loss(y_true, y_score, average='macro', sample_weight=None, max_fpr=None):
    return -roc_auc_score(y_true, y_score, average, sample_weight, max_fpr)


def accuracy_loss(y_true, y_pred, normalize=True, sample_weight=None):
    return -accuracy_score(y_true, y_pred.round().astype(int), normalize, sample_weight)


def clean_columns_poster(df):
    df = df.copy()
    cols = df.columns
    for col in cols:
        if '_poster' in col:
            new_col = col.replace('_poster', '')
            df.drop(columns=[new_col], inplace=True)
            df.rename(columns={col: new_col}, inplace=True)
    return df
