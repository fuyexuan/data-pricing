import numpy as np
from sklearn.linear_model import LogisticRegression
import inspect
from sklearn.metrics import roc_auc_score, f1_score
import warnings

class Shapley(object):
    # shapley也可扩展各种，为了简化这里只用tmc_shapley

    def __init__(self, X, y, X_test, y_test,
                 sample_weight=None, problem='classification',
                 model_family='logistic', metric='accuracy', seed=None,
                 **kwargs):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test covariates
            y_test: Test labels
            samples_weights: Weight of train samples in the loss function
                (for models where weighted training method is enabled.)
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting
                same permutations.
            **kwargs: Arguments of the model
        """

        if seed is not None:
            np.random.seed(seed)
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        if self.model_family == 'logistic':
            self.hidden_units = []

        self.X, self.y, self.X_test, self.y_test = X, y, X_test, y_test
        self.sample_weight = sample_weight
        self.val = [0., 0.]

        # if len(set(self.y)) > 2:
        #     assert self.metric not in ['f1', 'auc'], f"Metric {self.metric} is invalid for multiclass!"
        is_regression = (np.mean(self.y // 1 == self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        if self.is_regression:
            warnings.warn("Regression problem is no implemented.")
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self._init_score(self.metric)

    def _init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            if (len(self.y_test.shape) > 1):
                y_test = np.argmax(self.y_test, axis=1)
            else:
                y_test = self.y_test
            # hist = np.bincount(y_test).astype(float) / len(y_test)
            # 使用numpy.unique函数来统计y_test中每个元素出现的次数
            values, counts = np.unique(y_test, return_counts=True)
            # 把counts数组转换成浮点数并除以y_test的长度
            hist = counts.astype(float) / len(y_test)
            return np.max(hist)
        if metric == 'f1':
            rnd_f1s = []
            for _ in range(100):  # 随机初始化100次输出求f1_score取平均
                rnd_y = np.random.permutation(self.y)
                rnd_f1s.append(f1_score(self.y_test, rnd_y))
            return np.mean(rnd_f1s)
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):  # 随机初始化100次输出求交叉熵取平均
            rnd_y = np.random.permutation(self.y)
            if self.sample_weight is None:
                self.model.fit(self.X, rnd_y)
            else:
                self.model.fit(self.X, rnd_y,
                               sample_weight=self.sample_weight)
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)
    def run(self,X_new,y_new):
        self.model.fit(self.X, self.y)
        val_init = self.value(self.model, self.metric, X=self.X_test, y=self.y_test)
        vals = [val_init]

        # if isinstance(self.X, dict):
        #     X_combine = {k: [self.X[k], X_new[k]] for k in self.X}
        #     # X_init.update(X_new)
        #     self.model.fit(X_combine, np.concatenate([self.y, y_new]))
        # else:
        #     self.model.fit(np.concatenate([self.X, X_new]),
        #                       np.concatenate([self.y, y_new]))

        self.model.fit(X_new, y_new)
        # self.model.fit(np.concatenate([self.X, X_new]),
        #                   np.concatenate([self.y, y_new]))

        vals.append(self.value(self.model, self.metric, X=self.X_test, y=self.y_test))
        self.val = np.array(vals)

    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data
                different from test set.
            y: Labels, if valuation is performed on a data
                different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if inspect.isfunction(metric):
            return metric(model, X, y)
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return calculate_auc_score(model, X, y)
        if metric == 'xe':
            return calculate_xe_score(model, X, y)
        raise ValueError('Invalid metric!')


def return_model(model, **kwargs):

    if inspect.isclass(model):
        assert getattr(model, 'fit', None) is not None, 'Custom model family should have a fit() method'
        model = model(**kwargs)
    elif isinstance(model, LogisticRegression):
        return model
    # elif isinstance(model, ):
    #     return model
    elif model=='logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', None)
        C = kwargs.get('C', 1.)
        max_iter = kwargs.get('max_iter', 5000)
        model = LogisticRegression(solver=solver, n_jobs=n_jobs, C=C,
                                 max_iter=max_iter, random_state=666)
    else:
        pass
        # raise ValueError("Invalid model!")
    return model


def calculate_accuracy_score(clf, X, y):

    probs = clf.predict_proba(X)
    predictions = np.argmax(probs, -1)
    return np.mean(np.equal(predictions, y))


def calculate_f1_score(clf, X, y):

    predictions = clf.predict(X)
    if len(set(y)) == 2:
        return f1_score(y, predictions)
    return f1_score(y, predictions, average='macro')


def calculate_auc_score(clf, X, y):

    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    return roc_auc_score(y, true_probs)


def calculate_xe_score(clf, X, y):

    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    true_log_probs = - np.log(np.clip(true_probs, 1e-12, None))
    return np.mean(true_log_probs)

