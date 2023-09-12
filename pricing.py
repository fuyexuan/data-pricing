import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression

from benefit_utils import Shapley

import warnings

warnings.filterwarnings("ignore")


def call_pricing(raw_test_data: bytes, benefit_rate: float, history_states: List, model=None) -> float:
    # 根据数据价值和时间戳得出的价格推荐
    # 暂时先不用raw_test_data，假设给定的历史记录都是关于当前数据的，直接用市场价格来线性回归拟合，后期再匹配当前数据的价格和全局价格波动

    if not benefit_rate:
        pass
        # call_benefit()

    DATA_VALUE_ARGS = 20.0
    # 价格和价值的默认设定系数，后期可以根据不同类型数据设置不同的系数

    standard_price = benefit_rate * DATA_VALUE_ARGS

    if not history_states:
        return standard_price

    x = np.array([history_state["timestamp"] for history_state in history_states])
    y = np.array(
        [(history_state["price"] - standard_price) * 1.0 / (standard_price * 1.0) for history_state in history_states])

    x = x.reshape(-1, 1)

    if model is None:
        model = LinearRegression
    model = model()
    model.fit(x, y)

    time_now = [[2.0]]
    # 暂不确定history_state中时间戳的表示形式，先简单设定一个当前时间
    predictions = model.predict(time_now)

    return standard_price * (1.0 + predictions[0])


def call_benefit(X_new, y_new: np.ndarray,
                 X_init, y_init: np.ndarray,
                 X_test, y_test: np.ndarray,
                 methods='Shapley',
                 num_test=100,
                 sample_weight=None,
                 model_family='logistic',
                 metric='accuracy') -> np.ndarray:
    # 可选择shapley等不同的数据评估方式，默认先只写shapley
    """Computes the data shapley value of the new-added data.

    Args:
        dshap: The shapley algorithms used to compute the data shapley value.
        order: The order to add the new data.
        points: The num of new data used to train the model.
        X_new, y_new: The new-added data.
        X_init, y_init: The original data.
        X_test, y_test: The test data to measure the model performance.
    """

    if methods == 'Shapley':
        benefit = Shapley(X=X_init, y=y_init,
                          X_test=X_test, y_test=y_test,
                          num_test=num_test,
                          sample_weight=sample_weight,
                          model_family=model_family,
                          metric=metric)
    else:
        benefit = Shapley(X=X_init, y=y_init,
                          X_test=X_test, y_test=y_test,
                          num_test=num_test,
                          sample_weight=sample_weight,
                          model_family=model_family,
                          metric=metric)

    # benefit.model.fit(X_init, y_init)
    # val_init = benefit.value(benefit.model, benefit.metric, X=X_test, y=y_test)
    # vals = [val_init]
    #
    # if isinstance(X_init,dict):
    #     # X_combine = {k: [X_init[k], X_new[k]] for k in X_init}
    #     # # X_init.update(X_new)
    #     # benefit.model.fit(X_combine, np.concatenate([y_init, y_new]))
    #     benefit.model.fit(X_new, y_new)
    # else:
    #     # benefit.model.fit(np.concatenate([X_init, X_new]),
    #     #                   np.concatenate([y_init, y_new]))
    #     benefit.model.fit(X_new, y_new)
    #
    #
    # # benefit.model.fit(np.concatenate([X_init, X_new]),
    # #                   np.concatenate([y_init, y_new]))
    # vals.append(benefit.value(benefit.model, benefit.metric, X=X_test, y=y_test))
    # return np.array(vals)
    benefit.run(X_new, y_new)
    return benefit.val

if __name__ == "__main__":
    # dic1 = {"a":[[1]]}
    pass