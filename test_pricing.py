import unittest
from unittest import mock
import socket
import hashlib
import pickle
import numpy as np
from typing import List
import pricing


# TODO @yuyi @fuyexuan
class TestPricing(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    @unittest.skip("Need development")
    def test_poc_and_pricing(self):
        # 这里的raw_test_data,是一个二进制数据流，
        # 通过pickle.loads函数将二进制流还原为文本、图像等数据
        raw_test_data = (
            b"\x80\x04\x95\x0f\x00\x00\x00\x00\x00\x00\x00\x8c\x0bhello_world\x94."
        )
        # 得出数据的价值
        benefit_rate = pricing.poc_rate(pickle.loads(raw_test_data))

        # 这里是所有的交易记录
        test_blockchain_state = [
            {
                "timestamp": 0.0,
                "from": "a" * 64,
                "to": "b" * 64,
                "price": 10.9,
                "unique_id": "v" * 64,
            },
            {
                "timestamp": 1.1,
                "from": "a" * 64,
                "to": "b" * 64,
                "price": 20.9,
                "unique_id": "v" * 64,
            },
        ]
        # 给出数据的推荐价格
        price = pricing.call_pricing(raw_test_data, test_blockchain_state)

    def test_call_pricing(self):
        raw_test_data = (
            b"\x80\x04\x95\x0f\x00\x00\x00\x00\x00\x00\x00\x8c\x0bhello_world\x94."
        )
        benefit_rate = 1.5
        test_blockchain_state = [
            {
                "timestamp": 0.0,
                "from": "a" * 64,
                "to": "b" * 64,
                "price": 10.9,
                "unique_id": "v" * 64,
            },
            {
                "timestamp": 1.1,
                "from": "a" * 64,
                "to": "b" * 64,
                "price": 20.9,
                "unique_id": "v" * 64,
            },
        ]

        recommend_price = pricing.call_pricing(raw_test_data, benefit_rate, test_blockchain_state)

        assert isinstance(recommend_price, float)

        benefit_rate = 3.1
        test_blockchain_state = []
        recommend_price = pricing.call_pricing(raw_test_data, benefit_rate, test_blockchain_state)

        assert isinstance(recommend_price, float)

    def test_call_benefit(self):
        X_train, y_train = np.random.uniform(0, 1, (200, 2)), np.random.randint(0, 2, 200)
        X_test, y_test = np.random.uniform(0, 1, (100, 2)), np.random.randint(0, 2, 100)
        X_new, y_new = X_train[100:], y_train[100:]
        X_init, y_init = X_train[:100], y_train[:100]

        benefit = pricing.call_benefit(X_new=X_new,   y_new=y_new,
                                       X_init=X_init, y_init=y_init,
                                       X_test=X_test, y_test=y_test,
                                       sample_weight=None,
                                       model_family='logistic',
                                       metric='accuracy')

        assert len(X_new[0]) == len(X_test[0])
        assert len(benefit) == 2
        assert isinstance(benefit, np.ndarray)
        assert isinstance(benefit[1], float)
