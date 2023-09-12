from pricing import *
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
dataset_new = dataset["train"][3000:6000]  # Just take the training split for now
dataset_train = dataset["train"][1000:2000]  # Just take the training split for now
dataset_test = dataset["train"][2000:3000]  # Just take the training split for now

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_data_new = tokenizer(dataset_new["sentence"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data_new = dict(tokenized_data_new)
tokenized_data_train = tokenizer(dataset_train["sentence"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data_train = dict(tokenized_data_train)
tokenized_data_test = tokenizer(dataset_test["sentence"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data_test = dict(tokenized_data_test)
print("tokenized_data_new:",tokenized_data_new['input_ids'].shape)
print("tokenized_data_train:",tokenized_data_train['input_ids'].shape)
print("tokenized_data_test:",tokenized_data_test['input_ids'].shape)

import numpy as np

labels_new = np.array(dataset_new["label"])  # Label is already an array of 0 and 1
labels_train = np.array(dataset_train["label"])  # Label is already an array of 0 and 1
labels_test = np.array(dataset_test["label"])  # Label is already an array of 0 and 1

from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")
# Lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5))  # No loss argument!

def score(tokenized_data, labels):
    import tensorflow as tf
    from sklearn.metrics import accuracy_score

    outputs = model.predict(tokenized_data)
    logits = outputs.logits
    # 用softmax函数将分数转换为概率
    probs = tf.nn.softmax(logits, axis=-1)
    # 用argmax函数将概率转换为标签
    pred_labels = tf.argmax(probs, axis=-1)
    # 计算准确率
    acc = accuracy_score(labels, pred_labels)

    # 打印结果
    return acc

model.score = score

# print("tokenized_data_new:",tokenized_data_new['input_ids'])
# print("tokenized_data_train:",tokenized_data_train['input_ids'])
# X_combine = {k: tokenized_data_train.get(k, []) + tokenized_data_new.get(k, [])
#              for k in set(tokenized_data_train) | set(tokenized_data_new)}
# # X_combine = {k: [tokenized_data_train[k], tokenized_data_new[k]] for k in tokenized_data_train}
# print("update:", X_combine['input_ids'])
preds = model.predict(tokenized_data_test)
print("before:",model.score(tokenized_data_test, labels_test))


# model.fit(tokenized_data_train, labels_train)

benefit = call_benefit(tokenized_data_new, labels_new,
                       tokenized_data_train, labels_train,
                       tokenized_data_test, labels_test,
                       model_family=model)

preds = model.predict(tokenized_data_test)
print("after:",model.score(tokenized_data_test, labels_test))

print(benefit)


