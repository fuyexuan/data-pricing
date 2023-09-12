from pricing import *
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import os

def resnet50_model():
	#include_top为是否包括原始Resnet50模型的全连接层，如果不需要自己定义可以设为True
	#不需要预训练模型可以将weights设为None
    resnet50=tf.keras.applications.ResNet50(include_top=False,
                                            weights='imagenet',
                                            input_shape=(32,32,3),
                                            )
	#设置预训练模型冻结的层，可根据自己的需要自行设置
    for layer in resnet50.layers[:15]:
        layer.trainable = False  #

	#选择模型连接到全连接层的位置
    last=resnet50.get_layer(index=30).output
    #建立新的全连接层
    x=tf.keras.layers.Flatten(name='flatten')(last)
    x=tf.keras.layers.Dense(1024,activation='relu')(x)
    x=tf.keras.layers.Dropout(0.5)(x)
    x=tf.keras.layers.Dense(128,activation='relu',name='dense1')(x)
    x=tf.keras.layers.Dropout(0.5,name='dense_dropout')(x)
    x=tf.keras.layers.Dense(10,activation='softmax')(x)

    model = tf.keras.models.Model(inputs=resnet50.input, outputs=x)
    model.summary() #打印模型结构

    def score(x_test, y):
        y_pred = model.predict(x_test)
        m = tf.keras.metrics.Accuracy()
        m.update_state(np.argmax(y, -1), np.argmax(y_pred, -1))
        acc = m.result().numpy()
        return acc

    model.score = score

    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255  # 预处理
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)  # 对数据集进行onehot编码
    y_test = to_categorical(y_test, 10)
    model = resnet50_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.legacy.SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    # checkpoint_save_path = "./checkpoint/resnet50_cifar.ckpt"
    # if os.path.exists(checkpoint_save_path + '.index'):
    #     print('-------------load the model-------------------')
    #     model.load_weights(checkpoint_save_path)
    #
    # checkpointer = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_save_path, verbose=1, save_best_only=True, save_weights_only=True)

    # model.fit(
    #     x=x_train,
    #     y=y_train,
    #     batch_size=64,
    #     epochs=200,
    #     verbose=1,
    #     # callbacks=[checkpointer],
    #     validation_split=0.1,
    #     shuffle=True)

    print(x_train.shape)

    benefit = call_benefit(x_train[:1000], y_train[:1000],
                           x_train[1000:], y_train[1000:],
                           x_test, y_test,
                           model_family=model)

    preds = model.predict(x_test)
    print("after:", model.score(x_test, y_test))

    print(benefit)


