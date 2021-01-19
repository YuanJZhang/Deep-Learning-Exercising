from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf
from model import MyModel
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

#download and load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# imgs = x_test[:5]
# labs = y_test[:5]
# print(labs)
# plot_imgs = np.hstack(imgs)   #horizontal水平的 将三张图片水平拼接
# plt.imshow(plot_imgs, cmap='gray') #绘制图片
# plt.show()

#Add  a channels dimension
x_train = x_train[...,tf.newaxis]
x_test = x_test[...,tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)   #10000代表在内存中先一次性载入10000张图片然后再从10000张图片随机选取32张图片
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

#create model
model = MyModel()

#define loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()  # 稀疏的多类别损失
#defin optimizer
optimizer = tf.keras.optimizers.Adam()

#defin train_loss and train_accuracy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

#defin test_loss and test_accuracy
test_loss = tf.keras.metrics.Mean(name='train_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# define train function including calculating loss ,applying gradient and calculating accuracy
@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels,predictions)
        gradients =tape.gradient(loss,model.trainable_variables)         #误差反向传播 用一个batch的样本计算出误差函数  然后在求得每个变量的梯度
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))   #再利用优化器迭代更新每个参数

        train_loss(loss)
        train_accuracy(labels,predictions)

#define test function including calculating loss and calculating accuracy
@tf.function
def test_step(images,labels):
    predictions = model(images)
    t_loss = loss_object(labels,predictions)

    test_loss(t_loss)
    test_accuracy(labels,predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss.reset_states() #clean history info
    train_accuracy.reset_states() # clear history info
    test_loss.reset_states()  # clean history info
    test_accuracy.reset_states()  # clear history info

    for images, labels in train_ds:
        train_step(images,labels)

    for test_images,test_labels in test_ds:
        test_step(test_images,test_labels)
        
    template = 'Epoch {},Loss: {},Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
