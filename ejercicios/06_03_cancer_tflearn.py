

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X=cancer.data
Y=cancer.target
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,  Y_test = train_test_split(X,Y, test_size=0.25
                                                    , random_state=2
                                                    )
from tflearn.data_utils import to_categorical

y2_train=to_categorical(Y_train,2)
y2_test=to_categorical(Y_test,2)
print(type(y2_train))
print(y2_train)

import tflearn
# Build neural network
net = tflearn.input_data(shape=[None, 30])
net = tflearn.fully_connected(net, 18, activation="relu")
net = tflearn.fully_connected(net, 9, activation="relu")
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)


logs_path = 'logs/cancer'
# tensorboard --logdir='logs/basic'
# Define model
model = tflearn.DNN(net,  tensorboard_verbose=1,tensorboard_dir=logs_path)
# Start training (apply gradient descent algorithm)
model.fit(X_train, y2_train, n_epoch=1900, batch_size=1000, show_metric=True)
score = model.evaluate(X_test, y2_test)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))

# Run the model on one example
prediction = model.predict([X_test[0]])
print("Prediction: %s" % str(prediction[0]))

