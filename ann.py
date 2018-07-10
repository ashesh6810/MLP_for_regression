import tensorflow as tf
import numpy as np
from numpy import genfromtxt
train=np.load('Xtrain.npy')
label=np.load('Ytrain.npy')
print(label)
test_x=np.load('Xtest.npy')
test_y=np.load('Ytest.npy')
# Parameters
learning_rate = 0.01
training_epochs = 200
batch_size = 32
display_step = 1

# Network Parameters
n_hidden_1 = 1059 # 1st layer number of neurons
n_hidden_2 = 1059 # 2nd layer number of neurons
n_input = 3 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.square(logits-Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for step in range(training_epochs):
        batch_x=(train)
        batch_y=(label)
        _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
        avg_cost =c
        # Display logs per epoch step
       
        print("Epoch:", '%04d' % (step+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")
        
   # Test model
      # Apply softmax to logits
    predicted_vals = sess.run(logits, feed_dict={X:(test_x)})

   # Calculate accuracy
   
    np.savetxt("prediction_ann.csv", predicted_vals, delimiter=",")
