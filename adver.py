import tensorflow as tf
import numpy as np

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

# Define TF model graph
model = model_mnist()
predictions = model(x)
print("Defined TensorFlow model graph.")

# Get MNIST test data
X_train, Y_train, X_test, Y_test = data_mnist()

# Train an MNIST model
model_train(sess, x, y, predictions, X_train, Y_train)

# Evaluate the accuracy of the MNIST model on legitimate test examples
accuracy = model_eval(sess, x, y, predictions, X_test, Y_test)
assert X_test.shape[0] == 10000, X_test.shape
print 'Test accuracy on legitimate test examples: ' + str(accuracy)

# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
adv_x = fgsm(x, predictions, eps=0.3)
X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
assert X_test_adv.shape[0] == 10000, X_test_adv.shape

# Evaluate the accuracy of the MNIST model on adversarial examples
accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test)
print'Test accuracy on adversarial examples: ' + str(accuracy)



X_test_adv2 = GeneticAlgorithm.run(X_test_adv)
assert X_test_adv_2.shape[0] == 10000, X_test_adv_2.shape

accuracy = model_eval(sess, x, y, predictions, X_test_adv2, Y_test)
print'Test accuracy on adversarial examples: ' + str(accuracy)

