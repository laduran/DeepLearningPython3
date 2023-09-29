import tensorflow as tf
from tensorflow.keras.datasets import mnist
import mnist_loader
from sklearn import svm
import numpy as np


def load_mnist():
    '''
    Load the MNIST dataset from Keras datasets
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Flatten and normalize the data
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0
    return (x_train, y_train), (x_test, y_test)


def load_mnist_pickle_file():
    '''
    Load the MNIST dataset from a local copy of the file.
    Can also get bigger/smaller datasets from here:
    http://yann.lecun.com/exdb/mnist/

    Can train (fit) the model on the smaller set and predict on the larger
    '''
    print ("loading MNIST training data...")
    training_data, _, test_data = mnist_loader.load_data(pickle_filename='./data/mnist_expanded.pkl.gz')
    return (training_data[0], training_data[1]), (test_data[0], test_data[1])

# Train and test the SVM model
def svm_baseline():
    print("loading MNIST training data...")
    (x_train1, y_train1), (x_test1, y_test1) = load_mnist()
    (x_train, y_train), (x_test, y_test) = load_mnist_pickle_file()

    # Create an SVM classifier
    clf = svm.SVC()

    print("Using Tensorflow, training the model on MNIST dataset...")
    clf.fit(x_train, y_train)

    # Test
    print("Creating predictions based on new inputs")
    predictions = clf.predict(x_test)
    num_correct = np.sum(predictions == y_test)

    print("Baseline classifier using an SVM.")
    print(f"{(num_correct/float(len(y_test)))*100}% values correct.")

if __name__ == "__main__":
    # Use GPU acceleration if available
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU for training.")
        # Set GPU memory growth to prevent memory allocation issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    else:
        print("Using CPU for training.")

    svm_baseline()
