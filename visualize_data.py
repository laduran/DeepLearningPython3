'''
Render the first 9 images of the MNIST data in a grid.
Just to show that these are indeed low-resolution
handwriting samples.
'''
import matplotlib.pyplot as plt
import mnist_loader


def load_mnist_pickle_file():
    '''
    Load the MNIST dataset from a local copy of the file.
    Can also get bigger/smaller datasets from here:
    http://yann.lecun.com/exdb/mnist/

    Can train (fit) the model on the smaller set and predict on the larger
    '''
    print ("loading MNIST training data...")
    training_data, _, test_data = mnist_loader.load_data(pickle_filename='./mnist.pkl.gz')
    return (training_data[0], training_data[1]), (test_data[0], test_data[1])

if __name__ == "__main__":

    # x_train will be the training images
    # y_train will be the labels for each item
    _ , (x_test, y_test) = load_mnist_pickle_file()

    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_test[i].reshape((28,28)), cmap=plt.get_cmap('gray'))
    plt.show()
