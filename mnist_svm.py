"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# Third-party libraries
from sklearn import svm

# My libraries
import mnist_loader

def svm_baseline():
    print ("loading MNIST training data...")
    training_data, _, test_data = mnist_loader.load_data(pickle_filename='./data/mnist_expanded.pkl.gz')
    # train
    clf = svm.SVC()
    print ("Using sci-kit learn, training the model on MNIST data set...")
    clf.fit(training_data[0], training_data[1])
    # test
    print("Creating predictions based on new inputs")
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("Baseline classifier using an SVM.")
    print(str(num_correct) + " of " + str(len(test_data[1])) + " values correct.")

if __name__ == "__main__":
    svm_baseline()
    
