
import numpy as np
from sklearn.model_selection import train_test_split

def get_mnist_dataset():
    X = np.load('../data/test_images.npy')
    y = np.load('../data/test_labels.npy')
    X=X/255.
    y = y.astype('int32')
    n_values = np.max(y) + 1
    y = np.eye(int(n_values))[y]
    y = np.squeeze(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test