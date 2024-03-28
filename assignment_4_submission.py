from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

class Perceptron():
    #initialize hyperparameters (learning rate and number of iterations)
    def __init__(self, eta=0.1, n_iter=50, batch_size=10, nameA='', nameB=''):
        self.eta = eta
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.nameA = nameA
        self.nameB = nameB

    def step_fit(self, X, y):
        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X[i:i+self.batch_size]
            y_batch = y[i:i+self.batch_size]
            errors, Loss = self.loss(X_batch,y_batch)
            self.Loss = Loss
            self.w_[1:] += self.eta * X_batch.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def loss(self, X, y):
        errors = y - self.predict(X)
        Loss = ((errors ** 2).sum()) ** 0.5
        return errors,Loss

    def shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
   
    def conf_matrix(self, y_true, y_pred, title):
        #print classification report
        print(classification_report(y_true, y_pred))

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Define class labels
        classes = [self.nameA, self.nameB]
        # Plot confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted output')
        plt.ylabel('True output')
        # Fill in confusion matrix with values
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment='center',
                    color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.show()
    
    def input_data(self, X_train, y_train, X_test, y_test):
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1+X_train.shape[1])]
        for _ in range(self.n_iter):
            X, y = self.shuffle(X_train, y_train)
            self.step_fit(X, y)

        y_train_pred = self.predict(X_train)
        self.conf_matrix(y_train, y_train_pred, "Training Data")

        y_test_pred = self.predict(X_test)
        self.conf_matrix(y_test, y_test_pred, "Testing Data")

#import dataset
df = pd.read_csv('iris_data.txt', header=None)

SPECIES_1= {'name':"Iris-setosa",'s':0,'end':50} #0:50 # small size
SPECIES_2= {'name':"Iris-versicolor",'s':50,'end':100} # 50:100
SPECIES_3= {'name':"Iris-virginica",'s':100,'end':150} # 100:150
spA,spB=SPECIES_2,SPECIES_3

#preparing our data to be understood by our model
X = df.iloc[np.r_[spA['s']:spA['end'],spB['s']:spB['end']], [0,2]].values
y = df.iloc[np.r_[spA['s']:spA['end'],spB['s']:spB['end']], 4].values
y = np.where(y == spB['name'], -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ppn = Perceptron(eta=0.01, n_iter=100, nameA=spA['name'], nameB=spB['name'], batch_size=25) #initializing a new perceptron
ppn.input_data(X_train, y_train, X_test, y_test)