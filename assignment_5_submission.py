import tensorflow as tf
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import numpy as np

CREATE_CLASS=True # Create class from scratch; otherwise use nn.Sequential to create the class
SGD=False # SGD or Adam
IRIS=False # iris or mnist
SHOW=False # returns picture of digit for mnist

# Load Iris dataset
if IRIS:
    examples = load_iris()
else:
    examples = load_digits() # https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html; 10 digits;  1797 examples
    if SHOW:
        idx=random.randint(0,len(examples.target))
        print(examples.data[idx])
        print(examples.data[idx].reshape(8,8))
        print(examples.target[idx])
        plt.matshow(examples.data[idx].reshape(8,8), cmap=plt.cm.gray_r)
        plt.show()

X = examples.data
y = examples.target

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate the model
input_size = X_train.shape[1]
hidden_size = 16
output_size = len(examples.target_names)
batch_size = 120
num_epochs = 100
# Optimizer specific options
learning_rate=0.07
regularization_param=0.001
momentum_param=0.9
# Dropout: if p>0
dropout_p=0.2 # During training, randomly zeroes some of the elements of the input tensor with probability p. Prevents overfitting, makes network more robust

# Create dataloader which makes it easier to use mini batches
train_dl = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size, drop_remainder=True)

########################################################### NN model
if CREATE_CLASS:
    # Create model, first defining the class with a forward method
    class ThreeLayerNet(tf.keras.Model):
        def __init__(self, hidden_size, output_size):
            super(ThreeLayerNet, self).__init__()
            self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
            self.fc2 = tf.keras.layers.Dense(hidden_size, activation='relu')
            self.fc3 = tf.keras.layers.Dense(output_size)
            self.dropout = tf.keras.layers.Dropout(dropout_p)
            
        def call(self, x):
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.fc3(x)
            return x
        
    model = ThreeLayerNet(hidden_size, output_size)

else:
    # tf.keras.Sequential
    model=tf.keras.Sequential([
        tf.keras.Input(shape=(input_size,)),
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dropout(dropout_p),
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dropout(dropout_p),
        tf.keras.layers.Dense(output_size)
    ])
####################################################################################################
# Define loss function and optimizer
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

if SGD: # Stochastic Gradient Decent
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, weight_decay=regularization_param, momentum=momentum_param)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=regularization_param)

# Lists to store train and test losses
train_losses = []
test_losses = []

# Training the model
for epoch in range(num_epochs):
    train_loss = 0.0
    num_iter = len(train_dl)
    for i, (x_batch, y_batch) in enumerate(train_dl):
        # Forward pass
        with tf.GradientTape() as tape:
            pred = model(x_batch, training=True)
            loss = criterion(y_batch, pred)

        # Backward pass and optimization
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss += np.squeeze(loss)

        if i == num_iter - 1:
            break

    train_loss /= len(train_dl)
    train_losses.append(train_loss)

    # Test the model
    outputs = model(X_test, training=False)
    test_loss = criterion(y_test, outputs)
    test_losses.append(np.squeeze(test_loss))

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Plotting train and test losses
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Losses')
plt.legend()
plt.show()

# Testing the model
pred_outcomes = model(X_test, training=False)
pred = np.argmax(pred_outcomes, axis=1)
accuracy = accuracy_score(y_test, pred)
print(f'Accuracy on test set: {accuracy:.4f}')
cm=confusion_matrix(y_test, pred)
labels = np.unique(y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()