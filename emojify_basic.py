""" ML libaries independence model """

import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt

# number of classes
C = len(emoji_dictionary)

# get dataset
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

# get maximum of the sentence in training set
maxLen = len(max(X_train, key=len).split())

# print out first training example
index = 1
print("{}: {}".format(X_train[index], label_to_emoji(Y_train[index])))

# convert to Y train and test dataset to one hot vector
Y_oh_train = convert_to_one_hot(Y_train, C = C)
Y_oh_test = convert_to_one_hot(Y_test, C = C)
# print out the one hot vector of Y train with index of 50
index = 50
print(Y_train[index], "is converted into one hot", Y_oh_train[index])

# get word embeddings
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.200d.txt')
""" 
word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words, with the valid indices ranging from 0 to 400,000)
index_to_word -- dictionary mapping from indices to their corresponding words in the vocabulary
word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
"""
word = "hello"
index = word_to_index[word]
print("the index of", word, "in the vocabulary is", index)
print("the", str(index) + "th word in the vocabulary is", index_to_word[index])
print("")

print("Testing getting the average of the sentence from the gloVe embeddings.")
avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map, glove_dimension=200)
print("avg = ", avg)

def model(X, Y, word_to_vec_map, n_y, n_h, learning_rate = 0.01, num_iterations = 100):
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    n_y -- number of classes
    n_h -- dimensions of the GloVe vectors
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    
    np.random.seed(1)

    # Define number of training examples
    m = Y.shape[0]                          # number of training examples
    
    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C = n_y) 
    
    # Optimization loop
    for t in range(num_iterations):                       # Loop over the number of iterations
        for i in range(m):                                # Loop over the training examples
            
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map, glove_dimension=n_h)
            # Forward propagate the avg through the softmax layer
            z = np.dot(W, avg) + b
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = - np.sum(Y_oh[i] * np.log(a))
            
            # Compute gradients 
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db
        
        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map, glove_dimension=n_h)

    return pred, W, b

pred, W, b = model(X_train, Y_train, word_to_vec_map, n_y=C, n_h=200)


# sentences want to predict, if want to add more sentence, append at the end of the list
sentences = ["i hate you", "ha ha ha", "so funny"]
X_my_sentences = np.array(sentences)

# get prediction and print prediction
pred = predict(X_my_sentences, None, W, b, word_to_vec_map, customized=True)
