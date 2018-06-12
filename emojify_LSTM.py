import numpy as np
from helpers import *
import emoji
from keras import backend as K
from keras.models import Model
NUM_OF_LABELS = 22
maxLen = 20

# load data training data and test data
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

# convert data into one hot vector
Y_oh_train = convert_to_one_hot(Y_train, C = NUM_OF_LABELS)
Y_oh_test = convert_to_one_hot(Y_test, C = NUM_OF_LABELS)

# get word embeddings
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.200d.txt')

from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

# create model instance
def EmojiModel(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype="int32")
    
    # Create the embedding layer pretrained with GloVe Vectors 
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(256, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    X = LSTM(256, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)  
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(22)(X)
    # Add a softmax activation
    X = Activation("softmax")(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
        
    return model


model = EmojiModel((maxLen,), word_to_vec_map, word_to_index)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

arr = sentences_to_indices(np.array(["i am so fucked up"]), word_to_index, maxLen)

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = NUM_OF_LABELS)

model.fit(X_train_indices, Y_train_oh, epochs = 120, batch_size = 32, shuffle=True)
model.save('emojifier_LSTM.h5')

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = NUM_OF_LABELS)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print("The current test accuracy is not reliable since it is only generalized for the first 5 emojis.")
print("Test accuracy = ", acc)


# uncomment in order to see some sentences model predicts wrong
# it is not reliable because test data is only generalized for the first 5 emojis

# y_test_oh = np.eye(NUM_OF_LABELS)[Y_test.reshape(-1)]
# X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
# pred = model.predict(X_test_indices)
# for i in range(len(X_test)):
#     x = X_test_indices
#     num = np.argmax(pred[i])
#     if(num != Y_test[i]):
#         print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())



