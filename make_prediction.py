"""
- Load in saved model and intialize new model instance
- Make predictions with the new model
- Only applicable for LSTM
"""

from keras.models import Model, load_model
from helpers import *

model = load_model('emojifier_LSTM.h5')

# load in glove vector
word_to_index, _, _ = read_glove_vecs('data/glove.6B.200d.txt')

# max length of the sentence
maxLen = 20

# number of emojis
NUM_OF_LABELS = 22

# if you want to make a prediction for a sentence, append it to the list below
list_of_sentences = ["ha ha it is not funny at all"]
x_test = np.array(list_of_sentences)

# get indices and print out the prediction
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))


# uncomment this to see a list of emojis
# list_of_emos = ["{} {} ".format(str(i),label_to_emoji(i)) for i in range(NUM_OF_LABELS)]
# print(list_of_emos)

