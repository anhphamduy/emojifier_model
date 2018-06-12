"""
- Load in saved model and intialize new model instance
- Make predictions with the new model
- Only applicable for LSTM
"""

from keras.models import Model, load_model
from emo_utils import *

model = load_model('emojify_LSTM.h5')

word_to_index, _, _ = read_glove_vecs('data/glove.6B.200d.txt')

maxLen = 20
NUM_OF_LABELS = 22

# if you want to make a prediction for a sentence, append it to the list below
list_of_sentences = ["ha ha it is not funny at all"]
x_test = np.array(list_of_sentences)


X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))

list_of_emos = ["{} {} ".format(str(i),label_to_emoji(i)) for i in range(NUM_OF_LABELS)]

print(list_of_emos)

