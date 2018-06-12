# emojifier_model
This includes Network Models for Emojifier: Standard RNNs and LSTM. 

## Usage

In order to get the program to work apart from getting prediction by loading in pre-trained model, we need to have pre-trained GloVE embedding vectors. It could be downloaded from one of the current research in Stanford: https://nlp.stanford.edu/projects/glove/

## Get prediction

Before using the file `make_prediction.py`, it is noted that the file will load a saved model from `emojifier_LSTM.h5`, so therefore, without the `emojifier_LSTM.h5`, it is not able to make any prediction.

#### Basic Recurrent Neural Networks

The model is located in the file named `emojifier_basic.py`. It is vital to note that this basic model has no saving capability. Everytime you want to get a prediction for a sentence, you have to go append new sentence into `sentences` list variable in the file and the model will re-train itself.

#### Long Short Term Memory

##### Model

The model is located in the file named `emojify_LSTM.py`. Since `Keras` library has been used for this model, so it now can save all the parameters into a file which is `emojfier_LSTM.h5`. Now, if you re-train the model by using LSTM, the saved model would be overwritten be the new one.

##### Getting prediction

You can go into `make_prediction.py`. In the `list_of_sentences` list variable, feel free to delete and append new items to test out the application.