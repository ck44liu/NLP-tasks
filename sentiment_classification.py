# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FFNN(nn.Module):
    """
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, inp, hid1, hid2, out):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(FFNN, self).__init__()
        self.V1 = nn.Linear(inp, hid1)
        # self.g1 = nn.Tanh()
        # self.g1 = nn.ReLU()
        self.W2 = nn.Linear(hid1, hid2)
        # self.g2 = nn.Tanh()
        self.W3 = nn.Linear(hid2, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W3.weight)
        # Initialize with zeros instead
        # nn.init.zeros_(self.V.weight)
        # nn.init.zeros_(self.W.weight)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        return self.log_softmax(self.W3(self.W2(self.V1(x))))


class FFNN_Fancy(nn.Module):
    """
    Fancier model for sentiment classification.
    """
    def __init__(self, input_size, hidden_size):
        super(FFNN_Fancy, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10*hidden_size,
                              stride=hidden_size, padding=10*hidden_size)
        # 64*(70+1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=2)
        # 64*32
        self.fc1 = nn.Linear(64*32, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.25)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x, drop=True):
        lstm_out, _ = self.lstm(x.view(x.size(0), 1, -1))
        # return self.log_softmax(self.fc2(self.fc1(lstm_out[-1,0,:])))
        output = self.relu(self.conv2(self.relu(self.conv1(lstm_out.reshape(1,1,-1)))))  # [60*hidden_size]
        # print(output.shape)
        # [1,64,32]
        output = output.reshape(-1)
        # This part is being explicit. We can also use model.train() and model.eval() instead so that
        # dropout only applies to training stage
        if drop:
            pred = self.log_softmax(self.fc3(self.drop(self.relu(self.fc2(self.drop(self.relu(self.fc1(output))))))))
        else:
            pred = self.log_softmax(self.fc3(self.relu(self.fc2(self.relu(self.fc1(output))))))
        return pred


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, fn, word_vectors: WordEmbeddings, fancy: bool):
        self.fn = fn
        self.word_vectors = word_vectors
        self.fancy = fancy

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        if not self.fancy:
            x = form_input(ex_words, self.word_vectors)
            x = torch.from_numpy(x).float()
            log_probs = self.fn.forward(x)
        else:
            x = np.zeros((60, self.word_vectors.get_embedding_length()))
            for i in range(len(ex_words)):
                x[i] = self.word_vectors.get_embedding(ex_words[i])
            x = torch.from_numpy(x).float()
            log_probs = self.fn.forward(x, drop=False)

        prediction = torch.argmax(log_probs)
        return prediction.int()


def form_input(words: List[str], word_vectors: WordEmbeddings) -> np.ndarray:
    """
    :param words: a string list representing a sentence
    :return: a [word_vectors.get_embedding_length() x 1] Tensor
    """
    inp = np.zeros((word_vectors.get_embedding_length(),))
    for word in words:
        emb = word_vectors.get_embedding(word)
        inp += emb
    inp /= len(words)

    return inp


def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


def train_ffnn(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    a trained NeuralSentimentClassifier.
    :param args: Command-line args so you can access them here
    :param train_exs:
    :param dev_exs:
    :param word_vectors:
    :return: the trained NeuralSentimentClassifier model -- this model should wrap your PyTorch module and implement
    prediction on new examples
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60

    # print some sentences
    print(train_exs[1].words)
    print(train_exs[1].label)
    print(len(train_exs[1].words))
    print(train_exs[1].words[1])
    emb_temp = word_vectors.get_embedding(train_exs[1].words[1])
    print(emb_temp)

    # Define some constants
    # Inputs
    feat_vec_size = word_vectors.get_embedding_length()
    # Hidden units
    embedding_size1 = 60
    embedding_size2 = 10
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # Transform input data
    inputs = np.zeros((len(train_exs), feat_vec_size))
    for i in range(len(train_exs)):
        inputs[i] = form_input(train_exs[i].words, word_vectors)

    # RUN TRAINING AND TEST
    num_epochs = 20
    ffnn = FFNN(feat_vec_size, embedding_size1, embedding_size2, num_classes)
    # print(ffnn)
    initial_learning_rate = 0.001
    optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)
    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = inputs[idx]
            y = train_exs[idx].label
            # Build one-hot representation of y.
            y_onehot = torch.zeros(num_classes)
            # scatter will write the value of 1 into the position of y_onehot given by y
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y, dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object.
            ffnn.zero_grad()
            log_probs = ffnn.forward(torch.from_numpy(x).float())

            # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
            loss = torch.neg(log_probs).dot(y_onehot)

            total_loss += loss
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    return NeuralSentimentClassifier(ffnn, word_vectors, fancy=False)


# Analogous to train_ffnn, but trains your fancier model.
def train_fancy(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> NeuralSentimentClassifier:
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    feat_vec_size = word_vectors.get_embedding_length()
    hidden_size = 64
    num_classes = 2

    # get embeddings for each training sentiment example
    inputs = np.zeros((len(train_exs), seq_max_len, feat_vec_size))
    for i in range(len(train_exs)):
        sentence = train_exs[i]
        length = len(sentence.words)
        for j in range(length):
            inputs[i][j] = word_vectors.get_embedding(sentence.words[j])

    # get embeddings for each dev sentiment example
    dev_inp = np.zeros((len(dev_exs), seq_max_len, feat_vec_size))
    for i in range(len(dev_exs)):
        sentence = dev_exs[i]
        length = len(sentence.words)
        for j in range(length):
            dev_inp[i][j] = word_vectors.get_embedding(sentence.words[j])

    # RUN TRAINING AND TEST
    num_epochs = 3
    ffnn_fancy = FFNN_Fancy(input_size=feat_vec_size, hidden_size=hidden_size)
    # print(ffnn_fancy)
    initial_learning_rate = 0.001
    optimizer = optim.Adam(ffnn_fancy.parameters(), lr=initial_learning_rate)
    for epoch in range(num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        # ex_indices = [i for i in range(300)]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = inputs[idx]
            y = train_exs[idx].label
            # Build one-hot representation of y.
            y_onehot = torch.zeros(num_classes)
            # scatter will write the value of 1 into the position of y_onehot given by y
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y, dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object.
            ffnn_fancy.zero_grad()
            log_probs = ffnn_fancy.forward(torch.from_numpy(x).float())

            # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
            loss = torch.neg(log_probs).dot(y_onehot)

            total_loss += loss
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

        # Evaluate on the train set
        correct = 0
        for idx in range(0, len(train_exs)):
            x_ = inputs[idx]
            y_ = train_exs[idx].label
            probs = ffnn_fancy.forward(torch.from_numpy(x_).float(), drop=False)
            prediction = torch.argmax(probs)
            if y_ == prediction:
                correct += 1
        print(repr(correct) + "/" + repr(len(train_exs)) + " correct after epoch", epoch)
        # Evaluate on the dev set
        correct = 0
        for idx in range(0, len(dev_exs)):
            x_ = dev_inp[idx]
            y_ = dev_exs[idx].label
            probs = ffnn_fancy.forward(torch.from_numpy(x_).float(), drop=False)
            prediction = torch.argmax(probs)
            if y_ == prediction:
                correct += 1
        accu = correct/len(dev_exs)
        print(repr(correct) + "/" + repr(len(dev_exs)) + " = " + repr(accu) + " correct after epoch", epoch)

    return NeuralSentimentClassifier(ffnn_fancy, word_vectors, fancy=True)


