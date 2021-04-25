import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var
from utils import *
from data import *
from lf_evaluator import *
import numpy as np
from typing import List
import os
# enables correct operation on Mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """

    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap / float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


###################################################################################################################
# You do not have to use any of the classes in this file, but they're meant to give you a starting implementation.
# for your network.
###################################################################################################################

# The BeamDec here is adapted from the given Beam class in utils.py, some functions are
# commented out or simplified to better fit the seq2seq decoding scenario
class BeamDec(object):
    """
    Beam data structure. Maintains a list of scored elements like a Counter, but only keeps the top n
    elements after every insertion operation. Insertion is O(n) (list is maintained in
    sorted order), access is O(1). Still fast enough for practical purposes for small beams.
    """
    def __init__(self, size):
        self.size = size
        self.elts = []
        self.scores = []

    def __repr__(self):
        return "Beam(" + repr(list(self.get_elts_and_scores())) + ")"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.elts)

    def get_elts_and_scores(self):
        return zip(self.elts, self.scores)

    def add(self, elt, score):
        """
        Adds the element to the beam with the given score if the beam has room or if the score
        is better than the score of the worst element currently on the beam

        :param elt: element to add
        :param score: score corresponding to the element
        """
        if len(self.elts) == self.size and score < self.scores[-1]:
            # Do nothing because this element is the worst
            return
        # # If the list contains the item with a lower score, remove it
        # i = 0
        # while i < len(self.elts):
        #     if self.elts[i] == elt and score > self.scores[i]:
        #         del self.elts[i]
        #         del self.scores[i]
        #     i += 1
        # If the list is empty, just insert the item
        if len(self.elts) == 0:
            self.elts.insert(0, elt)
            self.scores.insert(0, score)
        # Find the insertion point with binary search
        else:
            lb = 0
            ub = len(self.scores) - 1
            # We're searching for the index of the first element with score less than score
            while lb < ub:
                m = (lb + ub) // 2
                # Check > because the list is sorted in descending order
                if self.scores[m] > score:
                    # Put the lower bound ahead of m because all elements before this are greater
                    lb = m + 1
                else:
                    # m could still be the insertion point
                    ub = m
            # lb and ub should be equal and indicate the index of the first element with score less than score.
            # Might be necessary to insert at the end of the list.
            if self.scores[lb] > score:
                self.elts.insert(lb + 1, elt)
                self.scores.insert(lb + 1, score)
            else:
                self.elts.insert(lb, elt)
                self.scores.insert(lb, score)
            # Drop and item from the beam if necessary
            if len(self.scores) > self.size:
                self.elts.pop()
                self.scores.pop()

    # def get_elts(self):
    #     return self.elts
    #
    # def head(self):
    #     return self.elts[0]


class Seq2SeqSemanticParser(nn.Module):
    def __init__(self, input_indexer, output_indexer, emb_dim, hidden_size,
                 decoder_len_limit, embedding_dropout=0, bidirect=False):
        # We've include some args for setting up the input embedding and encoder
        # You'll need to add code for output embedding and decoder
        super(Seq2SeqSemanticParser, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.decoder_len_limit = decoder_len_limit

        self.input_emb = EmbeddingLayer(emb_dim, len(input_indexer), embedding_dropout)
        self.output_emb = EmbeddingLayer(emb_dim, len(output_indexer), embedding_dropout)
        self.encoder = RNNEncoder(emb_dim, hidden_size, bidirect)
        self.decoder = RNNDecoder(emb_dim, hidden_size, len(output_indexer))

    def encode_input(self, x_tensor, inp_lens):
        """
        Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
        inp_lens_tensor lengths.
        YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
        as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
        :param x_tensor: [batch size, sent len] tensor of input token indices
        :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
        :param model_input_emb: EmbeddingLayer
        :param model_enc: RNNEncoder
        :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
        are real and which ones are pad tokens), and the encoder final states (h and c tuple)
        E.g., calling this with x_tensor (0 is pad token):
        [[12, 25, 0, 0],
        [1, 2, 3, 0],
        [2, 0, 0, 0]]
        inp_lens = [2, 3, 1]
        will return outputs with the following shape:
        enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        enc_final_states = 3 x dim
        """
        input_emb = self.input_emb.forward(x_tensor)
        input_emb = input_emb.unsqueeze(0)
        # print(f"input_emb is: {input_emb.shape}")
        # print(f"pack input is {np.asarray([inp_lens])}")
        inp_lens = np.asarray([inp_lens])
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)

    def encode_output(self, y_tensor):
        output_emb = self.output_emb.forward(y_tensor)
        output_emb = output_emb.unsqueeze(0)
        return output_emb

    def forward(self, x_tensor, inp_lens, y_tensor, out_lens, teacher_forcing_rate):
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len x voc size] or a batched input/output
        [batch size x sent len x voc size]
        :param inp_lens_tensor/out_lens_tensor: either a vecor of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """
        (enc_output, _, final_states) = self.encode_input(x_tensor, inp_lens)
        hidden, cell = final_states[0], final_states[1]
        # return final_states
        # 1 is '<SOS>'
        start = [1]
        input = self.encode_output(start)
        # print(f"first input is: {input.shape}")
        outputs = torch.zeros(out_lens, len(self.output_indexer))

        for i in range(out_lens):
            output, (hidden, cell) = self.decoder.forward(enc_output, input, hidden, cell)
            outputs[i] = output
            # print(outputs[i])
            pred = torch.argmax(output)
            if i < out_lens:
                if random.random() < teacher_forcing_rate:  # teacher forcing
                    label = [y_tensor[i]]
                else:
                    label = [pred]
                input = self.encode_output(label)
        return outputs

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        # Create indexed input
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        all_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, reverse_input=False)

        ans = []
        for i in range(len(test_data)):
            (enc_output, _, final_states) = self.encode_input(all_input_data[i], input_max_len)
            hidden, cell = final_states[0], final_states[1]
            # return final_states
            # 1 is '<SOS>'
            start = [1]
            input = self.encode_output(start)
            # print(f"first input is: {input.shape}")
            outputs = torch.zeros(self.decoder_len_limit, len(self.output_indexer))

            p = 1.0
            y_toks = []
            for i in range(self.decoder_len_limit):
                output, (hidden, cell) = self.decoder.forward(enc_output, input, hidden, cell)
                pred = torch.argmax(output)

                word = self.output_indexer.get_object(pred.item())
                if word != '<PAD>' and word != '<EOS>':
                    y_toks.append(word)
                    label = [pred]
                    input = self.encode_output(label)
                else:
                    break

            # p = torch.prod(torch.Tensor(torch.max(output, dim=1)))
            # print(f"^ {torch.argmax(output, dim=1)}")
            ans.append([Derivation(test_data[i], p, y_toks)])
        return ans

    # trying beam search
    def decode_beam(self, test_data: List[Example], size=2) -> List[List[Derivation]]:
        # Create indexed input
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        all_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, reverse_input=False)

        ans = []
        # size = 2
        for i in range(len(test_data)):
            (enc_output, _, final_states) = self.encode_input(all_input_data[i], input_max_len)
            hidden, cell = final_states[0], final_states[1]
            # return final_states
            # 1 is '<SOS>'
            start = [1]
            input = self.encode_output(start)
            # enc_output is used for attention
            output, (hidden, cell) = self.decoder.forward(enc_output, input, hidden, cell)

            # step 0
            # initialize beams, a list of BeamDec objects which will be appended later
            beams = [BeamDec(size)]
            for idx in range(len(output)):
                beams[0].add(elt=[[idx], hidden, cell], score=output[idx])

            # step 1
            beams.append(BeamDec(size))
            for beam_idx in range(size):
                old_idx_list = beams[0].elts[beam_idx][0]
                input = self.encode_output([old_idx_list[-1]])
                hidden = beams[0].elts[beam_idx][1]
                cell = beams[0].elts[beam_idx][2]
                output, (hidden, cell) = self.decoder.forward(enc_output, input, hidden, cell)
                for idx in range(len(output)):
                    log_prob = beams[0].scores[beam_idx] + output[idx]
                    new_idx_list = old_idx_list + [idx]
                    beams[1].add(elt=[new_idx_list, hidden, cell], score=log_prob)

            last_stage_idx = self.decoder_len_limit - 1
            # following steps
            for stage in range(2, self.decoder_len_limit):
                beams.append(BeamDec(size))
                num_end = 0
                for beam_idx in range(size):
                    old_idx_list = beams[stage-1].elts[beam_idx][0]
                    last_idx = old_idx_list[-1]
                    last_token = self.output_indexer.get_object(last_idx)
                    if last_token != '<PAD>' and last_token != '<EOS>':
                        input = self.encode_output([last_idx])
                        hidden = beams[stage-1].elts[beam_idx][1]
                        cell = beams[stage-1].elts[beam_idx][2]
                        output, (hidden, cell) = self.decoder.forward(enc_output, input, hidden, cell)
                        for idx in range(len(output)):
                            log_prob = beams[stage-1].scores[beam_idx] + output[idx]
                            new_idx_list = old_idx_list + [idx]
                            beams[stage].add(elt=[new_idx_list, hidden, cell], score=log_prob)
                    else:
                        num_end += 1
                        beams[stage].add(elt=beams[stage-1].elts[beam_idx], score=beams[stage-1].scores[beam_idx])
                if num_end == size:
                    last_stage_idx = stage
                    break

            # what we get:
            derivations = []
            for beam_idx in range(size):
                p = np.exp(float(beams[last_stage_idx].scores[beam_idx]))
                y_indices = beams[last_stage_idx].elts[beam_idx][0]
                y_toks = []
                for pos in range(len(y_indices)):
                    token = self.output_indexer.get_object(y_indices[pos])
                    if token != '<PAD>' and token != '<EOS>':
                        y_toks.append(token)
                derivations.append(Derivation(test_data[i], p, y_toks))

            ans.append(derivations)
        return ans


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """

    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.size = full_dict_size
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size + 1, input_dim, padding_idx=full_dict_size)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        for i in range(len(input)):
            if input[i] == -1: input[i] = self.size

        input = torch.IntTensor(input)
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """

    def __init__(self, input_size: int, hidden_size: int, bidirect: bool):
        """
        :param input_size: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                           dropout=0., bidirectional=self.bidirect)
        self.init_weight()

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray(
            [[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True,
                                                             enforce_sorted=False)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        # print(f"packed_embedding: {packed_embedding}")
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = input_lens.data[0]
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
            output = self.reduce_h_W(output[:,0,:])
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t)


class RNNDecoder(nn.Module):
    def __init__(self, emb_size: int, hidden_size: int, output_size: int):
        super(RNNDecoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(emb_size, hidden_size)
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.fc3 = nn.Linear(output_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=0)

        # self.W_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.init_weight()

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, attn_vec, input, hidden, cell):
        #print(f"attn_vec shape: {attn_vec.shape}")     # [19, hidden_size]
        input = input.view(1, 1, -1)
        output, hn = self.rnn(input, (hidden, cell))

        # print(f"attn_vec shape: {attn_vec.shape}")
        max_len = attn_vec.shape[0]
        # print(f"output shape: {output.shape}")

        f = torch.empty(max_len)
        for i in range(max_len):
            f[i] = torch.matmul(output[0,0,:], attn_vec[i,0,:])

        # print(f"f shape: {f.shape}")    # 19
        f = nn.Softmax(dim=0)(f)
        # print(f"f shape: {f.shape}")
        c = torch.matmul(torch.transpose(attn_vec[:,0,:], 0, 1), f)
        # print(f"c shape: {c.shape}")
        output = torch.cat((c, output[0,0,:]), dim=0)
        #print(f"output shape: {output.shape}")
        output = self.fc2(self.fc1(output))
        output = self.logsoftmax(output)
        return output, (hn[0], hn[1])


###################################################################################################################
# End optional classes
###################################################################################################################

def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int,
                             reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])


def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array(
        [[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)]
         for ex in exs])


def train_model_encdec(train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer,
                       args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Set Random Seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    if args.print_dataset:
        print("Train length: %i" % input_max_len)
        print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
        print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    print(f"input_max_len: {input_max_len}")
    print(f"train input data sample: {all_train_input_data[2]}")
    print(f"output_max_len: {output_max_len}")
    print(f"train output data sample: {all_train_output_data[2]}")

    # print(f"train token: {input_indexer.get_object(all_train_input_data[2][1])}")
    print(f"output_indexer: {output_indexer}")

    # First create a model. Then loop over epochs, loop over examples, and given some indexed words
    # call your seq-to-seq model, accumulate losses, update parameters

    model = Seq2SeqSemanticParser(input_indexer, output_indexer, emb_dim=200, hidden_size=200,
                                  decoder_len_limit=args.decoder_len_limit)
    # print(f"len: {type(len(train_data[2].x_indexed))}")

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 5
    # rate = 1
    rate = 0.8
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(train_data)):
            outputs = model.forward(all_train_input_data[i], input_max_len,
                                    all_train_output_data[i], output_max_len, rate)
            optimizer.zero_grad()
            loss = torch.zeros(1)
            sentence_len = len(train_data[i].y_indexed)
            for j in range(outputs.shape[0]):
                gold = int(all_train_output_data[i][j])
                if output_indexer.get_object(gold) == '<PAD>':
                    break
                loss += -outputs[j][gold]

            loss = loss/sentence_len
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # print(f"temp: {outputs.shape}")
            # print(f"pred sentence: {torch.argmax(outputs, dim=1)}")

        print(f"total loss on epoch {epoch}: {total_loss}")
        # evaluate(index_data(train_data, input_indexer, output_indexer, output_max_len),
        #          decoder=model, use_java=False)

    return model

