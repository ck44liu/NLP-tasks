# models.py

from optimizers import *
from nerdata import *
from utils import *

from collections import Counter
from typing import List

import numpy as np


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """

    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray,
                 transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of(
            "UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """

    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs,
                 emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs
        # hard-code transition function
        num = -np.inf
        prob = np.log(1/5)
        self.phi_transition = np.array([[prob,prob,prob,prob,num,prob,num,num,num],
                                        [prob,-100,prob,prob,num,prob,prob,num,num],
                                        [prob,prob,-100,prob,num,prob,num,prob,num],
                                        [prob,prob,prob,-100,prob,prob,num,num,num],
                                        [prob,prob,prob,num,prob,prob,num,num,num],
                                        [prob,prob,prob,prob,num,-100,num,num,prob],
                                        [prob,num,prob,prob,num,prob,prob,num,num],
                                        [prob,prob,num,prob,num,prob,num,prob,num],
                                        [prob,prob,prob,prob,num,num,num,num,prob]])

    def decode(self, sentence_tokens: List[Token]):
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        pred_tags = []
        tags = len(self.tag_indexer)
        score = np.zeros((tags, len(sentence_tokens)))
        dict_backpt = {}

        for idx in range(len(sentence_tokens)):
            word = sentence_tokens[idx].word
            word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(
                word) else self.word_indexer.index_of("UNK")

            for curr_tag in range(tags):
                if idx == 0:  # initial
                    score[curr_tag, idx] = self.init_log_probs[curr_tag] + self.emission_log_probs[curr_tag, word_idx]
                else:  # recurrence
                    score_vals = np.zeros(tags)
                    for prev_tag in range(tags):
                        transition = self.transition_log_probs[prev_tag, curr_tag]
                        emission = self.emission_log_probs[curr_tag, word_idx]
                        score_vals[prev_tag] = transition + emission + score[prev_tag, idx - 1]
                    # check which previous y_(i-1) tag leads to the maximum score_i(s), and then
                    # store max into score, store argmax into a dictionary
                    score[curr_tag, idx] = np.max(score_vals)
                    # use 'current_tag' + ',' + 'word index' as the format of dictionary keys to store argmax backpointers
                    dict_backpt[str(curr_tag) + ',' + str(idx)] = np.argmax(score_vals)

        # final state
        final_score = np.max(score[:, -1])

        # append pred_tags from last token to first token
        pred_tags.append(np.argmax(score[:, -1]))
        for i in range(len(sentence_tokens) - 1, 0, -1):
            pred_tags.append(dict_backpt[str(pred_tags[-1]) + ',' + str(i)])

        # reverse the order of pred_tags, and then store the actual tag labels through tag_indexer
        pred_tags.reverse()
        for tag in range(len(pred_tags)):
            pred_tags[tag] = self.tag_indexer.get_object(pred_tags[tag])

        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))


    def forward_backward(self, sentence_tokens: List[Token], feature_sentence, weights):
        tags = len(self.tag_indexer)
        alpha_score = np.zeros((tags, len(sentence_tokens)))
        beta_score = np.zeros((tags, len(sentence_tokens)))

        for idx in range(len(sentence_tokens)):
            # word = sentence_tokens[idx].word
            # word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")

            for curr_tag in range(tags):
                emission_feat = feature_sentence[idx][curr_tag]
                init_score = 0
                for i in range(len(emission_feat)):
                    init_score += weights[emission_feat[i]]

                if idx == 0:  # initial
                    alpha_score[curr_tag, idx] = init_score
                else:  # recurrence
                    score_vals = np.zeros(tags)
                    alpha_score[curr_tag, idx] = -np.inf
                    for prev_tag in range(tags):
                        emission = init_score
                        transition = self.phi_transition[prev_tag, curr_tag]
                        score_vals[prev_tag] = transition + emission + alpha_score[prev_tag, idx - 1]
                        alpha_score[curr_tag, idx] = np.logaddexp(alpha_score[curr_tag, idx], score_vals[prev_tag])

        for idx in range(len(sentence_tokens) - 1, -1, -1):
            for curr_tag in range(tags):
                if idx == len(sentence_tokens) - 1:  # initial
                    beta_score[curr_tag, idx] = 0
                else:  # recurrence
                    # word = sentence_tokens[idx + 1].word
                    # word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")

                    score_vals = np.zeros(tags)
                    beta_score[curr_tag, idx] = -np.inf
                    for next_tag in range(tags):
                        next_emission_feat = feature_sentence[idx + 1][next_tag]
                        emit_score = 0
                        for i in range(len(next_emission_feat)):
                            emit_score += weights[next_emission_feat[i]]
                        emission = emit_score
                        transition = self.phi_transition[curr_tag, next_tag]
                        score_vals[next_tag] = transition + emission + beta_score[next_tag, idx + 1]
                        beta_score[curr_tag, idx] = np.logaddexp(beta_score[curr_tag, idx], score_vals[next_tag])

        return [alpha_score, beta_score]


def train_hmm_model(sentences: List[LabeledSentence], silent: bool = False) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer), len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer), len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i - 1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    if not silent:
        print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    if not silent:
        print("Tag indexer: %s" % tag_indexer)
        print("Initial state log probabilities: %s" % init_counts)
        print("Transition log probabilities: %s" % transition_counts)
        print("Emission log probs too big to print...")
        print("Emission log probs for India: %s" % emission_counts[:, word_indexer.add_and_get_index("India")])
        print("Emission log probs for Phil: %s" % emission_counts[:, word_indexer.add_and_get_index("Phil")])
        print(
            "   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights, transition_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.phi_transition = transition_weights

    def decode(self, sentence_tokens):
        pred_tags = []
        tags = len(self.tag_indexer)
        score = np.zeros((tags, len(sentence_tokens)))
        dict_backpt = {}

        feature_cache = [[[] for k in range(0, len(self.tag_indexer))] for j in range(0, len(sentence_tokens))]

        for word_idx in range(0, len(sentence_tokens)):
            for tag_idx in range(0, tags):
                feature_cache[word_idx][tag_idx] = extract_emission_features(
                    sentence_tokens, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer,
                    add_to_indexer=False)

        for idx in range(len(sentence_tokens)):
            # word = sentence_tokens[idx].word
            # word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")

            for curr_tag in range(tags):
                emission_feat = feature_cache[idx][curr_tag]
                init_score = 0
                for i in range(len(emission_feat)):
                    init_score += self.feature_weights[emission_feat[i]]

                if idx == 0:  # initial
                    score[curr_tag, idx] = init_score
                else:  # recurrence
                    score_vals = np.zeros(tags)
                    for prev_tag in range(tags):
                        emission = init_score
                        transition = self.phi_transition[prev_tag, curr_tag]
                        score_vals[prev_tag] = transition + emission + score[prev_tag, idx - 1]
                    # check which previous y_(i-1) tag leads to the maximum score_i(s), and then
                    # store max into score, store argmax into a dictionary
                    score[curr_tag, idx] = np.max(score_vals)
                    # use 'current_tag' + ',' + 'word index' as the format of dictionary keys to store argmax backpointers
                    dict_backpt[str(curr_tag) + ',' + str(idx)] = np.argmax(score_vals)

        # final state
        final_score = np.max(score[:, -1])

        # append pred_tags from last token to first token
        pred_tags.append(np.argmax(score[:, -1]))
        for i in range(len(sentence_tokens) - 1, 0, -1):
            pred_tags.append(dict_backpt[str(pred_tags[-1]) + ',' + str(i)])

        # reverse the order of pred_tags, and then store the actual tag labels through tag_indexer
        pred_tags.reverse()
        for tag in range(len(pred_tags)):
            pred_tags[tag] = self.tag_indexer.get_object(pred_tags[tag])

        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))

    def decode_beam(self, sentence_tokens):
        raise Exception("IMPLEMENT ME")

# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences: List[LabeledSentence], silent: bool = False) -> CrfNerModel:
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    if not silent:
        print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in
                     range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0 and not silent:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(
                    sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer,
                    add_to_indexer=True)
    if not silent:
        print("Training")

    # create a simple HmmNerModel object
    Hmm = HmmNerModel(tag_indexer, Indexer(), 0, 0, 0)

    # print("feature cache:", feature_cache)
    epochs = 3
    # set weights' dimension to the length of feature_indexer
    w = np.zeros(len(feature_indexer))
    adagrad = UnregularizedAdagradTrainer(init_weights=w, eta=1)
    for i in range(epochs):
        for idx in range(len(sentences)):
        # for idx in range(5001):
            if idx % 500 == 0:
                print("{}th epoch, {}th idx".format(i, idx))
            sentence = sentences[idx]
            bio_tags = sentence.get_bio_tags()
            # create relevant Counter objects
            feature_sum = Counter()
            product_sum = Counter()
            # gradient = Counter()

            alpha_score, beta_score = Hmm.forward_backward(sentence.tokens, feature_cache[idx], adagrad.weights)

            # compute marginal probability
            for wordpos in range(len(sentence)):
                feature_star = list(feature_cache[idx][wordpos][tag_indexer.index_of(bio_tags[wordpos])])
                # print("feature_star:", feature_star)
                pos_dict = {}
                for val in range(len(feature_star)):
                    pos_dict[feature_star[val]] = 1
                feature_sum.update(Counter(pos_dict))

                denominator = -np.inf
                for tag in range(len(tag_indexer)):
                    denominator = np.logaddexp(denominator, alpha_score[tag, wordpos] + beta_score[tag, wordpos])

                for s in range(len(tag_indexer)):
                    p_yi_s = np.exp(alpha_score[s, wordpos] + beta_score[s, wordpos] - denominator)
                    feature_star_s = list(feature_cache[idx][wordpos][s])
                    product = {}
                    for val in range(len(feature_star_s)):
                        product[feature_star_s[val]] = p_yi_s
                    product_sum.update(Counter(product))

            # subtract from product_sum and update gradient
            feature_sum.subtract(product_sum)
            adagrad.apply_gradient_update(feature_sum, 1)

    return CrfNerModel(tag_indexer, feature_indexer, adagrad.weights, Hmm.phi_transition)


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer,
                              add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size + 1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        # # extension part 1: add new word shapes
        # elif curr_word[i] == "." or curr_word[i] == "&":
        #     new_word += curr_word[i]
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))

    # extension part 2, approach 1: add indicator for each word length
    # word_len = len(curr_word)
    # maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordLength=" + repr(word_len))

    # extension part 2, approach 2: only add four indicators w.r.t. word length
    # word_len = len(curr_word)
    # if word_len <= 3:
    #     maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordLength=" + repr(word_len))
    # else:
    #     maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordLength>3")

    return np.asarray(feats, dtype=int)
