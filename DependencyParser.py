import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
            # Declare placeholders for different variables
            self.train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size, Config.n_Tokens])
            self.train_labels = tf.placeholder(tf.int32, shape=[Config.batch_size, parsing_system.numTransitions()])
            self.test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens])

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """
            # Initialize embedding and input and output weight matrices. Using a truncated normal distribution 
            # with stddev 0.1 for the weight matrices
            train_embed_lookup = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embed = tf.reshape(train_embed_lookup, [Config.batch_size, -1])
            train_weights_input = tf.Variable(tf.truncated_normal(shape=[Config.hidden_size, Config.n_Tokens*Config.embedding_size], stddev=0.1))
            train_weights_output = tf.Variable(tf.truncated_normal(shape=[parsing_system.numTransitions(), Config.hidden_size], stddev=0.1))
            train_bias_input = tf.Variable(tf.zeros([Config.hidden_size, 1]))

            # Call forward pass on the function to get the prediction values
            self.train_pred = self.forward_pass(train_embed, train_weights_input, train_bias_input, train_weights_output)
            
            # Mask out the invalid(-1) transitions in the labels
            train_labels = tf.nn.relu(self.train_labels)

            # Calculate the softmax and cross entropy loss of the prediction scores.
            ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.train_pred, labels=train_labels)

            # Calculate the L2 loss values for the weights and embeddings 
            l2_input = Config.lam * tf.nn.l2_loss(train_weights_input)
            l2_output = Config.lam * tf.nn.l2_loss(train_weights_output)
            l2_bias = Config.lam * tf.nn.l2_loss(train_bias_input)
            l2_embed = Config.lam * tf.nn.l2_loss(train_embed)

            # Add L2 loss to the cross entropy loss
            ce_l2_loss = ce_loss + l2_input + l2_output + l2_bias + l2_embed

            # Take average loss over the entire batch
            self.loss = tf.reduce_mean(ce_l2_loss)

            # Perform gradiemt descent with gradient clipping
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)
            
            #self.app = optimizer.apply_gradients(grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, train_weights_input, train_bias_input, train_weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print("Initailized")

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print ("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print ("\nTesting on dev set at step ", step)
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print (result)

        print ("Train Finished.")

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print ("Starting to predict on test set")
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print ("Saved the test results.")
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_input, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        y = tf.constant([3.0])

        # Multiply input weights with the embeddings
        h1 = tf.matmul(weights_input, embed, transpose_b=True)
        
        # Add input bias
        h2 = h1 + biases_input
        
        # Raise values to the power of 3
        h3 = tf.pow(h2, 3)
        
        # Uncomment to change activation function to sigmoid
        #h3 = tf.nn.sigmoid(h2)
        
        # Uncomment to change activation funtion to tanh
        #h3 = tf.tanh(h2)
        
        # Uncomment to change activation function to ReLu
        #h3 = tf.nn.relu(h2)
        
        # Multiply with output weights and transpose result
        h4 = tf.matmul(weights_output, h3)
        h = tf.transpose(h4)

        return h   



def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    features = []
    idx = 0

    for i in range(3):
        features.append(c.getStack(i))

    for i in range(3):
        features.append(c.getBuffer(i))

    n = len(features)
    for i in range(2):
        l1 = c.getLeftChild(features[i], 1)
        r1 = c.getRightChild(features[i], 1)
        l2 = c.getLeftChild(features[i], 2)
        r2 = c.getRightChild(features[i], 2)
        ll1 = c.getLeftChild(l1, 1)
        rr1 = c.getRightChild(r1, 1)
        features.extend([l1, r1, l2, r2, ll1, rr1])

    n = len(features)
    for i in range(n):
        features.append(c.getPOS(features[i]))

    n = 18
    for i in range(6,18):
        features.append(c.tree.getLabel(features[i]))

    for i in range(18):
        features[i] = getWordID(c.getWord(features[i]))

    for i in range(18, 36):
        features[i] = getPosID(features[i])

    for i in range(36, 48):
        features[i] = getLabelID(features[i])

    return features


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print (i, label)
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    # Please remove the second argument in below call if you see an exception. Argument added for python 3
    dictionary, word_embeds = pickle.load(open(filename, 'rb'), encoding='latin1')

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = list(wordDict.keys())
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print ("Found embeddings: ", foundEmbed, "/", len(knownWords))

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, testTrees = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(list(labelDict.values())):
        labelInfo.append(list(labelDict.keys())[idx])

    parsing_system = ParsingSystem(labelInfo[1:])
    
    print ("Generating Traning Examples")
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print ("Done.")

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

