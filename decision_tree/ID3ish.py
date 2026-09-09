import numpy as np
import pandas as pd

# The idea is to make a function that I can use anywhere to make a decision tree
# for now the tree will only output boolean functions and work on discrete input

def entropy(attribute_values, position, S):
    entropy = 0
    nS = len(S)
    for i in range(len(attribute_values)):
        s_i = len([x for x in S if x[position == attribute_values[i]]])
        entropy -= (s_i/nS)* np.log2(s_i/nS)
    return entropy

def bool_decision_tree(attributeNames, dictAttributes, training_examples):

    # here Dictattributes is the schema of attributes that I will recieve for any training example
    # attributeNames is the names of attributes in order
    # training_examples is the list of all the training examples

    # get information from attributes

    nAttributes = len(attributeNames)

    # while loop to make tree

    while True:

        tree = []

        # for loop to find information gain for each attribute
        entropy(training_examples, -1, dictAttributes['target'])

        min_information_gain = 0

        for i in range(attributeNames):
            entropy_i = 0
            for attribute in dictAttributes[attributeNames[i]]:
                entropy_i += entropy([x for x in training_examples if x[i] == attribute], -1, dictAttributes['target'])
            
    return 0