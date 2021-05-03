import datetime
import numpy as np


def mod_dcgs(pos, neg, tmp, seeds, k_most_similar, switch):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # Modified Discounted Cumulative Gain equations:
    # The first equation corresponds to the actual DCG, where the occurrence is weighted on the position
    # The second one is the negative one.
    #
    # A switch system is implemented for distinguishing the two possible usage: 
    # - the switch = True corresponds to the found seed-word inside the embedding. 
    #   So basically if switch = True the equations are applied on the most similar occurrences, looking for 
    #   them inside the seed list of words.
    #
    # - the switch = False corresponds to the not found seed-word inside the embedding. In this case a row of 
    #   zeros for the negative, and a row of weighted ones for the positive, is computed.
    #------------------------------------------------------------------------------------------------------------
    #
    #
    # if switch true, comes in
    if switch:
        # loop over the k_most_similar words (the list tmp)
        for l, word in enumerate(tmp):
            # Check: if the word is in the seeds list
            if word in seeds:
                # Computing the DCG, positive and negative
                pos.append(1/np.math.log(l+2,2))
                neg.append(0)
            # it the word is not in the seeds list
            else:
                #Computing the DCG, positive and negative
                pos.append(0)
                neg.append(1/np.math.log(l+2,2))
    # if switch not true
    else:
        # Computes list of k_most_similar length, filled with 0s for the positiveDCG and with 1/log2(i+2) for 
        # negative 
        pos = [0 for i in range(k_most_similar)]
        neg = [1/np.math.log(i+2,2) for i in range(k_most_similar)]

    return pos, neg


def neg_dcg(d):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The method performs a sum over the negative DCG values.
    # It takes as input a list of fourples (pos, neg, seed-word, OOV or k-th most similar word of the seed):
    # the output of take_most_similar method.
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    return sum([j[1] for i in list(d.values()) for j in i])

        
def occurred_labels(model, seed):
    #
    #
    #----------------------------------------------------------------------------------------------
    # The method computes a count of preferred and not preferred labels inside the word 
    # embedding model. This method is used only on word embedding models.
    #
    # It takes as input the gensim KeyedVectors.word2vec model and a dictionary of CUIs and the
    # correspondent preferred and not preferred label, inside a list.
    #
    # It returns a dictionary with a CUI as key and a tuple as value, with first element the 
    # correspondent label, either if it is occurred or not, and second element the count of occurred
    # labels inside the embedding. 
    #----------------------------------------------------------------------------------------------
    #
    #
    # Starting time
    a = datetime.datetime.now().replace(microsecond=0)
    dict_ = {}
    # Cycling on dictionary: the keys are CUIs and the values are lists of strings (labels)
    for k, v in zip(seed.keys(), seed.values()):
        # Starting the counter of occurred labels inside the embedding
        count = 0
        # Cycling over the labels of the list v
        for j in v:
            # The try module is due to the not sure presence of the word inside the wordembedding
            try:
                # Check if the label is inside the word embedding
                _ = model[j][0]
                # the first occurred label is stored 
                if count==0:
                    tmp = j
                # The counter is incremented
                count+=1
            except:
                None
        # if no labels are found inside the wordembedding, the first label is stored anyway
        if count==0:
            tmp = v[0]
        # Creation of the dictionary: key: CUI (k), value: a tuple, with the first occurred (or not)
        # word, counter of the number of labels occurred.
        dict_[k] = (tmp, count)        
    # Printing the total time
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return dict_
    

def occurred_words(model, seeds, k_most_similar=10):
    a = datetime.datetime.now().replace(microsecond=0)
    d = {}
    print(len(seeds))
    for seed in seeds:
        most_similar_words = take_most_similar(model, seed, seeds, k_most_similar=10)
        d[seed] = most_similar_words
    return d


def occurrence_word(model, seeds, k=10):
    #
    #
    #-------------------------------------------------------------------------------------------
    # DEPRECATED
    #-------------------------------------------------------------------------------------------
    #
    #
    a = datetime.datetime.now().replace(microsecond=0)
    box = []
    box_ = []
    d = {}
    oov = []
    iov = []
    for seed in seeds:
        try:
            occurrences = []
            occurrences_ = []  
            tmp = list(zip(*model.most_similar(seed,topn=k)))[0]
            
            for l, word in enumerate(tmp):
                if word in seeds:
                    occurrences.append(1/np.math.log(l+2,2))
                    occurrences_.append(0)
                else:
                    occurrences.append(0)
                    occurrences_.append(1/np.math.log(l+2,2))
            
            iov.append(seed)
            most_similar_words = list(map(lambda x, y, z, w:(x,y,z,w), occurrences, occurrences_,[seed for i in range(k)], tmp))
            
        except:
            occurrences = [0 for i in range(k)]
            occurrences_ = [1/np.math.log(i+2,2) for i in range(k)]
            most_similar_words = list(map(lambda x, y, z, w:(x,y,z,w), occurrences, occurrences_,
                                          [seed for i in range(k)], ['OOV' for i in range(k)]))
            oov.append(seed)
            
        box.append(occurrences)
        box_.append(occurrences_)
        d[seed] = most_similar_words
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return box, box_, d, oov, iov


def percentage_dcg(d):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The method performs a weighted count over the positive DCG values.
    # It takes as input a list of fourples (pos, neg, seed-word, OOV or k-th most similar word of the seed):
    # the output of take_most_similar method.
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    c = [1 if (j[0]!=0) else 0 for i in list(d.values()) for j in i ]
    return sum(c)/len(c)
    
    
def pos_dcg(d):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The method performs a sum over the positive DCG values.
    # It takes as input a list of fourples (pos, neg, seed-word, OOV or k-th most similar word of the seed):
    # the output of take_most_similar method.
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    return sum([j[0] for i in list(d.values()) for j in i])

        
def take_most_similar(model, seed, seeds, k_most_similar=10):

    try:
        pos = []
        neg = []
        tmp = list(zip(*model.most_similar(seed,topn=k_most_similar)))[0]
        pos, neg = mod_dcgs(pos, neg, tmp, seeds, k_most_similar, switch=True)
        most_similar_words = list(map(lambda x, y, z, w:(x,y,z,w), pos, neg,[seed for i in range(k_most_similar)], tmp))
    except:
        pos, neg = mod_dcgs(pos, neg, [], [], k_most_similar, switch=False)
        most_similar_words = list(map(lambda x, y, z, w:(x,y,z,w), pos, neg,
                                      [seed for i in range(k_most_similar)], ['OOV' for i in range(k_most_similar)]))
            
    return most_similar_words


