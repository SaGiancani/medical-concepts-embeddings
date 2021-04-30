import datetime
import numpy as np

def occurrence_word(model, seeds, k=10):
    a = datetime.datetime.now().replace(microsecond=0)
    box = []
    box_ = []
    d = {}
    oov = []
    iov = []
    for seed in seeds:
        most_similar_words = []
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



def take_most_similar(seed, k=10, most_similar_switch = True):
    oov = []
    try:
        occurrences = []
        occurrences_ = []  
        tmp = list(zip(*model.most_similar(seed,topn=k)))[0]
        
        if most_similar_switch:
            most_similar_occurrences(k, switch=True)
            
            iov.append(seed)
            most_similar_words = list(map(lambda x, y, z, w:(x,y,z,w), occurrences, occurrences_,[seed for i in range(k)], tmp))
    
    except:
        
        if most_similar_switch:
            occurrences, occurrences_ = most_similar_occurrences(k, switch=False)
        
            oov.append(seed)
            most_similar_words = list(map(lambda x, y, z, w:(x,y,z,w), occurrences, occurrences_,
                                      [seed for i in range(k)], ['OOV' for i in range(k)]))
    
        
        
def most_similar_occurrences(k_most_similar, switch):
    if switch:
        for l, word in enumerate(tmp):
            if word in seeds:
                occurrences.append(1/np.math.log(l+2,2))
                occurrences_.append(0)
            else:
                occurrences.append(0)
                occurrences_.append(1/np.math.log(l+2,2))
    else:
        occurrences = [0 for i in range(k)]
        occurrences_ = [1/np.math.log(i+2,2) for i in range(k)]

    return occurrences, occurrences_
        
        
        
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
    

