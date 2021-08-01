import datetime, itertools, utils
import numpy as np


def analogy_compute(L_umls, K_umls, model, k_most_similar, logger = None, dict_labels_for_L = None, emb_type = 'cui'):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The method implements the analogic reasoning. 
    # It has two modes, handled by the emb_type switch: a cui mode, where the processed concepts are CUIs, and a
    # labels mode, where the processed concepts are preferred and not preferred labels.
    #
    # The L and K are list of pairs for a specific relation. The model is the gensim model of the considered
    # embedding. 
    # k_most_similar is the value of k used for the k-NN.
    # The logger is not compulsary and it is used for debugging and keeping track of running scripts. It is 
    # tought for background running scripts.
    # dict_labels_for_L is compulsary only for emb_type = 'labels' case. It is the dictionary with all
    # the unique concepts of L set. For each key-concept the dictionary has as value a list of labels iov 
    # 
    # The method returns the storing_list iteratively computed by cos3add.
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    ab = datetime.datetime.now().replace(microsecond=0)
    storing_list = []
    count = 0
    print(np.shape(L_umls))
    print(np.shape(K_umls))
    # Control check for avoiding empty list processing 
    if (0 in np.shape(L_umls)) | (0 in np.shape(K_umls)):
        return storing_list
    
    # Control check for print-constants
    if (len(L_umls)%2)==0:
        length = len(L_umls)
    else:
        length = len(L_umls) + 1
        
    b = datetime.datetime.now().replace(microsecond=0)    
    
    if emb_type == 'cui':
        for concept_L in L_umls:
            temporary = [concept_L[0], concept_L[1]]
            for concept_K in K_umls:
                temporary_ = temporary + [concept_K[0], concept_K[1]]
                if concept_L != concept_K:
                    #if len(check)==0:
                    storing_list = cos3add(concept_L, concept_K, model, k_most_similar, storing_list)
                
            # Printing checkpoints
            count +=1
            if (count%int(np.floor(length/20)) == 0):
                if logger:
                    logger.info(str(datetime.datetime.now().replace(microsecond=0)-b))
                    logger.info('At couple number ' + str(count) + '/' + str(len(L_umls)) + '\n')
                print(datetime.datetime.now().replace(microsecond=0)-b)
                print('At couple number ' + str(count) + '/' + str(len(L_umls)) + '\n')
                b = datetime.datetime.now().replace(microsecond=0)
    
    elif (emb_type == 'labels') and (dict_labels_for_L is not None):
        for concept_L in L_umls:
            #print(dict_labels_for_L[concept_L[0]])
            # Check for evaluating the existence of labels for the choosen concepts inside the vocabulary.
            # They are an overkill: the L and K are choosen for being inside the vocabulary
            #if (len(dict_labels_for_L[concept_L[0]])>0) & (len(dict_labels_for_L[concept_L[1]])>0):
            temporary = [dict_labels_for_L[concept_L[0]], dict_labels_for_L[concept_L[1]]]
            for concept_K in K_umls:
                #if (len(dict_labels_for_L[concept_L[0]])>0) & (len(dict_labels_for_L[concept_L[1]])>0):
                temporary_ = temporary + [dict_labels_for_L[concept_K[0]], dict_labels_for_L[concept_K[1]]]
                kl = list(itertools.product(*temporary_))
                tmp_store = []
                for i in range(len(kl)):
                    if (kl[i][0], kl[i][1]) != (kl[i][2], kl[i][3]):
                        #if len(check)==0:
                        tmp_store = cos3add((kl[i][0], kl[i][1]), # the L-pair
                                            (kl[i][2], kl[i][3]), # the K-pair
                                            model, 
                                            k_most_similar, 
                                            tmp_store)
                        if (tmp_store[-1][-1] == 1) | (len(tmp_store) == len(kl)):
                            storing_list.append(tmp_store[-1])
                            break
                
            # Printing checkpoints
            count +=1
            if (count%int(np.floor(length/20)) == 0):
                if logger:
                    logger.info(str(datetime.datetime.now().replace(microsecond=0)-b))
                    logger.info('At couple number ' + str(count) + '/' + str(len(L_umls)) + '\n')
                print(datetime.datetime.now().replace(microsecond=0)-b)
                print('At couple number ' + str(count) + '/' + str(len(L_umls)) + '\n')
                b = datetime.datetime.now().replace(microsecond=0)
            
    #model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn = 5)
    #vec1, vec2, vec3 = model.wv["king"], model.wv["man"], model.wv["woman"]
    #result = model.similar_by_vector(vec1 - vec2 + vec3, topn=5)
    if logger:
        logger.info(str(datetime.datetime.now().replace(microsecond=0)-ab))
    print(datetime.datetime.now().replace(microsecond=0)-ab) 
    return storing_list
        

def cos3add(concept_L, concept_K, model, k_most_similar, storing_list):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # It is the implementation of the 3CosAdd by Mikolov, aka the analogy computation of the classic
    # analogic relation king-man = queen-woman
    #
    # Example of analogic reasoning: 
    # model_g.wv.most_similar(positive=["king", "woman"], negative=["man"], topn = 5) # L0, K1  L1
    #
    # The method returns a list of tuples, with each tuple with the pair of L, the pair of K and a value (0 or 1)
    # depending by the occurrence: 1 for occurrence, 0 for not occurrence.
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    tmp = list(zip(*model.most_similar(positive=[concept_L[0], concept_K[1]], negative=[concept_L[1]], topn=k_most_similar)))[0]
    if concept_K[0] in tmp:
        storing_list.append((concept_L, concept_K,  1))
    else:
        storing_list.append((concept_L, concept_K,  0))
    return storing_list


def count_occurred_labels(model, seed):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The method computes a count of preferred and not preferred labels inside the word 
    # embedding model. This method is used only on word embedding models.
    #
    # It takes as input the gensim KeyedVectors.word2vec model and a dictionary of CUIs and the
    # correspondent preferred and not preferred label, inside a list.
    #
    # It returns a dictionary with a CUI as key and a tuple as value, with first element the 
    # correspondent label, either if it is occurred or not, and second element the count of occurred
    # labels inside the embedding.---------FIXED
    #
    # It returns a dictionary with a CUI as key and a counter of the CUI's labels (preferred or not) inside the
    # embedding.
    #-----------------------------------------------------------------------------------------------------------
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
                #if count==0:
                #    tmp = j
                
                # The counter is incremented
                count+=1
            except:
                None
        # if no labels are found inside the wordembedding, the first label is stored anyway
        #if count==0:
        #    tmp = v[0]
        
        # Creation of the dictionary: key: CUI (k), value: a tuple, with the first occurred (or not)
        # word, counter of the number of labels occurred.
        #dict_[k] = (tmp, count)        
        dict_[k] = count        
    # Printing the total time
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return dict_


def iov(d):
    #
    #
    #----------------------------------------------------------------------------------------------------------   
    # The method performs a count of inside vocabulary concepts.
    #
    # It takes as input a dictionary of fourples (pos, neg, seed-word, OOV or k-th most similar word of the seed):
    # the output of take_most_similar method.
    #
    # It returns an integer of elements inside the embedding vocabulary
    #----------------------------------------------------------------------------------------------------------
    #
    #
    tmp = oov(d)
    return len(d)-tmp
    

def k_n_l_iov(L_umls_rel, K_umls_rel, model, logger = None, dict_labels_for_L = None, emb_type = 'cui'):
    #
    #
    #----------------------------------------------------------------------------------------------------
    # Accessory method: it preprocesses the K and L sets, discarding the pairs with OOV elements, in 
    # both the sets. This allows a cut of computational cost, fasting the computation of analogy.
    #
    # The method gets as input a list of pairs, given a relation, for L. 
    # A list of pairs for the same relation for K set.
    # The model of considered embedding.
    # A logger is not compulsary: it is tought for background run on vm and for keeping track of errors.
    # dict_labels_for_L is compulsary only for emb_type = 'labels' case. It is the dictionary with all
    # the unique concepts of L set. For each key-concept the dictionary has as value a list of labels iov 
    #
    # The method returns the polished K and L sets of pairs, all iov.
    #----------------------------------------------------------------------------------------------------
    #
    #
    # Timer started
    ab = datetime.datetime.now().replace(microsecond=0)
    # Changing format to the two lists, K_umls and L_umls
    print(np.shape(np.array(list(zip(*L_umls_rel)))))
    l_x = np.array(list(zip(*L_umls_rel))[0])
    l_y = np.array(list(zip(*L_umls_rel))[1])
    l_stacked = np.stack((l_x, l_y))
    k_x = np.array(list(zip(*K_umls_rel))[0])
    k_y = np.array(list(zip(*K_umls_rel))[1])

    k_stacked = np.stack((k_x, k_y))
    stacked = [l_stacked, k_stacked]

    q = []
    if emb_type == 'cui':
        # Extraction Vemb
        Vemb = np.array(list(utils.extract_w2v_vocab(model)))
        # Extracting indeces of Vemb, sorting values in growing way.
        index = np.argsort(Vemb)
        # Sorting Vemb in a growing way. I dont understand the meaning with strings
        sorted_Vemb = Vemb[index]        
        # Making presence masks for discarding pairs oov by the Vemb 
        for j in [[l_x, l_y], [k_x, k_y]]:
            temp = []
            for i in j:
                sorted_index_i = np.searchsorted(sorted_Vemb, i)
                yindex = np.take(index, sorted_index_i, mode="clip")
                mask = Vemb[yindex] != i
                array_ids = np.where(mask)
                tmp = array_ids[0].tolist()
                # The indeces of the first element of the pair are added to the indeces of 
                # the second element of the pair.
                temp = temp + tmp
            q.append(list(set(temp)))
                
    elif (emb_type == 'labels') and (dict_labels_for_L is not None):
        # Polishing the set L keeping only concepts having labels IoV
        #dict_labels_iov = umls_tables_processing.discarding_labels_oov(Vemb, dict_labels_for_L)
        # Making presence masks for discarding pairs oov by the Vemb
        for j in [[l_x, l_y], [k_x, k_y]]:
            temp = []
            for i in j:
                # Keep track the concepts which dont follow the condition
                mask = np.array([False if len(dict_labels_for_L[u])>0 else True for u in i])
                array_ids = np.where(mask)
                tmp = array_ids[0].tolist()
                # The indeces of the first element of the pair are added to the indeces of 
                # the second element of the pair.
                temp = temp + tmp
            q.append(list(set(temp)))
    # Applying the mask to the previous stacked arrays
    tu = []
    for k, s in zip(q, stacked):
        # The sum of the indeces from the two elements of pairs doesnt sort the values
        # For avoiding bugs, the indeces are sorted before deleting the correspondent elements.
        k = np.sort(k)
        # Check for avoiding empty lists processing
        if len(k)>0:
            # Deletion of stored indeces.
            polished_ = np.delete(s, np.array(k), 1)
            new_k_umls = map(tuple, polished_.transpose())
            new_k_umls = list(new_k_umls)
            tu.append(new_k_umls)
        else:
            tu.append([])
                
    print(datetime.datetime.now().replace(microsecond=0)-ab) 
    if logger:
        logger.info(str(datetime.datetime.now().replace(microsecond=0)-ab))
    
    # Returning data with same format of input
    return tu[0], tu[1]


def max_dcg(k_neighs, sub = 2):
    #
    #
    #----------------------------------------------------------------------------------------------------------   
    # Normalization factor for pos and neg DCG.
    #
    # The only input parameter is the value of k, for weighting the position.
    #
    # The method returns a value: it is the max possible DCG obtainable using given k
    #----------------------------------------------------------------------------------------------------------   
    #
    #
    return sum([1/np.math.log(i+sub,sub) for i in range(k_neighs)])
    
    
def mod_dcgs_hit(pos, neg, tmp, seeds, k_most_similar):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # Modified Discounted Cumulative Gain equations:
    # The first equation corresponds to the actual DCG, where the occurrence is weighted on the position
    # The second one is the negative one.
    #
    # The hit version of the method corresponds to the found seed-word inside the embedding. So basically
    # the equations are applied on the most similar occurrences, looking for them inside the seed list of words.
    #------------------------------------------------------------------------------------------------------------
    #
    #
    # loop over the k_most_similar words (the list tmp)
    for l, word in enumerate(tmp):
        # Check: if the word is in the seeds list
        if word in seeds:
            # Computing the DCG, positive and negative
            pos.append(1/np.math.log(l+2,2))
            neg.append(0)
        # if the word is not in the seeds list
        else:
            #Computing the DCG, positive and negative
            pos.append(0)
            neg.append(1/np.math.log(l+2,2))
    return pos, neg
        

    
def mod_dcgs_nohit(pos, neg, k_most_similar):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The nohit version of the method corresponds to the not found seed-word inside the embedding. 
    # In this case a row of zeros for the negative, and a row of weighted ones for the positive, is computed.
    #
    # Computes list of k_most_similar length, filled with 0s for the positiveDCG and with 1/log2(i+2) for 
    # negativeDCG
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    pos = [0 for i in range(k_most_similar)]
    neg = [1/np.math.log(i+2,2) for i in range(k_most_similar)]
    return pos, neg



def neg_dcg(d, normalization = False, norm_fact=1):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The method performs a sum over the negative DCG values.
    # It takes as input a list of fourples (pos, neg, seed-word, OOV or k-th most similar word of the seed):
    # the output of take_most_similar method.
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    a = sum([j[1] for i in list(d.values()) for j in i])
    if normalization:
        return a/(len(d)*norm_fact)
    else:
        return a


    
def occurred_labels(model, seed, k_most_similar=10):
    #
    #
    #----------------------------------------------------------------------------------------------
    # The method computes a count of preferred and not preferred labels inside the word 
    # embedding model and applies the DCGs measures. 
    # This method is used only on word embedding models.
    #
    # It represents an extension of the occurrence_words method.
    #
    # It takes as input the gensim KeyedVectors.word2vec model and a dictionary of CUIs and the
    # correspondent preferred and not preferred label, inside a list. 
    # It ONLY works with strings inside a list (or iterable objects). 
    #
    # It returns a dictionary with a CUI as key and as value a list of fourples (pos, neg, seed-word,
    # OOV or k-th most similar word of the seed) 
    #----------------------------------------------------------------------------------------------
    #
    #
    # Starting time
    a = datetime.datetime.now().replace(microsecond=0)
    dict_ = {}
    new_seed = []
    # The several lists of the CUIs are flattened into a unique list.
    newlist = [item for items in seed.values() for item in items]
    # Cycling on dictionary: the keys are CUIs and the values are lists of strings (labels)
    for k, v in zip(seed.keys(), seed.values()):
        # Starting the counter of occurred labels inside the embedding
        #count = 0
        t = -1
        # If there are no labels for computing occurrence, treat the concept as an OOV 
        if len(v) == 0:
            pos, neg = [], []
            pos, neg = mod_dcgs_nohit(pos, neg, k_most_similar)
            # Data preparation for output: list of 4-tuple with the list of posDCG, negDCG, a list of 
            # k-times duplicated seed, a list of k-times duplicated 'OOV'.
            tmp = list(map(lambda x, y, z, w:(x,y,z,w), 
                           pos, neg,
                           [seed for i in range(k_most_similar)], 
                           ['OOV' for i in range(k_most_similar)]))
            supp = None
            
        # If labels exist so compute occurrence
        else:
            # Cycling over the labels of the list v
            for h, j in enumerate(v):
                # j is one of the labels from the list per CUI
                most_similar_words, _ = take_most_similar(model, j, newlist, k_most_similar=k_most_similar)
            
                # Isolating positiveDCG values for the sum 
                pos = [i[0] for i in most_similar_words]

                # If exists already a positiveDCG value more the actual in the loop, the if is skipped
                # otherwise it is substitued 
                if (t<(sum(pos))) | ((t==(sum(pos)))&(most_similar_words[0][3]!='OOV')):
                    t = sum(pos)
                    supp = j
                    tmp = most_similar_words
                #print('{:s}, {:f}\n'.format(supp, t))
                #print(str(tmp)+'\n')
            #print('\nPicked choice: {:s}, {:f}, {:s}\n'.format(supp, t, tmp[0][3]))        
            # Storing a 4-tuple with the biggest value of posDCG, the correspondent negDCG, an array of k labels (the seed),
            # and the k-most similar words for the label, in a dictionary, using the correspondent CUI as key                 
        dict_[k] = tmp
        new_seed.append(supp)
                
    # Printing the total time
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return dict_, new_seed




def occurred_concept(model, seeds, k_most_similar=10):
    #
    #
    #----------------------------------------------------------------------------------------------------------------
    # It takes as input a gensim.model, a plain list of seeds and a k-number for the number of most similar to each
    # seed-word. 
    #
    # It returns a dictionary with each seed element as key and and as value a list of fourples (pos, neg, seed-word,
    # OOV or k-th most similar word of the seed) 
    #----------------------------------------------------------------------------------------------------------------
    #
    #
    a = datetime.datetime.now().replace(microsecond=0)
    d = {}
    #print(len(seeds))
    for seed in seeds:
        most_similar_words, _ = take_most_similar(model, seed, seeds, k_most_similar=k_most_similar)
        d[seed] = most_similar_words
    return d



def occurrence_word(model, seeds, k=10):
    #
    #
    #-------------------------------------------------------------------------------------------
    # DEPRECATED: the old version of occurred words.
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



def oov(d):
    #
    #
    #----------------------------------------------------------------------------------------------------------   
    # The method performs a count of OOV concepts.
    # It takes as input a dictionary of fourples (pos, neg, seed-word, OOV or k-th most similar word of the seed):
    # the output of take_most_similar method.
    #
    # It returns an integer of elements out of the embedding vocabulary 
    #----------------------------------------------------------------------------------------------------------
    #
    #
    o = [1 for i in list(d.values()) if (i[0][3]=='OOV') ]
    return sum(o)


    
def percentage_dcg(d, k=1):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The method performs a weighted count over the positive DCG values.
    # It takes as input a dictionary of fourples (pos, neg, seed-word, OOV or k-th most similar word of the seed):
    # the output of take_most_similar method.
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    c = [1 if (j[0]!=0) else 0 for i in list(d.values()) for j in i ]
    #print('The normalization of percentage_dcg is: %s' % len(c))
    return sum(c)/(len(d)*k)
    
    
    
def pos_dcg(d, normalization = False, norm_fact=1):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The method performs a sum over the positive DCG values.
    # It takes as input a dictionary of fourples (pos, neg, seed-word, OOV or k-th most similar word of the seed)
    # and a boolean for normalization:
    # the output of take_most_similar method.
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    a = sum([j[0] for i in list(d.values()) for j in i])
    if normalization:
        return a/(len(d)*norm_fact)
    else:
        return a

        
    
def take_most_similar(model, seed, seeds, k_most_similar=10, counter=0):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The method tries to perform the k_most_similar (built-in method from gensim) inside the embedding.
    # If the seed is InsideVocabulary, the modifiedDCG algorithm is applied, computing the positive and the 
    # negative DCG versions.
    #
    # If the seed looked for is OutOfVocabulary, a pair of lists full of zeros for the positive, 
    # and weighted for position for negative, are computed.
    #
    # The method takes as input a gensim.model, a seed-word (or CUI), a list of seeds (for the DCG computation 
    # check), a k-most similar value of concepts found in embedding (10 by default) and a counter (0 by default).
    #
    # The method returns a list of fourples (pos, neg, seed-word, OOV or k-th most similar word of the seed) and
    # a counter of the number of tries: this works only if a counter is kept all over the methods, and fed as 
    # input as well.
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    # Trying the seed inside the embedding
    try:
        pos = []
        neg = []
        # List of k-most similar concepts inside the embeddings
        tmp = list(zip(*model.most_similar(seed,topn=k_most_similar)))[0]
        # Modified DCG is computed
        pos, neg = mod_dcgs_hit(pos, neg, tmp, seeds, k_most_similar)
        # Data preparation for output: list of 4-tuple with the list of posDCG, negDCG, a list of 
        # k-times duplicated seed, a list of k-most similar words.
        most_similar_words = list(map(lambda x, y, z, w:(x,y,z,w), pos, neg,[seed for i in range(k_most_similar)], tmp))
        counter +=1
    # if the seed is not inside the embedding
    except:
        # Modified DCG is computed
        pos, neg = mod_dcgs_nohit(pos, neg, k_most_similar)
        # Data preparation for output: list of 4-tuple with the list of posDCG, negDCG, a list of 
        # k-times duplicated seed, a list of k-times duplicated 'OOV'.
        most_similar_words = list(map(lambda x, y, z, w:(x,y,z,w), pos, neg,
                                      [seed for i in range(k_most_similar)], ['OOV' for i in range(k_most_similar)]))
            
    return most_similar_words, counter


