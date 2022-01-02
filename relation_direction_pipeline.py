import argparse, datetime, itertools, measures, umls_tables_processing, utils

from os import scandir

from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import datapath

import numpy as np

from scipy.spatial.distance import cdist

RELAS = umls_tables_processing.OPPOSITE_RELAS
ALL_RELAS = umls_tables_processing.USEFUL_RELA
PATH_EMBEDDINGS = './Embeddings'
SAVING_PATH = 'Utilities/RelationDirectionData/'
NAME_SAVED_FILE = 'rel_direction_data'



def distance_pairs_components(pairs_set, set_pairs_l, model, relas, lab, dict_labels = None, logger= None, all_labels = False):
    '''
    --------------------------------------------------------------------------------------
    The method computes the distance among couples direction and the rela direction
    --------------------------------------------------------------------------------------
    '''
    timer_distance = datetime.datetime.now().replace(microsecond=0)
    dict_ = {}
    for rela in relas:
        dict_[rela] = {}
        tmp = list()
        # Couples
        set_rel = pairs_set[rela]
        # First components PCAs
        rela_dir = set_pairs_l[rela]
        if (0 not in np.shape(set_rel)) and (0 not in np.shape(rela_dir)):
            if (lab == 'labels') and (dict_labels is not None):
                if all_labels:
                    couple_dirs = medoid_aggregation(dict_labels, set_rel, model)
                    if (0 not in np.shape(couple_dirs)) & (0 not in np.shape(rela_dir)):
                        tmp.append([cdist(np.array(c).reshape(1,-1), rela_dir) for c in couple_dirs])
                    else:
                        tmp.append(list())
                else:
                    # Try for Iov
                    try:
                        tmp.append([cdist( np.array(model[dict_labels[couple[0]][0]]-model[dict_labels[couple[1]][0]]).reshape(1,-1), rela_dir) for couple in set_rel])
                    # if oov
                    except:
                        tmp.append([])
            else:
                # Try for Iov
                try:
                    for couple in set_rel:
                        tmp.append(cdist(np.array(model[couple[0]]-model[couple[1]]).reshape(1,-1), rela_dir))

                except:
                    tmp.append(42)

        dict_[rela] = tmp

    logger.info('The processing for distance between relas couples and corresponding relas\' L pca is finished and spent %s\n',
                    str(datetime.datetime.now().replace(microsecond=0)-timer_distance))
    return dict_
        
    
    
def distance_among_relations(set_pairs_l, set_pairs_k, opposite_relas, logger = None, all_labels = False):
    '''
    -----------------------------------------------------------------------------------------------------------------
    The method gets a list of opposite relas tuples, two dictionaries with firsts principal components, one for rela.
    One dictionary for one set and the other for the second. They usually represent L and K.
    A logger is not compulsory and the all_labels strategy switch, in case the analyzed models are textual.
    
    It computes the distances among relas direction, for each set.
    
    It returns the computed distance among the arrays as a list of tuples
    -----------------------------------------------------------------------------------------------------------------
    '''
    timer_global = datetime.datetime.now().replace(microsecond=0)
    temp = {}
    for opposites in opposite_relas:
        timer_opposite = datetime.datetime.now().replace(microsecond=0)
        logger.info('The opposite relations are %s - %s \n', opposites[0], opposites[1])
        if (0 not in np.shape(set_pairs_l[opposites[0]])):
            q = 1. - cdist(set_pairs_l[opposites[0]], set_pairs_l[opposites[0]], 'cosine')
            if (0 not in np.shape(set_pairs_l[opposites[1]])):
                w = 1. - cdist(set_pairs_l[opposites[0]], set_pairs_l[opposites[1]], 'cosine')
            else:
                w = 42
                        
            if (0 not in np.shape(set_pairs_k[opposites[0]])):
                e = 1. - cdist(set_pairs_l[opposites[0]], set_pairs_k[opposites[0]], 'cosine')
            else:
                e = 42
                        
            if (0 not in np.shape(set_pairs_k[opposites[1]])):
                r = 1. - cdist(set_pairs_l[opposites[0]], set_pairs_k[opposites[1]], 'cosine')
            else:
                r = 42
                        
            first= [q, w, e, r]
                
        else:
            first = []


        if (0 not in np.shape(set_pairs_l[opposites[1]])):
            w = 1. - cdist(set_pairs_l[opposites[1]], set_pairs_l[opposites[1]], 'cosine')
            if (0 not in np.shape(set_pairs_l[opposites[0]])):
                q = 1. - cdist(set_pairs_l[opposites[1]], set_pairs_l[opposites[0]], 'cosine')
            else:
                q = 42
                                
            if (0 not in np.shape(set_pairs_k[opposites[0]])):
                e = 1. - cdist(set_pairs_l[opposites[1]], set_pairs_k[opposites[0]], 'cosine')
            else:
                e = 42
                                
            if (0 not in np.shape(set_pairs_k[opposites[1]])):
                r = 1. - cdist(set_pairs_l[opposites[1]], set_pairs_k[opposites[1]], 'cosine')
            else:
                r = 42
            second= [q, w, e, r]
                    
        else:
            second = []
        
        temp[opposites[0]+' / '+ opposites[1]] = [first, second]
        
        logger.info('The processing for opposite relations %s - %s is finished and spent %s\n',
                    opposites[0],
                    opposites[1],
                    str(datetime.datetime.now().replace(microsecond=0)-timer_opposite))
    logger.info('The processing time for distance among relations algorithm is %s\n',
                str(datetime.datetime.now().replace(microsecond=0)-timer_global))        
    return temp    



def relation_pipe(embeddings, L_umls, K_umls, dict_conso, logger, all_labels = False):
    full_relations = {}
    full_pairs = {}
    timer_global = datetime.datetime.now().replace(microsecond=0)
    for type_emb in embeddings:
        for emb in type_emb[1]:    
            timer_emb = datetime.datetime.now().replace(microsecond=0)
            # Embedding
            model = KeyedVectors.load_word2vec_format(PATH_EMBEDDINGS+type_emb[0]+emb, binary=emb.endswith('.bin'))
            logger.info('Embedding %s loaded\n', emb)
            
            if type_emb[0] == '/cuis/':
                lab = 'cui'
                dict_labels_inters_vemb = None
                
            if type_emb[0] == '/words/':
                lab = 'labels'
                # Filter the dictionary of labels keeping only the labels-words present into the embedding
                Vemb = utils.extract_w2v_vocab(model)
                dict_labels_inters_vemb = umls_tables_processing.discarding_labels_oov(Vemb,
                                                                                       dict_conso,
                                                                                       all_labels = all_labels)
        
            pcas_l = rela_direction(model, lab, ALL_RELAS, L_umls, logger, 
                                    dict_ = dict_labels_inters_vemb, all_labels = all_labels)
            pcas_k = rela_direction(model, lab, ALL_RELAS, K_umls, logger, 
                                    dict_ = dict_labels_inters_vemb, all_labels = all_labels)

            temp = distance_among_relations(pcas_l, 
                                            pcas_k,
                                            RELAS,
                                            logger = logger,
                                            all_labels = all_labels)
            
            # Distance 
            #temp_dict = distance_pairs_components(K_umls,
            #                                      pcas_l,
            #                                      model,
            #                                      ALL_RELAS,
            #                                      lab,
            #                                      dict_labels = dict_labels_inters_vemb,
            #                                      logger = logger,
            #                                      all_labels = all_labels)

            full_relations[emb] = temp
            #full_pairs[emb] = temp_dict
            
            # Here it can be implementable the distance_pairs_components method
                
        logger.info('The time for computation of %s is %s\n', 
                    emb, 
                    str(datetime.datetime.now().replace(microsecond=0)-timer_emb))
        
    logger.info('The time for global pcas distance computation is %s\n', 
                str(datetime.datetime.now().replace(microsecond=0)-timer_global))
    return full_relations#, full_pairs



def rela_direction(model, lab, relas, pairs_set, logger, dict_ = None, all_labels = False):
    '''
    -----------------------------------------------------------------------------------------------------------------
    The method gets a model, a list of relas, a list of concepts tuples, usually K or L.
    The method performs the filtering of OOV pairs, with the embedding vocabulary, both, for words and cuis.
    
    It distinguishes two different strategies for words: all label and only the best ranked one. It is represented
    by the boolean variable all_labels. False by default, since all_labels is a deprecated strategy for high 
    computational cost in analogical processing.
    
    It returns a dictionary of firsts components (PCAs), one for each relation. 
    -----------------------------------------------------------------------------------------------------------------
    '''
    pcas = {}
        
    for rela in relas:
        logger.info('The relation is %s\n', rela)
        timer_rela = datetime.datetime.now().replace(microsecond=0)
        
        # Filtering the couples set on embedding vocabulary
        set_p, _ = measures.k_n_l_iov(pairs_set[rela], 
                                      pairs_set[rela],
                                      model, 
                                      logger = logger,
                                      dict_labels_for_L = dict_,
                                      emb_type = lab)
        
        # Check if the set has available iov couples
        if len(set_p)>0:
            if (lab == 'labels') and (dict_ is not None):
                if all_labels:
                    tempor = medoid_aggregation(dict_, set_p, model)
                    pcas[rela] = np.array(measures.relation_direction(model, tempor, all_labels = all_labels)).reshape(1,-1)     
                # Only the ranked first label
                else:
                    set_p = list(set([(dict_[i[0]][0], dict_[i[1]][0]) for i in set_p]))
                    pcas[rela] = np.array(measures.relation_direction(model, set_p)).reshape(1,-1)
            # Case with CUIs
            else:
                pcas[rela] = np.array(measures.relation_direction(model, set_p)).reshape(1,-1)
        else:
            pcas[rela] = np.array(list())
        logger.info('The processing for rela %s is finished and spent %s\n',
            rela,
            str(datetime.datetime.now().replace(microsecond=0)-timer_rela))
    
    return pcas
            

def medoid_aggregation(dict_, set_p, model):
    '''
    -----------------------------------------------------------------------------------
    The method, joint with cosine_medoid, allows to obtain one representing vector for
    a cluster of vectors-labels.
    
    It gets as inputs a dictionary of labels, where to one UMLS concept, corresponds
    a set of labels, preferred and not, a set of couples linked by a certain relation,
    and an embedding-model.
    
    It returns a list of vectors-directions, one per couple.
    -----------------------------------------------------------------------------------
    '''
    tempor = list()
    #set_p = list(set([(dict_[i[0]], dict_[i[1]]) for i in set_p]))
    # Iteration through the concepts couples
    for i in set_p:
        # List of the two labels lists.
        temporary_ = [list(set(dict_[i[0]])), list(set(dict_[i[1]]))]
        # Combination of each label, for doing the label couples compatible with 
        # relation direction computation for all labels case.
        kl = list(itertools.product(*temporary_))
        kl = list(set(kl))
        temp = list()
        # Iterating over all the label couples
        if len(kl)>0:
            for j in kl:
                temp.append(np.array(model[j[0]] - model[j[1]]))
            # Discarding all the difference vectors, aka direction vectors.
            best_dir, _ = measures.cosine_medoid(temp)
            tempor.append(best_dir)
        else:
            tempor.append(list()) 
    return tempor

    
if __name__ == '__main__':
    # Parsing values for fast and intuitive launch of the script: 
    # paralleling, embedding_type, copd_K_switch are inserted by command line.
    parser = argparse.ArgumentParser(description='Launching relations direction computation')
    
    parser.add_argument('--t',
                        dest='embedding_type',
                        type=str,
                        default = 'both',
                        required=False,
                        help='The type of analyzed embedding: it could be "both", "cuis", or "words"')
    
    parser.add_argument('--lab', 
                        dest='all_labels',
                        type=bool,
                        default = False,
                        required=False,
                        help='Only preferred or all labels')
    
    args = parser.parse_args()
    # Check on quality of inserted data
    # Embedding type
    assert args.embedding_type in ['both', 'cuis', 'words'], "Insert a string like 'both', 'cuis', or 'words'"
    
    # Logger instantiation
    logger = utils.setup_custom_logger('myapp')
    logger.info(args)
    start = datetime.datetime.now().replace(microsecond=0)

    logger.info('Start\n')
    
        
    # CUIs 
    concepts = umls_tables_processing.concepts_related_to_concept(concept = umls_tables_processing.COPD,
                                                                  two_way = True,
                                                                  polishing_rels = False,
                                                                  switch_key = 'con',
                                                                  extract_labels = False)
    logger.info('Seeds built\n')
    
    # Set K and L
    K_umls = umls_tables_processing.count_pairs(umls_tables_processing.USEFUL_RELA, cuis_list = concepts)
    L_umls = umls_tables_processing.count_pairs(umls_tables_processing.USEFUL_RELA)
    logger.info('K and L sets created\n')
    
        
    # Building the dictionary for labels case
    # Collecting all the CUIs involved in set L
    if (args.embedding_type == 'words') or (args.embedding_type == 'both'):
        dict_conso = umls_tables_processing.cui_strings()    
    
    else:
        dict_conso = None
    
    logger.info('Dictionary of labels built\n')
    
    # Loading w2v files
    embeddings = []
    
    # CUI or Word Embeddings discrimination
    if (args.embedding_type == 'cuis') or (args.embedding_type == 'both'):
        logger.info('CUI embeddings\n'), 
        cuis = ('/cuis/', [f.name for f in scandir(PATH_EMBEDDINGS+'/cuis') if (f.is_file())&(f.name != 'README.md')])
        embeddings.append(cuis)
        
    if (args.embedding_type == 'words') or (args.embedding_type == 'both'):
        logger.info('Word embeddings\n'), 
        labels = ('/words/', [f.name for f in scandir(PATH_EMBEDDINGS+'/words') if (f.is_file())&(f.name != 'README.md')])
        embeddings.append(labels)
    
    #rel, pa = relation_pipe(embeddings, L_umls, K_umls, dict_conso, logger, args.all_labels)    
    rel = relation_pipe(embeddings, L_umls, K_umls, dict_conso, logger, args.all_labels)    
        
    # Storing data
    utils.inputs_save(rel, SAVING_PATH + NAME_SAVED_FILE + '_rels_distances_' +str(datetime.datetime.now()))
    #utils.inputs_save(pa, SAVING_PATH + NAME_SAVED_FILE + '_rels_pairs_distances_'+str(datetime.datetime.now()))
    # End
    logger.info('Execution time of relation direction computation is: %s\n',
                str(datetime.datetime.now().replace(microsecond=0) - start))
                    
                    
                    


                

    
    
    
        