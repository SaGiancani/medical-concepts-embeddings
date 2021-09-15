import argparse, datetime, measures, umls_tables_processing, utils

from collections import defaultdict

from os import scandir

from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import datapath

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

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
    
    args = parser.parse_args()

    # Check on quality of inserted data
    # Embedding type
    assert args.embedding_type in ['both', 'cuis', 'words'], "Insert a string like 'both', 'cuis', or 'words'"
    
    # Logger instantiation
    logger = utils.setup_custom_logger('myapp')
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
    #LsubK = {rela: list(set(L_umls[rela])-set(K_umls[rela])) for rela in umls_tables_processing.USEFUL_RELA}
    logger.info('K and L sets created\n')
    
        
    # Building the dictionary for labels case
    # Collecting all the CUIs involved in set L
    if (args.embedding_type == 'words') or (args.embedding_type == 'both'):
        jh = []
        for v in LsubK.values():
            jh.append(list(set(list(zip(*v))[0])))
            jh.append(list(set(list(zip(*v))[1])))
            tmp = set([j for i in jh for j in i ])
        dict_strings = umls_tables_processing.cui_strings()    
        dict_labels_LsubK, _ = umls_tables_processing.extracting_strings(list(tmp), dict_strings = dict_strings)
    
    else:
        dict_labels_LsubK = None
    
    logger.info('Dictionary of labels from set L built\n')
    
    # Loading w2v files
    PATH_EMBEDDINGS = './Embeddings'
    embeddings = []
    
    # CUI or Word Embeddings discrimination
    if (args.embedding_type == 'cuis') or (args.embedding_type == 'both'):
        logger.info('CUI embeddings\n'), 
        cuis = ('/cuis/', [f.name for f in scandir(PATH_EMBEDDINGS+'/cuis') if (f.is_file())&(f.name != 'README.md')])
        embeddings.append(cuis)
        
    elif (args.embedding_type == 'words') or (args.embedding_type == 'both'):
        logger.info('Word embeddings\n'), 
        labels = ('/words/', [f.name for f in scandir(PATH_EMBEDDINGS+'/words') if (f.is_file())&(f.name != 'README.md')])
        embeddings.append(labels)
    
    full = defaultdict(list)
    
    for type_emb in embeddings:
        #full[type_emb[1]]
        for emb in type_emb[1]:    
            timer_emb = datetime.datetime.now().replace(microsecond=0)
            # Embedding
            model = KeyedVectors.load_word2vec_format(PATH_EMBEDDINGS+type_emb[0]+emb, binary=emb.endswith('.bin'))
            logger.info('Embedding %s loaded\n', emb)
            for opposite_relas in umls_tables_processing.OPPOSITE_RELAS:
                logger.info('The opposite relations are %s - %s \n', opposite_relas[0], opposite_relas[1])
                if type_emb[0] == '/cuis/':
                    timer_opposite = datetime.datetime.now().replace(microsecond=0)
                    # Filtering the couples set on embedding vocabulary
                    l, _ = measures.k_n_l_iov(L_umls[opposite_relas[0]], 
                                              L_umls[opposite_relas[0]],
                                              model, 
                                              emb_type = 'cui')
                    
                    l_o, _ = measures.k_n_l_iov(L_umls[opposite_relas[1]], 
                                                L_umls[opposite_relas[1]],
                                                model, 
                                                emb_type = 'cui')
                    
                    
                    k, _ = measures.k_n_l_iov(K_umls[opposite_relas[0]], 
                                              K_umls[opposite_relas[0]],
                                              model, 
                                              emb_type = 'cui')

                    k_o, _ = measures.k_n_l_iov(K_umls[opposite_relas[1]], 
                                                K_umls[opposite_relas[1]],
                                                model, 
                                                emb_type = 'cui')
                    
                    # Compute PCA and preparing arrays for distance matrix                                         
                    # L/K
                    lk_dir, lko_dir, k_dir, ko_dir = list(), list(), list(), list()
                    if (len(k)>0) and (len(l)>0):
                        lk_dir = np.array(measures.relation_direction(model, 
                                                                      list(set(l).difference(set(k))))).reshape(1,-1)
                    
                    if (len(k_o)>0) and (len(l_o)>0):
                        lko_dir = np.array(measures.relation_direction(model,
                                                                       list(set(l_o).difference(set(k_o))))).reshape(1,-1)

                    # K
                    if (len(k)>0):
                        k_dir = np.array(measures.relation_direction(model, k)).reshape(1,-1)
    
                    if (len(k_o)>0):
                        ko_dir = np.array(measures.relation_direction(model, k_o)).reshape(1,-1)

                    # Cosine similarity matrix
                    # Rows are L/K and columns are K
                    #sim_mat = cosine_similarity(a,b)
                    if (0 not in np.shape(lk_dir)):
                        q = 1. - cdist(lk_dir, lk_dir, 'cosine')
                        
                        if (0 not in np.shape(lko_dir)):
                            w = 1. - cdist(lk_dir, lko_dir, 'cosine')
                        else:
                            w = 42
                        
                        if (0 not in np.shape(k_dir)):
                            e = 1. - cdist(lk_dir, k_dir, 'cosine')
                        else:
                            e = 42
                        
                        if (0 not in np.shape(ko_dir)):
                            r = 1. - cdist(lk_dir, ko_dir, 'cosine')
                        else:
                            r = 42
                        
                        first= [q, w, e, r]
                    else:
                        first = []


                    if (0 not in np.shape(lko_dir)):
                        w = 1. - cdist(lko_dir, lko_dir, 'cosine')
                        if (0 not in np.shape(lk_dir)):
                            q = 1. - cdist(lko_dir, lk_dir, 'cosine')
                        else:
                            q = 42
                            
                        if (0 not in np.shape(k_dir)):
                            e = 1. - cdist(lko_dir, k_dir, 'cosine')
                        else:
                            e = 42
                            
                        if (0 not in np.shape(ko_dir)):
                            r = 1. - cdist(lko_dir, ko_dir, 'cosine')
                        else:
                            r = 42
                        second= [q, w, e, r]
                    
                    else:
                        second = []
                    
                    full[emb].append((opposite_relas[0]+' / '+ opposite_relas[1], 
                                      [first,second],
                                      len(list(set(l).difference(set(k)))),
                                      len(set(k))))
                    
                logger.info('The processing for opposite relations %s - %s is finished and spent %s\n',
                            opposite_relas[0],
                            opposite_relas[1],
                            str(datetime.datetime.now().replace(microsecond=0)-timer_opposite))
                
        logger.info('The time for computation of %s is %s\n', 
                    emb, 
                    str(datetime.datetime.now().replace(microsecond=0)-timer_emb))
        
    # Storing data
    utils.inputs_save(full, 'Utilities/rel_direction_data')
    # End
    logger.info('Execution time of relation direction computation is: %s\n',
                str(datetime.datetime.now().replace(microsecond=0) - start))
                    
                    
                    


                

    
    
    
        