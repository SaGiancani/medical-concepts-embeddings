import argparse, datetime, measures, os, umls_tables_processing, utils
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import datapath


def analog_loop(path, binary_bool, name, type_emb, L, K, k_most_similar, K_type, logger, analog_comp_dict,
                sets_relations, dict_labels_for_L = None):
    
    # Load the w2v model
    model = KeyedVectors.load_word2vec_format(path, binary=binary_bool)
    
    # Instantiation and log print
    analog_comp_dict[name] = {}
    logger.info('\n\n The name of embedding is: %s\n', name)
    dict_t = {}
    dict_t[name] = {}
    
    # Loop over the relations
    for rela in umls_tables_processing.USEFUL_RELA:
        logger.info('\n The RELA is: %s\n', rela)
        
        # Check type of embedding
        if type_emb=='/cuis/':
            c = datetime.datetime.now().replace(microsecond=0)
            l0, k0 = measures.k_n_l_iov(L[rela], 
                                        K[rela],
                                        model, 
                                        logger = logger,
                                        emb_type = 'cui')
            
            # sets_relations keeps track of the number of pairs of K and L sets. 
            # The number of filtered pairs on Vemb, per relation are stored
            sets_relations[rela].append((name+'_l'+K_type, np.shape(l0)))
            sets_relations[rela].append((name+'_k'+K_type, np.shape(k0)))
            
            # Save the pickle variable
            utils.inputs_save(sets_relations, 'Utilities/sets_relations' + name + K_type)
                                
            # Compute the analogy and store the results
            tmp = measures.analogy_compute(l0, k0, 
                                           model,
                                           k_most_similar,
                                           logger = logger,
                                           emb_type = 'cui')
            dict_t[name][rela] = tmp
            
            # Compute the sum of analogy hits and store it.
            if len(tmp)>0:
                analog_comp_dict[name][rela] = (sum(list(zip(*tmp))[2]), len(tmp))
            else:
                analog_comp_dict[name][rela] = (0, len(tmp))                        

            utils.inputs_save(dict_t, 'Utilities/' + name + K_type)                    
            utils.inputs_save(analog_comp_dict, 'Utilities/count_analog_' + name + K_type)
            
            # Log of end of 'relation' operation
            logger.info('The time for RELA %s, for embedding %s is %s', 
                        rela,
                        name,
                        str(datetime.datetime.now().replace(microsecond=0)-c))
                    
        # Check type of embedding: for word embeddings the dictionary of labels per cui is required
        elif (type_emb=='/words/') and (dict_labels_for_L is not None):
            c = datetime.datetime.now().replace(microsecond=0)
            
            # Filter the dictionary of labels keeping only the labels-words present into the embedding
            Vemb =utils.extract_w2v_vocab(model)
            dict_labels_inters_vemb = umls_tables_processing.discarding_labels_oov(Vemb, dict_labels_for_L)
            # Filtering L and K sets for present labels inside the embedding
            l0, k0 = measures.k_n_l_iov(L[rela], 
                                        K[rela],
                                        model, 
                                        logger = logger,
                                        dict_labels_for_L = dict_labels_inters_vemb,
                                        emb_type = 'labels')

            # Store number of filtered pairs
            sets_relations[rela].append((name+'_l'+K_type, np.shape(l0)))
            sets_relations[rela].append((name+'_k'+K_type, np.shape(k0)))

            utils.inputs_save(sets_relations, 'Utilities/sets_relations' + name + K_type)
                    
            tmp = measures.analogy_compute(l0, k0, 
                                           model, 
                                           k_most_similar, 
                                           logger = logger,
                                           dict_labels_for_L = dict_labels_inters_vemb, 
                                           emb_type = 'labels')                    
                    
            dict_t[name][rela] = tmp
                    
            if len(tmp)>0:
                analog_comp_dict[name][rela] = (sum(list(zip(*tmp))[2]), len(tmp))
            else:
                analog_comp_dict[name][rela] = (0, len(tmp))
                        
            utils.inputs_save(dict_t, 'Utilities/' + name + K_type)                    
            utils.inputs_save(analog_comp_dict, 'Utilities/count_analog_' + name + K_type)
                    
            logger.info('The time for RELA %s, for embedding %s is %s', 
                        rela,
                        name,
                        str(datetime.datetime.now().replace(microsecond=0)-c))

    
def analog_pipe(L, K, k_most_similar, dict_labels_for_L, logger, K_type, parallel = False, embedding_type = 'both'):
    a = datetime.datetime.now().replace(microsecond=0)
    
    # Storing expression of relations in sets K and L
    sets_relations = defaultdict(list)
    for k in umls_tables_processing.USEFUL_RELA:
        sets_relations[k].append(('L_umls', np.shape(L[k])))
        sets_relations[k].append(('K'+K_type, np.shape(K[k])))
    print('Numbers of pairs for relationships stored')
    
    # Loading w2v files
    PATH_EMBEDDINGS = './Embeddings'
    embeddings = []
    
    # CUI or Word Embeddings discrimination
    if (embedding_type == 'cuis') or (embedding_type == 'both'):
        logger.info('CUI embeddings\n'), 
        cuis = ('/cuis/', [f.name for f in os.scandir(PATH_EMBEDDINGS+'/cuis') if (f.is_file())&(f.name != 'README.md')])
        embeddings.append(cuis)
        
    elif (embedding_type == 'words') or (embedding_type == 'both'):
        logger.info('Word embeddings\n'), 
        labels = ('/words/', [f.name for f in os.scandir(PATH_EMBEDDINGS+'/words') if (f.is_file())&(f.name != 'README.md')])
        embeddings.append(labels)
    
    # Universal dictionary instantiation
    analog_comp_dict = {}
    
    for type_emb in embeddings:
        b = datetime.datetime.now().replace(microsecond=0)
        if parallel:
            args = []
            for emb in type_emb[1]:
            # Instantiation of args for multiprocessing run
                args.append((PATH_EMBEDDINGS+type_emb[0]+emb,
                             emb.endswith('.bin'),
                             os.path.splitext(emb)[0],
                             type_emb[0], 
                             L, K,
                             k_most_similar,
                             K_type,
                             logger,
                             analog_comp_dict,
                             sets_relations,
                             dict_labels_for_L)) 
                logger.info('Preprocessing finished and multiprocessing running started\n')
                # Multiprocessing logic for evaluating at the same time K for only copd related concepts
                # and for seed related concepts
            with Pool(processes = len(args)) as pool:
                pool.starmap(analog_loop, args) 
        else:
            for emb in type_emb[1]:
                analog_loop(PATH_EMBEDDINGS+type_emb[0]+emb, 
                            emb.endswith('.bin'),
                            os.path.splitext(emb)[0],
                            type_emb[0], 
                            L, K,
                            k_most_similar,
                            K_type,
                            logger,
                            analog_comp_dict,
                            sets_relations,
                            dict_labels_for_L)
            
        logger.info('The time for analogical computation of %s is %s', 
                    type_emb,
                    str(datetime.datetime.now().replace(microsecond=0)-b))      
    logger.info('Execution time of analog_pipe: ' + str(datetime.datetime.now().replace(microsecond=0) - a) + '\n')

    
if __name__ == '__main__':
    # Parsing values for fast and intuitive launch of the script: 
    # paralleling, embedding_type, copd_K_switch are inserted by command line.
    parser = argparse.ArgumentParser(description='Launching analogy computation')
    parser.add_argument('--p', 
                        dest='paralleling',
                        type=bool,
                        default = False,
                        required=False,
                        help='The parallelization switch')
    
    parser.add_argument('--t',
                        dest='embedding_type',
                        type=str,
                        default = 'both',
                        required=False,
                        help='The type of analyzed embedding: it could be "both", "cuis", or "words"')
    
    parser.add_argument('--K_copd',
                        dest='copd_K_switch',
                        type=bool,
                        default = False,
                        required=False,
                        help='The choosen K_umls set: True for copd K')
    
    parser.add_argument('--L',
                        dest='L_type',
                        type=bool,
                        default = False,
                        required=False,
                        help='The choosen L_umls set: False for L, True for L=K')
    
    args = parser.parse_args()
    print(args)
    
    # Check on quality of inserted data
    assert args.embedding_type in ['both', 'cuis', 'words'], "Insert a string like 'both', 'cuis', or 'words'"

    # Constant and logger instantiation    
    K_MOST_SIMILAR = 10

    logger = utils.setup_custom_logger('myapp')
    logger.info('Start\n')
    
    # K_umls only for copd related concepts or for all.
    if args.copd_K_switch:
        K_umls = umls_tables_processing.count_pairs(umls_tables_processing.USEFUL_RELA, 
                                                         cuis_list = [umls_tables_processing.COPD])
        label_K = '_umls_copd' 
    else:
        # CUIs 
        concepts = umls_tables_processing.concepts_related_to_concept(concept = umls_tables_processing.COPD,
                                                                  two_way = True,
                                                                  polishing_rels = False,
                                                                  switch_key = 'con',
                                                                  extract_labels = False)
        logger.info('Seeds built\n')
        K_umls = umls_tables_processing.count_pairs(umls_tables_processing.USEFUL_RELA, cuis_list = concepts)
        label_K = '_umls'
    
    # Set L building - limited relations for lightening the compute
    if args.L_type:
        L_umls = K_umls 
    else:
        L_umls = umls_tables_processing.count_pairs(umls_tables_processing.USEFUL_RELA)
    
    logger.info('Sets created\n')
    
    # Building the dictionary for labels case
    # Collecting all the CUIs involved in set L
    jh = []
    for v in L_umls.values():
        jh.append(list(set(list(zip(*v))[0])))
        jh.append(list(set(list(zip(*v))[1])))
        tmp = set([j for i in jh for j in i ])
    dict_strings = umls_tables_processing.cui_strings()    
    dict_labels_for_L, _ = umls_tables_processing.extracting_strings(list(tmp), dict_strings = dict_strings)
    logger.info('Dictionary of labels from set L built\n')
        
    # Start analogy pipeline
    analog_pipe(L_umls, K_umls, 
                K_MOST_SIMILAR, 
                dict_labels_for_L, 
                logger, 
                label_K, 
                parallel = args.paralleling,
                embedding_type = args.embedding_type)
    
    