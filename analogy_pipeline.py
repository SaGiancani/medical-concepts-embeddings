import argparse, datetime, measures, os, umls_tables_processing, utils
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import datapath

PATH_EMBEDDINGS = './Embeddings'
SAVING_PATH = 'Utilities/Analogical Data/'

def analog_loop(path, 
                binary_bool,
                name,
                type_emb,
                L, K,
                K_type, logger,
                analog_comp_dict,
                #sets_relations,
                metrics,
                dict_labels_for_L = None,
                all_labels = False):
    
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
#            sets_relations[rela].append((name+'_l'+K_type, np.shape(l0)))
#            sets_relations[rela].append((name+'_k'+K_type, np.shape(k0)))
            
            # Save the pickle variable
#            utils.inputs_save(sets_relations, 'Utilities/sets_relations' + name + K_type)
                                
            # Compute the analogy and store the results
            tmp = measures.analogy_compute(l0, k0, 
                                           model,
                                           metrics,
                                           logger = logger,
                                           emb_type = 'cui')
            dict_t[name][rela] = tmp
            
            # Compute the sum of analogy hits and store it.
            #if len(tmp)>0:
            #    analog_comp_dict[name][rela] = (sum(list(zip(*tmp))[2]), len(tmp))
            #else:
            #    analog_comp_dict[name][rela] = (0, len(tmp))                        

            utils.inputs_save(dict_t, SAVING_PATH + name + K_type+str(datetime.datetime.now()))                    
            #utils.inputs_save(analog_comp_dict, 'Utilities/count_analog_' + name + K_type)
            
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
            dict_labels_inters_vemb = umls_tables_processing.discarding_labels_oov(Vemb, 
                                                                                   dict_labels_for_L,
                                                                                   all_labels = all_labels)
            # Filtering L and K sets for present labels inside the embedding
            l0, k0 = measures.k_n_l_iov(L[rela], 
                                        K[rela],
                                        model, 
                                        logger = logger,
                                        dict_labels_for_L = dict_labels_inters_vemb,
                                        emb_type = 'labels')

            # Store number of filtered pairs
#            sets_relations[rela].append((name+'_l'+K_type, np.shape(l0)))
#            sets_relations[rela].append((name+'_k'+K_type, np.shape(k0)))

#            utils.inputs_save(sets_relations, 'Utilities/sets_relations' + name + K_type)
                    
            tmp = measures.analogy_compute(l0, k0, 
                                           model, 
                                           metrics,
                                           logger = logger,
                                           dict_labels_for_L = dict_labels_inters_vemb, 
                                           emb_type = 'labels')                    
                    
            dict_t[name][rela] = tmp
                    
            #if len(tmp)>0:
            #    analog_comp_dict[name][rela] = (sum(list(zip(*tmp))[2]), len(tmp))
            #else:
            #    analog_comp_dict[name][rela] = (0, len(tmp))
                        
            utils.inputs_save(dict_t, SAVING_PATH + name + K_type+str(datetime.datetime.now()))                    
            #utils.inputs_save(analog_comp_dict, 'Utilities/count_analog_' + name + K_type)
                    
            logger.info('The time for RELA %s, for embedding %s is %s', 
                        rela,
                        name,
                        str(datetime.datetime.now().replace(microsecond=0)-c))

    
def analog_pipe(L, K,
                dict_labels_for_L,
                logger, 
                K_type,
                metrics,
                parallel = False,
                embedding_type = 'both', 
                all_labels = False):
    
    a = datetime.datetime.now().replace(microsecond=0)
    
    # Storing expression of relations in sets K and L
#    sets_relations = defaultdict(list)
#    for k in umls_tables_processing.USEFUL_RELA:
#        sets_relations[k].append(('L_umls', np.shape(L[k])))
#        sets_relations[k].append(('K'+K_type, np.shape(K[k])))
#    print('Numbers of pairs for relationships stored')
    
    # Loading w2v files
    embeddings = []
    
    # CUI or Word Embeddings discrimination
    if (embedding_type == 'cuis') or (embedding_type == 'both'):
        logger.info('CUI embeddings\n'), 
        cuis = ('/cuis/', [f.name for f in os.scandir(PATH_EMBEDDINGS+'/cuis') if (f.is_file())&(f.name != 'README.md')])
        embeddings.append(cuis)
        
    if (embedding_type == 'words') or (embedding_type == 'both'):
        logger.info('Word embeddings\n'), 
        labels = ('/words/', [f.name for f in os.scandir(PATH_EMBEDDINGS+'/words') if (f.is_file())&(f.name != 'README.md')])
        embeddings.append(labels)
    
    # Universal dictionary instantiation
    analog_comp_dict = {}
    
    for type_emb in embeddings:
        b = datetime.datetime.now().replace(microsecond=0)
        if parallel:
            # Multiprocessing logic for evaluating at the same time K for only copd related concepts
            # and for seed related concepts
            # If processes are more than 4, the performance is low given the expensive memory cost
            if len(type_emb[1]) > 4:
                # Processes set at 2
                processes = 2
                # Elements of a chunk
                #n = int(np.ceil(len(embeddings[1][1])/processes))
                n = processes
                # A list of sublist with embedding names
                chunk_embs = [type_emb[1][i:i + n] for i in range(0, len(type_emb[1]), n)]
                print(chunk_embs)
                for chunk in chunk_embs:
                    inp = []
                    for title in chunk:
                        # Creation of a process for each embedding (max two embeddings)
                        inp.append((PATH_EMBEDDINGS+type_emb[0]+title,
                                    title.endswith('.bin'),
                                    os.path.splitext(title)[0],
                                    type_emb[0], 
                                    L, K,
                                    K_type,
                                    logger,
                                    analog_comp_dict,
                                    #sets_relations,
                                    metrics,
                                    dict_labels_for_L, 
                                    all_labels))
                
                    with Pool(processes = n) as pool:
                        pool.starmap(analog_loop, inp)                 
            else:
                args = []
                for emb in type_emb[1]:
                    # Instantiation of args for multiprocessing run
                    args.append((PATH_EMBEDDINGS+type_emb[0]+emb,
                                 emb.endswith('.bin'),
                                 os.path.splitext(emb)[0],
                                 type_emb[0], 
                                 L, K,
                                 K_type,
                                 logger,
                                 analog_comp_dict,
                                 #sets_relations,
                                 metrics,
                                 dict_labels_for_L, 
                                 all_labels)) 
                    
                logger.info('Preprocessing finished and multiprocessing running started\n')
                with Pool(processes = len(args)) as pool:
                    pool.starmap(analog_loop, args) 
        else:
            for emb in type_emb[1]:
                analog_loop(PATH_EMBEDDINGS+type_emb[0]+emb, 
                            emb.endswith('.bin'),
                            os.path.splitext(emb)[0],
                            type_emb[0], 
                            L, K,
                            K_type,
                            logger,
                            analog_comp_dict,
                            #sets_relations,
                            metrics,
                            dict_labels_for_L)
            
        logger.info('The time for analogical computation of %s is %s', 
                    type_emb,
                    str(datetime.datetime.now().replace(microsecond=0)-b))      
    logger.info('Execution time of analog_pipe: ' + str(datetime.datetime.now().replace(microsecond=0) - a) + '\n')

    
def cardinality_kl(embeddings, useful_rela, L_umls, K_umls, dict_labels_for_L = None):
    #
    #
    #----------------------------------------------------------------------------------------------------------------
    # The method returns a pickle dictionary variable reusable by other methods for plotting.
    #
    # It gets as input a list of strings, with the names of analyzed embeddings, a list of analyzed relationships
    # and two pairs sets, K and L. The variable dict_labels_for_L is not compulsary: it is used only by the filtering
    # k_n_l_iov method for w2v embeddings.
    #
    # The method return informations about the cardinality of original UMLS sets and the filtered IoV one, given an 
    # embedding.
    #----------------------------------------------------------------------------------------------------------------
    #
    #
    a = datetime.datetime.now().replace(microsecond=0)
    sets_relations_k = {}
    sets_relations_l = {}
    # Loop over the relations
    for type_emb in embeddings:
        for emb in type_emb[1]:
            model = KeyedVectors.load_word2vec_format(PATH_EMBEDDINGS+type_emb[0]+emb, binary=emb.endswith('.bin'))
            name = os.path.splitext(emb)[0]
            print('Embedding ' + str(name) + ' is analyzed')
            for rela in useful_rela:
                c = datetime.datetime.now().replace(microsecond=0)
                # Check type of embedding
                if type_emb[0]=='/cuis/':
                    dict_labels_inters_vemb = None
                    labs = 'cui'

                # Check type of embedding: for word embeddings the dictionary of labels per cui is required
                elif (type_emb[0]=='/words/') and (dict_labels_for_L is not None):
                    # Filter the dictionary of labels keeping only the labels-words present into the embedding
                    Vemb = utils.extract_w2v_vocab(model)
                    dict_labels_inters_vemb = umls_tables_processing.discarding_labels_oov(Vemb, dict_labels_for_L)
                    labs = 'labels'

                # Filtering L and K sets for present labels inside the embedding
                l0, k0 = measures.k_n_l_iov(L_umls[rela], 
                                            K_umls[rela],
                                            model, 
                                            dict_labels_for_L = dict_labels_inters_vemb,
                                            emb_type = labs)

                # Store number of filtered pairs
                sets_relations_l[rela] = {name: np.shape(l0)[0]}
                sets_relations_k[rela] = {name: np.shape(k0)[0]}
                print('Execution time for rela ' + rela + ' : ' + 
                      str(datetime.datetime.now().replace(microsecond=0) - c) + '\n')
                
        sets_relations_l[rela] = {'L': np.shape(L_umls[rela])[0]}
        sets_relations_k[rela] = {'K': np.shape(K_umls[rela])[0]}
        sets_relations_l[rela] = {'L wor': np.shape(list(set(L_umls[rela])))[0]}
        sets_relations_k[rela] = {'K wor': np.shape(list(set(K_umls[rela])))[0]}
    
    utils.inputs_save(sets_relations_l, SAVING_PATH + 'l_cardinality_per_rel'+str(datetime.datetime.now()))                
    utils.inputs_save(sets_relations_k, SAVING_PATH + 'k_cardinality_per_rel'+str(datetime.datetime.now()))                
    print('Execution time : ' + str(datetime.datetime.now().replace(microsecond=0) - a) + '\n')
    

def processing_analog_pipe_outcome(name_emb, 
                                   dict_information,
                                   cardinality_relations = utils.inputs_load(SAVING_PATH+'k_cardinality_per_rel')):
    #
    #
    #-----------------------------------------------------------------------------------------------------------
    # The method gets the variable obtained with analogy_pipeline script and performs the cos3add custom formula
    # we designed: (n°of occurred expected concepts[rela]/#Kiov[rela])*(#Kiov[rela]/#(K_umls/Kiov[rela]))
    # The information for cardinality of several sets are taken by cardinality_relations variable, previously 
    # computed. It is a dictionary where keys are relas and values lists of tuples, where each tuple represents 
    # an embedding.
    #
    # N.B. #kiov does not count the same-couple analysis. The corrective factor is +1 
    #-----------------------------------------------------------------------------------------------------------
    #
    #
    dict_out = {}
    for rela in list(dict_information[name_emb].keys()):
        count = sum(dict_information[name_emb][rela]['add'])
        occurrences = len(dict_information[list(dict_information.keys())[0]][rela]['add'])
        kiov_cardin = cardinality_relations[rela][name_emb]
        k_umls_cardin = cardinality_relations[rela]['K wor']
        dict_out[rela] = measures.analog_comput_formula(kiov_cardin, k_umls_cardin, count, occurrences, rela)
    return dict_out

    
if __name__ == '__main__':
    # Parsing values for fast and intuitive launch of the script: 
    # paralleling, embedding_type, copd_K_switch are inserted by command line.
    parser = argparse.ArgumentParser(description='Launching analogy computation')
    parser.add_argument('--p', 
                        dest='paralleling',
                        type=bool,
                        default = False,
                        required=False,
                        help='The multiprocessing switch')
    
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
    
    parser.add_argument('--K',
                        dest='k_most_similar',
                        type=int,
                        default = 10,
                        required=False,
                        help='The choosen k_most_similar value')
    
    parser.add_argument('--eps',
                        dest='eps',
                        type=float,
                        default = 0.0001,
                        required=False,
                        help='The choosen epsilon value')
    
    parser.add_argument('--m', 
                        nargs='+', 
                        dest='measure',
                        default=['add'],
                        type=str,
                        help='The requested measures')
    
    parser.add_argument('--lab', 
                        dest='all_labels',
                        type=bool,
                        default = False,
                        required=False,
                        help='Only preferred or all labels')
    
    args = parser.parse_args()
    print(args)
    
    # Check on quality of inserted data
    # Embedding type
    assert args.embedding_type in ['both', 'cuis', 'words'], "Insert a string like 'both', 'cuis', or 'words'"
    
    # Measures check
    assert ('all' in args.measure and len(args.measure)==1) or (len(set(args.measure).intersection(set(['add', 'mul', 'pair']))) == len(args.measure)), "Choose if take 'all' or only certain measures among 'add', 'mul', 'pair'"

    # Logger instantiation
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
        label_K = '_LsameasK'+label_K
    else:
        L_umls = umls_tables_processing.count_pairs(umls_tables_processing.USEFUL_RELA)
    
    logger.info('Sets created\n')
    
    # Building the dictionary for labels case
    # Collecting all the CUIs involved in set L
    if (args.embedding_type == 'words') | (args.embedding_type == 'both'):
        # This was an overkill: for discarding 15mb of concept couples two very similar dictionaries were uploaded
        #jh = []
        #for v in L_umls.values():
        #    jh.append(list(set(list(zip(*v))[0])))
        #    jh.append(list(set(list(zip(*v))[1])))
        #    tmp = set([j for i in jh for j in i ])
        # Construction of UMLS dictionary: for each CUI are picked the correspondent labels.
        # all_labels handle the choice among all the possible labels and only the preferred one.
        dict_labels_for_L = umls_tables_processing.cui_strings()
        #dict_labels_for_L, _ = umls_tables_processing.extracting_strings(list(tmp), dict_strings = dict_strings)
    
    else:
        dict_labels_for_L = None
    logger.info('Dictionary of labels from set L built\n')
    
    # Building the dictionary for the measure, in place of the switch-case logic
    meas_dict = {'add': [measures.cos3add, args.k_most_similar],
                 'mul': [measures.cos3mul, args.eps], 
                 'pair': [measures.pair_direction, args.eps]}
    
    if 'all' in args.measure:
        meas_dict = meas_dict
        
    else:
        m = np.sort(args.measure).tolist()
        meas_dict = {key: meas_dict[key] for key in m if key in meas_dict}
        
    # Start analogy pipeline
    analog_pipe(L_umls, K_umls, 
                dict_labels_for_L, 
                logger, 
                label_K, 
                meas_dict,
                parallel = args.paralleling,
                embedding_type = args.embedding_type, 
                all_labels = args.all_labels)
    
    