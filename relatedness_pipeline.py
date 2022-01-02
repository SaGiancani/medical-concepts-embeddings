import argparse, datetime, measures, mmi_txt_to_cui, os, umls_tables_processing, utils
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import datapath


PATH_EMBEDDINGS = './Embeddings'
SAVING_PATH = 'Utilities/RelatednessData/'
NAME_SAVED_FILE = 'relatedness_data_'


def cardinality_embeddings():
    '''
    -------------------------------------------------------------------------------------------------------------
    The method gets no input and returns a dictionary with information about the cardinality of embeddings.
    
    The outcome corresponds to the variable cardinality_vembs stored in Utilities.
    -------------------------------------------------------------------------------------------------------------    
    '''
    a = datetime.datetime.now().replace(microsecond=0)
    cuis = ('/cuis/', [f.name for f in os.scandir(PATH_EMBEDDINGS+'/cuis') if (f.is_file())&(f.name != 'README.md')])
    words = ('/words/', [f.name for f in os.scandir(PATH_EMBEDDINGS+'/words') if (f.is_file())&(f.name != 'README.md')])
    embeddings = [cuis, words]
    cardinality_vemb = {}
    for type_emb in embeddings:
        for emb in type_emb[1]:
            model = KeyedVectors.load_word2vec_format(PATH_EMBEDDINGS+type_emb[0]+emb, binary=emb.endswith('.bin'))
            name = os.path.splitext(emb)[0]
            vemb = utils.extract_w2v_vocab(model)
            cardinality_vemb[name] = len(vemb)

    #utils.inputs_save(cardinality_vemb, 'Utilities/cardinality_vembs')
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return cardinality_vemb


def max_ks_loop(big_g, seeds, type_emb, model, name, logger, all_labels = False, aggregation = 'max'):
    '''
    -------------------------------------------------------------------------------------------------------------
    The method represents a further loop inside the relatedness loop, for the case k_most_similar = k_max 
    experimentation. 
    
    Following a similar logic to the one implemented in regular_ks_loop, it is splitted into two section, one
    for the cuis and another one for words. A system of debugging print is implemented via a logger. 
    
    The method gets as input a global variable, big_g, which is used for storing the values: at the ks choosen 
    with the parsing with the ks list variable, the k = IoV is added. The method is recalled thanks to a switch in
    parent methods.
    
    The method gets a list of seeds, the type of embedding, the embedding which has to be analyzed, the name of 
    the analyzed embedding, the logger and a switch for choosing all the labels or only the first ranked by UMLS
    inside the embedding's vocabulary at the same time.    
    -------------------------------------------------------------------------------------------------------------
    '''
    big_g[name]['max_k'] = {}    
    for seed in seeds:
        Vemb = utils.extract_w2v_vocab(model)
        if type_emb[0]=='/cuis/':
            k = len(list(set(Vemb).intersection(set(seed[1].keys()))))
            logger.info('\n k_value: %s\n', k)
            if k <=0:
                k = 1
            d = measures.occurred_concept(model, seed[1].keys(), k_most_similar=k)
            big_g[name]['max_k'][seed[0]] = [measures.pos_dcg(d, 
                                                        normalization = True, 
                                                        norm_fact = measures.max_dcg(k)),
                                       measures.neg_dcg(d, 
                                                        normalization = True,
                                                        norm_fact = measures.max_dcg(k)),
                                       measures.percentage_dcg(d, k=k),
                                       k,
                                       measures.oov(d),
                                       len(seed[1]), []]

        elif type_emb[0]=='/words/':
            processed_seed = umls_tables_processing.discarding_labels_oov(Vemb, seed[1], all_labels = all_labels)
            k = sum([1 for k,v in processed_seed.items() if len(v)>0])
            logger.info('\n k_value: %s\n', k)
            if k <= 0:
                k = 1
            d, _ = measures.occurred_labels(model, 
                                            processed_seed,
                                            k_most_similar=k,
                                            heuristic_aggregation = aggregation)

            big_g[name]['max_k'][seed[0]] = [measures.pos_dcg(d, 
                                                              normalization = True,
                                                              norm_fact = measures.max_dcg(k)),
                                             measures.neg_dcg(d,
                                                              normalization = True,
                                                              norm_fact = measures.max_dcg(k)),
                                             measures.percentage_dcg(d, k=k),
                                             k,
                                             measures.oov(d),
                                             len(seed[1]), []]
            
        logger.info('%s: pos_dcg: %.4f, neg_dcg: %.4f, perc_dcg: %.4f, iov/k-NN: %d, oov: %d, #seed: %d\n', 
                    seed[0],
                    big_g[name]['max_k'][seed[0]][0],
                    big_g[name]['max_k'][seed[0]][1],
                    big_g[name]['max_k'][seed[0]][2],
                    big_g[name]['max_k'][seed[0]][3],
                    big_g[name]['max_k'][seed[0]][4],
                    big_g[name]['max_k'][seed[0]][5])  
    return big_g


def regular_ks_loop(embeddings, ks, seeds, logger, max_k_switch, all_labels = False, aggregation = 'max'):
    '''
    -------------------------------------------------------------------------------------------------------------
    The method implements the logic for relatedness and occurrence experimentation. It is splitted in two blocks:
    one for cuis and another one for labels. Iteratively it loads an embedding model, among the ones inside the 
    folder indicated in PATH_EMBEDDINGS, and applies the occurrence and dcg measures.
    
    The experimentation is performed for the choosen ks got via parsing, in the variable ks, for the seeds in 
    seeds list. A logger is provided for debugging and the max_k_switch enable the max_ks_loop.
    The all_labels switch allows to take all the labels for a concept or only the best ranked by UMLS.
    
    The method returns the global variable big_g
    -------------------------------------------------------------------------------------------------------------    
    '''
    big_g = {}
    a = datetime.datetime.now().replace(microsecond=0)

    for type_emb in embeddings:
        for emb in type_emb[1]:
            model = KeyedVectors.load_word2vec_format(PATH_EMBEDDINGS+type_emb[0]+emb, binary=emb.endswith('.bin'))
            name = os.path.splitext(emb)[0]
            big_g[name] = {}
            logger.info('\n\n The name of embedding is: %s\n', name)
            for i, k in enumerate(ks):
                logger.info('\n k_value: %s\n', k)
                big_g[name][k] = {}
                for seed in seeds:
                    if type_emb[0]=='/cuis/':
                        d = measures.occurred_concept(model, seed[1].keys(), k_most_similar=k)
                        big_g[name][k][seed[0]] = [measures.pos_dcg(d, normalization = True, norm_fact = measures.max_dcg(k)),
                                                   measures.neg_dcg(d, normalization = True, norm_fact = measures.max_dcg(k)),
                                                   measures.percentage_dcg(d, k=k),
                                                   measures.iov(d),
                                                   measures.oov(d),
                                                   len(seed[1]), []]
                        
                    elif type_emb[0]=='/words/':
                        Vemb = utils.extract_w2v_vocab(model)
                        processed_seed = umls_tables_processing.discarding_labels_oov(Vemb, seed[1], all_labels = all_labels)
                        d, new_seed = measures.occurred_labels(model, 
                                                               processed_seed,
                                                               k_most_similar=k,
                                                               heuristic_aggregation = aggregation)
                        big_g[name][k][seed[0]] = [measures.pos_dcg(d, normalization = True, norm_fact = measures.max_dcg(k)),
                                                   measures.neg_dcg(d, normalization = True, norm_fact = measures.max_dcg(k)),
                                                   measures.percentage_dcg(d, k=k),
                                                   measures.iov(d),
                                                   measures.oov(d),
                                                   len(seed[1]),
                                                   new_seed]
                        
                    logger.info('%s: pos_dcg: %.4f, neg_dcg: %.4f, perc_dcg: %.4f, iov: %d, oov: %d, #seed: %d\n', 
                                seed[0],
                                big_g[name][k][seed[0]][0],
                                big_g[name][k][seed[0]][1],
                                big_g[name][k][seed[0]][2],
                                big_g[name][k][seed[0]][3],
                                big_g[name][k][seed[0]][4],
                                big_g[name][k][seed[0]][5])
                    
            if max_k_switch and (i == len(ks)-1):
                big_g = max_ks_loop(big_g, 
                                    seeds,
                                    type_emb,
                                    model,
                                    name,
                                    logger,
                                    all_labels = all_labels,
                                    aggregation = aggregation)
                    
    logger.info('Time for relatedness pipeline computation is: %s', str(datetime.datetime.now().replace(microsecond=0)-a))
    return big_g

def relatedness_pipeline(logger, all_labels, ks, embedding_type, seed_type, max_k, aggregation):
    '''
    -------------------------------------------------------------------------------------------------------------
    It is an accessory method which allows the preparation of utility variables. The picked choices performed
    with cli, stored on arg parse are processed inside this method.
    
    The method gets as inputs all the arg parse variables and returns the file of stored big_g variable.
    -------------------------------------------------------------------------------------------------------------    
    '''
    # Constants
    embeddings = []
    seeds = []
    
    # Construction of UMLS dictionary: for each CUI are picked the correspondent labels.
    dict_conso = umls_tables_processing.cui_strings()
    
    # Seed building: all the concepts at one hop from copd are picked
    copd_dict = umls_tables_processing.concepts_related_to_concept(two_way = True, extract_labels = False )
    copd_cuis = list(copd_dict.keys())
    
    # Seed building
    if (seed_type == 'rel') or (seed_type == 'all'):
        seed_rel, _ = umls_tables_processing.extracting_strings(copd_cuis, dict_strings = dict_conso)
        seeds.append(('seed_rel', seed_rel))
        
    if (seed_type == 'paper') or (seed_type == 'all'):
        paper_cuis = mmi_txt_to_cui.mmi_to_cui(sty = True)
        seed_paper, _ = umls_tables_processing.extracting_strings([i[0] for i in paper_cuis], dict_strings = dict_conso)
        seeds.append(('seed_paper', seed_paper))    
    
    if (seed_type == 'union') or (seed_type == 'all'):
        try:
            seed_union = {**seed_rel, **seed_paper}
            seeds.append(('seed_union', seed_union))
        except:
            seed_rel, _ = umls_tables_processing.extracting_strings(copd_cuis, dict_strings = dict_conso)
            paper_cuis = mmi_txt_to_cui.mmi_to_cui(sty = True)
            seed_paper, _ = umls_tables_processing.extracting_strings([i[0] for i in paper_cuis], dict_strings = dict_conso) 
            seed_union = {**seed_rel, **seed_paper}
            seeds.append(('seed_union', seed_union))
            
    logger.info('Seeds are built!')
    
    # CUI or Word Embeddings discrimination
    if (embedding_type == 'cuis') or (embedding_type == 'both'):
        logger.info('CUI embeddings\n'), 
        cuis = ('/cuis/', [f.name for f in os.scandir(PATH_EMBEDDINGS+'/cuis') if (f.is_file())&(f.name != 'README.md')])
        embeddings.append(cuis)
        
    if (embedding_type == 'words') or (embedding_type == 'both'):
        logger.info('Word embeddings\n'), 
        labels = ('/words/', [f.name for f in os.scandir(PATH_EMBEDDINGS+'/words') if (f.is_file())&(f.name != 'README.md')])
        embeddings.append(labels)

    
    big_g = regular_ks_loop(embeddings, ks, seeds, logger, max_k, all_labels = all_labels, aggregation = aggregation)
                        
    
    if all_labels:
        utils.inputs_save(big_g, SAVING_PATH + NAME_SAVED_FILE + '_' + aggregation+'_all_lab'+str(datetime.datetime.now()))
    else:
        utils.inputs_save(big_g, SAVING_PATH + NAME_SAVED_FILE +'_onlypreflab'+str(datetime.datetime.now()))
    
    logger.info('Data stored correctly!')
    

    
if __name__ == '__main__':
    # Parsing values for fast and intuitive launch of the script: 
    # paralleling, embedding_type, copd_K_switch are inserted by command line.
    parser = argparse.ArgumentParser(description='Launching relatedness computation')
    
    parser.add_argument('--lab', 
                        dest='all_labels',
                        type=bool,
                        default = False,
                        required=False,
                        help='Only preferred or all labels')
    
    parser.add_argument('--Ks', 
                        nargs='+', 
                        dest='ks',
                        default= [5, 10, 20, 30, 40],
                        type=int,
                        help='The Ks to evaluate')
    
    parser.add_argument('--t',
                        dest='embedding_type',
                        type=str,
                        default = 'both',
                        required=False,
                        help='The type of analyzed embedding: it could be "both", "cuis", or "words"')    
    
    parser.add_argument('--s',
                        dest='seed_type',
                        type=str,
                        default = 'all',
                        required=False,
                        help='The type of analyzed embedding: it could be "all", "union", "rel", or "paper"')
    
    parser.add_argument('--maxk', 
                        dest='max_k',
                        type=bool,
                        default = False,
                        required=False,
                        help='Max K pipeline')    
    
    parser.add_argument('--h',
                        dest='aggregation',
                        type=str,
                        default = 'max',
                        required=False,
                        help='The type of analyzed embedding: it could be "ext", "max", "med"')
        
    args = parser.parse_args()
    print(args)
    
    # Check on quality of inserted data
    # Embedding type
    assert args.embedding_type in ['both', 'cuis', 'words'], "Insert a string like 'both', 'cuis', or 'words'"    
    # Seed type
    assert args.seed_type in ['all', 'union', 'rel', 'paper'], "Insert a string like 'all', 'union', 'rel', or 'paper'" 
    # Strategy for label picking
    assert args.aggregation in ['max', 'ext', 'med'], "Pick one among 'max', 'ext', 'med'" 
    # Start time
    start_time = datetime.datetime.now().replace(microsecond=0)

    # Logger instantiation
    logger = utils.setup_custom_logger('myapp')
    logger.info('Start\n')

    relatedness_pipeline(logger, **vars(args))
    logger.info('The total computation time is: %s', 
                str(datetime.datetime.now().replace(microsecond=0)-start_time))

    

