import datetime, measures, os, umls_tables_processing, utils
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import datapath


# Some relationships are discarded: this is a subset of all_copd_relations, almost the half of them were an overkill
USEFUL_RELA = ['associated_finding_of',
               'associated_morphology_of',
               'associated_with_malfunction_of_gene_product',
               'clinical_course_of',
               'contraindicated_with_disease',
               'course_of',
               'disease_has_associated_anatomic_site',
               'disease_has_associated_gene',
               'finding_site_of',
               'gene_associated_with_disease',
               'gene_product_malfunction_associated_with_disease', 
               'has_associated_finding',
               'has_associated_morphology',
               'has_clinical_course',
               'has_contraindicated_drug', 
               'has_course',
               'has_finding_site',
               'has_manifestation',
               #     'inverse_isa',
               'is_associated_anatomic_site_of',
               #     'isa',
               'manifestation_of',
               'may_be_treated_by', 
               'may_treat',
               '']
K_MOST_SIMILAR = 10

    
def analog_pipe(L, K, k_most_similar, dict_labels_for_L, logger, K_type):
    a = datetime.datetime.now().replace(microsecond=0)
    
    # Storing expression of relations in sets K and L
    sets_relations = defaultdict(list)
    for k in USEFUL_RELA:
        sets_relations[k].append(('L_umls', np.shape(L[k])))
        sets_relations[k].append(('K'+K_type, np.shape(K[k])))
    print('Numbers of pairs for relationships stored')
    
    # Loading w2v files
    PATH_EMBEDDINGS = './Embeddings'
    cuis = ('/cuis/', [f.name for f in os.scandir(PATH_EMBEDDINGS+'/cuis') if (f.is_file())&(f.name != 'README.md')])
    labels = ('/words/', [f.name for f in os.scandir(PATH_EMBEDDINGS+'/words') if (f.is_file())&(f.name != 'README.md')])
    embeddings = [cuis, labels]
    
    # Timer for loop
    analog_comp_dict = {}
    
    for type_emb in embeddings:
        b = datetime.datetime.now().replace(microsecond=0)
        for emb in type_emb[1]:
            model = KeyedVectors.load_word2vec_format(PATH_EMBEDDINGS+type_emb[0]+emb, binary=emb.endswith('.bin'))
            name = os.path.splitext(emb)[0]
            analog_comp_dict[name] = {}
            logger.info('\n\n The name of embedding is: %s\n', name)
            for rela in USEFUL_RELA:
                logger.info('\n The RELA is: %s\n', rela)

                if type_emb[0]=='/cuis/':
                    c = datetime.datetime.now().replace(microsecond=0)
                    l0, k0 = measures.k_n_l_iov(L[rela], 
                                                K[rela],
                                                model, 
                                                logger = logger,
                                                emb_type = 'cui')
                    
                    sets_relations[rela].append((emb+'_l'+K_type, np.shape(l0)))
                    sets_relations[rela].append((emb+'_k'+K_type, np.shape(k0)))
                    
                    analog_comp_dict[name][rela] = measures.analogy_compute(l0, k0, 
                                                                            model,
                                                                            k_most_similar,
                                                                            emb_type = 'cui')
                    utils.inputs_save(analog_comp_dict, 'Utilities/' + name + K_type)
                    utils.inputs_save(sets_relations, 'Utilities/sets_relations' + K_type)
                    
                    logger.info('The time for RELA %s is %s', rela,
                    str(datetime.datetime.now().replace(microsecond=0)-c))
                    
                elif type_emb[0]=='/words/':
                    c = datetime.datetime.now().replace(microsecond=0)
                    Vemb =utils.extract_w2v_vocab(model)
                    dict_labels_inters_vemb = umls_tables_processing.discarding_labels_oov(Vemb, dict_labels_for_L)
                    l0, k0 = measures.k_n_l_iov(L[rela], 
                                                K[rela],
                                                model, 
                                                logger = logger,
                                                dict_labels_for_L = dict_labels_inters_vemb,
                                                emb_type = 'labels')
                    
                    sets_relations[rela].append((emb+'_l'+K_type, np.shape(l0)))
                    sets_relations[rela].append((emb+'_k'+K_type, np.shape(k0)))
                    
                    analog_comp_dict[name][rela] = measures.analogy_compute(l0, k0, 
                                                                            model,
                                                                            k_most_similar,
                                                                            dict_labels_for_L = dict_labels_inters_vemb, 
                                                                            emb_type = 'labels')                    
                    
                    utils.inputs_save(analog_comp_dict, 'Utilities/' + name + K_type)
                    utils.inputs_save(sets_relations, 'Utilities/sets_relations' + K_type)
                    
                    logger.info('The time for RELA %s is %s', rela,
                    str(datetime.datetime.now().replace(microsecond=0)-c))
            
        logger.info('The time for analogical computation of %s is %s', 
                    type_emb,
                    str(datetime.datetime.now().replace(microsecond=0)-b))      
    logger.info('Execution time of analog_pipe: ' + str(datetime.datetime.now().replace(microsecond=0) - a) + '\n')

    
if __name__ == '__main__':
    a = datetime.datetime.now().replace(microsecond=0)    
    # Constant and logger instantiation    
    PATH_EMBEDDINGS = './Embeddings'
    logger = utils.setup_custom_logger('myapp')
    logger.info('Start\n')
    
    # Seed complete by relationships
    seed_analog_both = umls_tables_processing.concepts_related_to_concept(concept = 'C0024117',
                                                                          two_way = True,
                                                                          polishing_rels = False,
                                                                          switch_key = 'rel',
                                                                          extract_labels = True)
    
    # CUIs 
    concepts = umls_tables_processing.concepts_related_to_concept(concept = 'C0024117',
                                                                  two_way = True,
                                                                  polishing_rels = False,
                                                                  switch_key = 'con',
                                                                  extract_labels = False)
    
    print('Seeds built')
    # Relationships
    all_copd_relations = list(seed_analog_both.keys())
    
    # Set building
    L_umls = umls_tables_processing.count_pairs(all_copd_relations)
    K_umls = umls_tables_processing.count_pairs(all_copd_relations, cuis_list = concepts)
    K_umls_copd = umls_tables_processing.count_pairs(all_copd_relations, cuis_list = ['C0024117'])
    print('Sets created')
    
    # Building the dictionary for labels case
    # Collecting all the CUIs involved in set L
    jh = []
    for v in L_umls.values():
        jh.append(list(set(list(zip(*v))[0])))
        jh.append(list(set(list(zip(*v))[1])))
        tmp = set([j for i in jh for j in i ])
    dict_strings = umls_tables_processing.cui_strings()    
    dict_labels_for_L, _ = umls_tables_processing.extracting_strings(list(tmp), dict_strings = dict_strings)
    print('Dictionary of labels from set L built')
    
    # Instantiation of args for multiprocessing run
    args = [(L_umls, K_umls, K_MOST_SIMILAR, dict_labels_for_L, logger, '_umls'), 
           (L_umls, K_umls_copd, K_MOST_SIMILAR, dict_labels_for_L, logger, '_umls_copd')]
    
    logger.info('Preprocessing finished and multiprocessing running started\n')
    # Multiprocessing logic for evaluating at the same time K for only copd related concepts and for seed related concepts
    with Pool(processes = 2) as pool:
        pool.starmap(analog_pipe, args) 
    
    