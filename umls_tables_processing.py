import csv, datetime, utils
from collections import defaultdict

MRREL = 'UMLS_data/MRREL.RRF'
MRCONSO = 'UMLS_data/MRCONSO.RRF'
MRSTY = 'UMLS_data/MRSTY.RRF'
DICT_CONSO = 'Utilities/dict_conso'

def count_relationships(mrrel_path = MRREL, rel_type = 'RELA'):
    #
    #
    #----------------------------------------------------------------------------------------------------
    # Accessory method for computing the number of particular relationships inside whole UMLS.
    # The method get just the MRREL's path and the kind of relation considered (RELA or REL).
    #
    # MRREL.RRF row example: C0000005|A26634265|SCUI|RB|C0036775|A0115649|SCUI||R31979041||MSH|MSH|||N||
    #
    #----------------------------------------------------------------------------------------------------
    #
    #
    a = datetime.datetime.now().replace(microsecond=0)
    set_tmp = set()
    # Open the RFF table
    with open(mrrel_path, newline='') as file:
    # Loop among the lines of the table
        for line in file.readlines():
            # Split the rows on the separator |
            array = line.split('|')
            if rel_type == 'RELA':
                # Considering the RELA elements
                rel = array[7]
            if rel_type == 'REL':
                # Considering the REL elements
                rel = array[3]
            # Adding it into a set: only if it is not already present    
            set_tmp.add(rel)
        print(len(set_tmp))
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return set_tmp



def count_pairs(relations, cuis_list = None, mrrel_path = MRREL):
    #
    #
    #----------------------------------------------------------------------------------------------------
    # The method takes a list of relations and the path of MRREL table by UMLS.
    # The cuis_list input is not mandatory: it allows a further check for picking CUIs-pairs on the 
    # presence inside a choosen seed.
    # 
    # Create a dictionary of relations -as keys-, and the pairs -as tuples- of CUIs linked by those
    # relations. If cuis_list is not None, one of the two CUIs of the pair has to be present into the seed
    #
    # The method allows the building of the two sets for analogy computation: K_umls (with CUIs in pair
    # related to a seed) and L_umls (all the pairs of CUIs in UMLS having a certain relation)
    #
    # The method gets as input the MRREL.RRF table's path as well.
    # MRREL.RRF row example: C0000005|A26634265|SCUI|RB|C0036775|A0115649|SCUI||R31979041||MSH|MSH|||N||
    #
    # Have to be noticed: the order of storing is CUI2 - REL - CUI1. This is a feature of MRREL description
    #----------------------------------------------------------------------------------------------------
    #
    #
    a = datetime.datetime.now().replace(microsecond=0)
    tmpd = defaultdict(list)
    # Open the RFF table
    with open(mrrel_path, newline='') as file:
    # Loop among the lines of the table
        if cuis_list:
            for line in file.readlines():
                # Split the rows on the separator |
                array = line.split('|')
                if (array[7] in relations) & ((array[0] in cuis_list)|(array[4] in cuis_list)):
                    # Considering the RELA elements
                    pair_tuple = ((array[4], array[0]))
                    # Adding it into a set: only if it is not already present    
                    tmpd[array[7]].append(pair_tuple)

        if not cuis_list:
            for line in file.readlines():
                # Split the rows on the separator |
                array = line.split('|')
                if (array[7] in relations):
                    # Considering the RELA elements
                    pair_tuple = ((array[4], array[0]))
                    # Adding it into a set: only if it is not already present    
                    tmpd[array[7]].append(pair_tuple)
                
        print(len(tmpd))
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return tmpd


def concepts_related_to_concept(mrrel_path = MRREL, 
                                concept = 'C0024117', 
                                two_way = True, 
                                polishing_rels = False,
                                switch_key= 'con',
                                extract_labels = False):
    #
    #
    #----------------------------------------------------------------------------------------------------
    # The method substitues the related_cuis_concept, adding even the relationship to the dict. 
    # 
    # Create a dictionary of related CUIs to a concept (or relationships) -as keys-, given as input, and 
    # the RELA as lists of strings (or concepts as a list of strings).
    #
    # The two modalities are choosen by the switch_key value: it may be 'con' for CUIs as keys and RELAs 
    # as values or 'rel' for RELAs as keys and CUIs as values
    #
    # extract_labels represents a further switch which allows to augment the returned dictionary with
    # labels -preferred and not-
    #
    # The method gets as input the MRREL.RRF table's path as well.
    # MRREL.RRF row example: C0000005|A26634265|SCUI|RB|C0036775|A0115649|SCUI||R31979041||MSH|MSH|||N||
    #
    # It represents the first step in building the seed_rel: 
    # (MRREL) --FIRST_HOP--> (ordered list of CUIs)
    #
    # The method extracts the second CUIs of the relation CUI1|REL|CUI2: the relation is doubled, so in 
    # the table you can find the both ways of the relationship.
    #----------------------------------------------------------------------------------------------------
    #
    #
    # Timer for keeping track of the activity time
    a = datetime.datetime.now().replace(microsecond=0)
    tmpd = defaultdict(list)
    # Appending the concept which is looked for
    # Open the RFF table
    with open(mrrel_path, newline='') as file:
    # Loop among the lines of the table
        for line in file.readlines():
            # Split the rows on the separator |
            array = line.split('|')
            # Take the valuable data: 1st's element cui , 2nd's element cui
            cuione_item = array[0]
            cuitwo_item = array[4]
            rela = array[7]
            
            if (switch_key == 'con') and not extract_labels:
                key_one = cuitwo_item
                key_two = cuione_item
                value_one = rela
                value_two = rela
            
            elif (switch_key == 'rel') | extract_labels:
                key_one = rela
                key_two = rela
                value_one = cuitwo_item
                value_two = cuione_item             

            # Check on CUI: if the string's CUI is not inside the dictionary 
            # yet, come in
            # Only one way implementation
            if (cuione_item == concept):
                # Appending the CUIs inside the temporary list
                tmpd[key_one].append(value_one)
            # Both way implementation    
            elif (cuitwo_item == concept) & two_way: 
                # Appending the CUIs inside the temporary list
                tmpd[key_two].append(value_two)
            
    # If True, extracting labels -preferred and not- inside the method        
    if extract_labels:
        time_extract = datetime.datetime.now().replace(microsecond=0)
        dict_conso = utils.inputs_load(DICT_CONSO)
        cuis = list(set([i for k,v in tmpd.items() for i in v]))
        print(len(cuis))
        h, _ = extracting_strings(cuis, dict_conso)
        l = {}
        for rel,cuis in tmpd.items():
            r = {i:h[i] for i in cuis}
            l[rel] = r
        switch_key = 'rel'
        tmpd = l
        print('Extracting time: '+str(datetime.datetime.now().replace(microsecond=0)-time_extract))

    # Discard duplicate relationships and the empty ones
    # This is tricky, because the '' relation has half of the CUIs
    if polishing_rels:
        print('Relation \'\' discarded ')
        return utils.polish_relations(tmpd, ty = switch_key)
            

    print(datetime.datetime.now().replace(microsecond=0)-a)
    return tmpd


    
def cui_strings(mrconso_path = MRCONSO):
    #
    #
    #-----------------------------------------------------------------------
    # The method gets as input the MRCONSO.RRF table from UMLS.
    # It returns a dictionary with CUIs as keys and a list of strings 
    # (correspondent to unique SUIs) as values
    #
    # The strings with same SUI are discarded: for doing this a list of SUIs
    # for each CUI is built. For each CUI a further check is performed on 
    # the SUI: only an unique SUI (string) is considered.
    #-----------------------------------------------------------------------
    #
    #
    # Timer for keeping track of the activity time
    a = datetime.datetime.now().replace(microsecond=0)
    dict_strings = {}
    # Open the RFF table as a csv file
    with open(mrconso_path, newline='') as csvfile:
    # Loop among the lines of the table
        for line in csvfile.readlines():
            # Split the rows on the separator |
            array = line.split('|')
            # Take the valuable data: cui, sui, string
            cui_item = array[0]
            sui_item = array[5]
            string_item = array[14]
            # Check on CUI: if the string's CUI is not inside the dictionary 
            # yet, come in
            if cui_item not in dict_strings.keys():
                # Temporary strings and SUIs lists for each CUI
                list_tmp = []
                sui_tmp = []
                # Appending the SUI and strings inside the temporary list
                sui_tmp.append(sui_item)
                list_tmp.append(string_item)
                # Inserting the string temporary list to the relative CUI
                dict_strings[cui_item] = list_tmp
            # If the string's CUI is already inside the dictionary
            # and string is not considered yet, come in
            elif (cui_item in dict_strings.keys()) & (sui_item not in sui_tmp):
                # Appending SUIs
                sui_tmp.append(sui_item)
                dict_strings[cui_item].append(string_item)
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return dict_strings 



def discarding_labels_oov(emb_vocab, seed):
    #
    #
    #-------------------------------------------------------------------------------------------------
    # emb_vocab is a list containing the vocabulary of the analyzed embedding
    # seed is the classic dictionary with CUIs as keys and labels (preferred or not) as values
    # The method returns a polished seed, with only the labels contained inside the embedding vocabulary
    #-------------------------------------------------------------------------------------------------
    #
    #
    t = datetime.datetime.now().replace(microsecond=0)
    vemb = set(emb_vocab)
    new_dict = {}
    for cui, labels in seed.items():
        new_dict[cui] = list(vemb.intersection(set(labels)))
    print(datetime.datetime.now().replace(microsecond=0)-t)    
    return new_dict



def extracting_strings(cuis_list , dict_strings = utils.inputs_load(DICT_CONSO)):
    #
    #
    #-------------------------------------------------------------------------------------------------
    # Taking a list of CUIs and the dictionary coming from MRCONSO.RRF table as inputs, it returns
    # a dictionary with preferred and not preferred labels for each CUIs of the list.
    #-------------------------------------------------------------------------------------------------
    #
    #
    a = datetime.datetime.now().replace(microsecond=0)
    #dict_tmp = {i:dict_strings[i] for i in cuis_list}
    dict_tmp = {}
    # The not found CUIs in MRCONSO table are stored in not_found list
    not_found = []
    for i in cuis_list:
        try:
            dict_tmp[i] = dict_strings[i]
        except:
            not_found.append(i)
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return dict_tmp, not_found



def extracting_stys(cuis, mrsty_path= MRSTY):
    #
    #
    #----------------------------------------------------------------------------------------------------------
    # The method, taken a list of CUIs, returns a dictionary with CUIs as keys and a list of semantic types 
    # (strings) as values.
    #
    # It takes as input a list of CUIs and the path of the MRSTY.RRF table, the table where you can find the 
    # CUI related to correspondents semantic types.
    #
    # ex. line: C0000132|T126|A1.4.1.1.3.3|Enzyme|AT17739337|256|
    #----------------------------------------------------------------------------------------------------------
    #
    #
    # Timer for keeping track of the activity time
    a = datetime.datetime.now().replace(microsecond=0)
    tmp = defaultdict(list)
    # Loop among the lines of the table
    for i in cuis:
        count = 0
        with open(mrsty_path, newline='') as file:    
            for line in file.readlines():
                safety_count = count
                # Split the rows on the separator |
                array = line.split('|')
                # Take the valuable data: 1st's element cui , the correspondent semantic type
                cui = array[0]
                sty = array[3]

                if i == cui:
                    tmp[cui].append(sty)
                    count +=1
            
                elif (count>0) & (count==safety_count):
                    break
        
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return tmp
