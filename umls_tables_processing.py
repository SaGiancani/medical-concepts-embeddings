import csv, datetime, utils
from collections import defaultdict


def count_relationships(mrrel_path = 'UMLS_data/MRREL.RRF', rel_type = 'RELA'):
    #
    #
    #----------------------------------------------------------------------------------------------------
    # Accessory method for computing the number of particular relationships inside UMLS.
    # The method get just the MRREL's path.
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
                rel = array[8]
            if rel_type == 'REL':
                # Considering the REL elements
                rel = array[3]
            # Adding it into a set: only if it is not already present    
            set_tmp.add(rel)
        print(len(set_tmp))
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return set_tmp



def concepts_related_to_concept(mrrel_path = 'UMLS_data/MRREL.RRF', 
                                concept = 'C0024117', 
                                two_way = True, 
                                polishing_rels = True):
    #
    #
    #----------------------------------------------------------------------------------------------------
    # The method substitues the related_cuis_concept, adding even the relationship to the dict. 
    # 
    # Create a dictionary of related CUIs to a concept -as keys-, given as input, and the RELA as lists 
    # of strings.
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
    tmpd[concept].append('')
    # Open the RFF table
    with open(mrrel_path, newline='') as file:
    # Loop among the lines of the table
        for line in file.readlines():
            # Split the rows on the separator |
            array = line.split('|')
            # Take the valuable data: 1st's element cui , 2nd's element cui
            cuione_item = array[0]
            cuitwo_item = array[4]
            # Check on CUI: if the string's CUI is not inside the dictionary 
            # yet, come in
            # Only one way implementation
            if (cuione_item == concept):
                # Appending the CUIs inside the temporary list
                tmpd[cuitwo_item].append(array[7])
            # Both way implementation    
            elif (cuitwo_item == concept) & two_way:
                # Appending the CUIs inside the temporary list
                tmpd[cuione_item].append(array[7])
    # Discard duplicate relationships and the empty ones
    if polishing_rels:
        return utils.polish_relations(tmpd)

    #list(set(double_rel[i]))

    print(datetime.datetime.now().replace(microsecond=0)-a)
    return tmpd


    
def cui_strings(mrconso_path = 'UMLS_data/MRCONSO.RRF'):
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



def extracting_strings(cuis_list , dict_strings):
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



def extracting_stys(cuis, mrsty_path= 'UMLS_data/MRSTY.RRF'):
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

    
    
def related_cuis_concept(concept = 'C0024117', mrrel_path = 'UMLS_data/MRREL.RRF'):
    #
    #
    #----------------------------------------------------------------------------------------------------
    #--------------------------------------------- DEPRECATED--------------------------------------------
    #
    # Create a list of related CUIs to a concept, given as input.
    # The method gets as input the MRREL.RRF table's path as well.
    # MRREL.RRF row example: C0000005|A26634265|SCUI|RB|C0036775|A0115649|SCUI||R31979041||MSH|MSH|||N||
    #
    # It represents the first step in building the seed_rel: 
    # (MRREL) --FIRST_HOP--> (ordered list of CUIs)
    #
    # The method extracts the second CUIs of the relation CUI1|REL|CUI2: the relation is doubled, so in 
    # the table you can find the both ways of the relationship.
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    #
    #
    # Timer for keeping track of the activity time
    a = datetime.datetime.now().replace(microsecond=0)
    list_tmp = []
    # Appending the concept which is looked for
    list_tmp.append(concept)
    # Open the RFF table as a csv file
    with open(mrrel_path, newline='') as csvfile:
    # Loop among the lines of the table
        for line in csvfile.readlines():
            # Split the rows on the separator |
            array = line.split('|')
            # Take the valuable data: 1st's element cui , 2nd's element cui
            cuione_item = array[0]
            cuitwo_item = array[4]
            # Check on CUI: if the string's CUI is not inside the dictionary 
            # yet, come in
            if ((cuione_item == concept)&(cuitwo_item not in list_tmp)):
                # Appending the CUIs inside the temporary list
                list_tmp.append(cuitwo_item)
            # This check is deprecated because the relationships are biunivoque and duplicated
            #if ((cuitwo_item == concept)&(cuione_item not in list_tmp)):
            # Appending CUI
            #    print('COPD is the second related word')
            #    list_tmp.append(cuione_item)
        print(len(list_tmp))
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return list_tmp 
    
    
    
