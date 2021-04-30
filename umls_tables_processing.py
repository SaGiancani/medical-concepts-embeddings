import csv, datetime


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
    
    
def related_cuis_concept(concept = 'C0024117', mrrel_path = 'UMLS_data/MRREL.RRF'):
    #
    #
    #----------------------------------------------------------------------------------------------------
    # Create a list of related CUIs to a concept, given as input.
    # The method gets as input the MRREL.RRF table's path as well.
    # MRREL.RRF row example: C0000005|A26634265|SCUI|RB|C0036775|A0115649|SCUI||R31979041||MSH|MSH|||N||
    #
    # The method extract the second CUIs of the relation CUI1|REL|CUI2: the relation is doubled, so in 
    # the table you can find the both ways of the relationship.
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