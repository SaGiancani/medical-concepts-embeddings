from collections import Counter
import datetime, json, requests
import pandas as pd


# UMLS Semantic Types from the MetaMap manual. version July 14, 2006

DICT_STY = {'acab': 'Acquired Abnormality',
            'acty': 'Activity',
            'aggp': 'Age Group',
            'alga': 'Alga', 
            'amas': 'Amino Acid Sequence',
            'aapp': 'Amino Acid, Peptide, or Protein',
            'amph': 'Amphibian',
            'anab': 'Anatomical Abnormality',
            'anst': 'Anatomical Structure',
            'anim': 'Animal',
            'antb': 'Antibiotic',
            'arch': 'Archaeon',
            'bact': 'Bacterium',
            'bhvr': 'Behavior',
            'biof': 'Biologic Function',
            'bacs': 'Biologically Active Substance',
            'bmod': 'Biomedical Occupation or Discipline',
            'bodm': 'Biomedical or Dental Material',
            'bird': 'Bird',
            'blor': 'Body Location or Region',
            'bpoc': 'Body Part, Organ, or Organ Component',
            'bsoj': 'Body Space or Junction',
            'bdsu': 'Body Substance',
            'bdsy': 'Body System',
            'carb': 'Carbohydrate',
            'crbs': 'Carbohydrate Sequence',
            'cell': 'Cell',
            'celc': 'Cell Component',
            'celf': 'Cell Function',
            'comd': 'Cell or Molecular Dysfunction',
            'chem': 'Chemical',
            'chvf': 'Chemical Viewed Functionally',
            'chvs': 'Chemical Viewed Structurally',
            'clas': 'Classification',
            'clna': 'Clinical Attribute',
            'clnd': 'Clinical Drug',
            'cnce': 'Conceptual Entity',
            'cgab': 'Congenital Abnormality',
            'dora': 'Daily or Recreational Activity',
            'diap': 'Diagnostic Procedure',
            'dsyn': 'Disease or Syndrome',
            'drdd': 'Drug Delivery Device',
            'edac': 'Educational Activity',
            'eico': 'Eicosanoid',
            'elii': 'Element, Ion, or Isotope',
            'emst': 'Embryonic Structure',
            'enty': 'Entity',
            'eehu': 'Environmental Effect of Humans',
            'enzy': 'Enzyme',
            'evnt': 'Event',
            'emod': 'Experimental Model of Disease',
            'famg': 'Family Group',
            'fndg': 'Finding',
            'fish': 'Fish',
            'food': 'Food',
            'ffas': 'Fully Formed Anatomical Structure',
            'ftcn': 'Functional Concept',
            'fngs': 'Fungus',
            'gngp': 'Gene or Gene Product (pseudo ST for gene terminology)',
            'gngm': 'Gene or Genome',
            'genf': 'Genetic Function',
            'geoa': 'Geographic Area',
            'gora': 'Governmental or Regulatory Activity',
            'grup': 'Group',
            'grpa': 'Group Attribute',
            'hops': 'Hazardous or Poisonous Substance',
            'hlca': 'Health Care Activity',
            'hcro': 'Health Care Related Organization',
            'horm': 'Hormone',
            'humn': 'Human',
            'hcpp': 'Human-caused Phenomenon or Process',
            'idcn': 'Idea or Concept',
            'imft': 'Immunologic Factor',
            'irda': 'Indicator, Reagent, or Diagnostic Aid',
            'inbe': 'Individual Behavior',
            'inpo': 'Injury or Poisoning',
            'inch': 'Inorganic Chemical',
            'inpr': 'Intellectual Product',
            'invt': 'Invertebrate',
            'lbpr': 'Laboratory Procedure',
            'lbtr': 'Laboratory or Test Result',
            'lang': 'Language',
            'lipd': 'Lipid',
            'mcha': 'Machine Activity',
            'mamm': 'Mammal',
            'mnob': 'Manufactured Object',
            'medd': 'Medical Device',
            'menp': 'Mental Process',
            'mobd': 'Mental or Behavioral Dysfunction',
            'mbrt': 'Molecular Biology Research Technique',
            'moft': 'Molecular Function',
            'mosq': 'Molecular Sequence',
            'npop': 'Natural Phenomenon or Process',
            'neop': 'Neoplastic Process',
            'nsba': 'Neuroreactive Substance or Biogenic Amine',
            'nnon': 'Nucleic Acid, Nucleoside, or Nucleotide',
            'nusq': 'Nucleotide Sequence',
            'ocdi': 'Occupation or Discipline',
            'ocac': 'Occupational Activity',
            'ortf': 'Organ or Tissue Function',
            'orch': 'Organic Chemical',
            'orgm': 'Organism',
            'orga': 'Organism Attribute',
            'orgf': 'Organism Function',
            'orgt': 'Organization',
            'opco': 'Organophosphorus Compound',
            'patf': 'Pathologic Function',
            'podg': 'Patient or Disabled Group',
            'phsu': 'Pharmacologic Substance',
            'phpr': 'Phenomenon or Process',
            'phob': 'Physical Object',
            'phsf': 'Physiologic Function',
            'plnt': 'Plant',
            'popg': 'Population Group',
            'pros': 'Professional Society',
            'prog': 'Professional or Occupational Group',
            'qlco': 'Qualitative Concept',
            'qnco': 'Quantitative Concept',
            'rcpt': 'Receptor',
            'rnlw': 'Regulation or Law',
            'rept': 'Reptile',
            'resa': 'Research Activity',
            'resd': 'Research Device',
            'rich': 'Rickettsia or Chlamydia',
            'shro': 'Self-help or Relief Organization',
            'sosy': 'Sign or Symptom',
            'socb': 'Social Behavior',
            'spco': 'Spatial Concept',
            'strd': 'Steroid',
            'sbst': 'Substance',
            'tmco': 'Temporal Concept',
            'topp': 'Therapeutic or Preventive Procedure',
            'tisu': 'Tissue',
            'vtbt': 'Vertebrate',
            'virs': 'Virus',
            'vita': 'Vitamin',
           }



def check_sty_mmi(proto_seed, dict_sty=DICT_STY):
    #
    #
    #-------------------------------------------------------------------------------------------------------------
    # The method takes as input the list of tuples coming from mmi_to_cui method: it has as first element a CUI
    # and the correspondent list of semantic types, even if the CUI has only one semantic type.
    #
    # It performs a count of the semantic types of the concepts contained into the proto_seed (proto because there
    # are no strings associated to the CUIs) and returns a dictionary with the semantic type MetaMap codes as keys
    # and a tuple of the number of semantic types occurrences and the extended string of the type code.
    #
    # 
    #-------------------------------------------------------------------------------------------------------------
    #
    #
    tmp = []
    tmp_d = {}
    for i in proto_seed:
        for j in i[1]:
            tmp.append(j)
            
    decr = sorted(dict(Counter(tmp)).items(), key=lambda x: x[1], reverse= True)
    for a in decr:
        if a[0] in dict_sty.keys():
            tmp_d[a[0]] = ((a[1], dict_sty[a[0]])) 
            
    return tmp_d



def convert_sty_stymmi(dict_cui_sty, dict_sty=DICT_STY):
    #
    #
    #---------------------------------------------------------------------------------------------------------------
    # Convert the string extended semantic type to the abbreviation using the DICT_STY constant coming from MMI
    # manual.
    #---------------------------------------------------------------------------------------------------------------
    #
    #
    tmp = dict_cui_sty.items()
    for i,j in dict_sty.items():
        for z, k in tmp:
            for h, n in enumerate(k):
                if n == j:
                    k[h] = i
    return tmp
    


def mmi_to_cui(stop_value = 400, mmi_file = 'Utilities/paper_seed.txt.out', sty = False):
    #
    #
    #-------------------------------------------------------------------------------------------------
    # The method extracts CUIs from MMI list file coming from MetaMap, after metamapping text.
    #
    # It takes as input a stop_value which represents the seed's length, the path where the 
    # mapped text is located and a boolean value for considering the semantic type as well: it's false
    # by default.
    # The method returns a list of first stop_value CUIs, considering only the MMI lines, if the sty 
    # is false: for briefness,the mapped concept with a CUI. If the sty is true, a list of tuples is 
    # returned, with a CUI as first element and the correspondent semantic_type as second element.
    #
    # It represents the first step in building the seed_paper: (free text) --MMI--> (ordered list of CUIs)
    #
    # For best comprehension of metamapping, looking for MMI format on MetaMap guide.
    #-------------------------------------------------------------------------------------------------
    #
    #
    a = datetime.datetime.now().replace(microsecond=0)
    list_cuis = []
    count = 0
    # Open the txt MMI file coming out MetaMap as a csv file
    with open(mmi_file, newline='') as txt:
    # Loop among the lines of the table
        if sty:
            for line in txt.readlines():
                # Split the rows on the separator |
                array = line.split('|')
                # Taking the block of the line representing the format
                mmi_sign = array[1]
                # If the format of the concept is MMI, the CUI is considered and appended on a list_cuis
                if (mmi_sign == 'MMI'):
                    cui_item = array[4]
                    sem_type = array[5]
                    newstr = sem_type.replace("]", "")
                    newstr = newstr.replace("[", "")
                    list_cuis.append((cui_item, newstr.split(',')))
                    count+=1
                if (count==(stop_value)):
                    print(datetime.datetime.now().replace(microsecond=0)-a)
                    return list_cuis
        else:
            for line in txt.readlines():
                # Split the rows on the separator |
                array = line.split('|')
                # Taking the block of the line representing the format
                mmi_sign = array[1]
                # If the format of the concept is MMI, the CUI is considered and appended on a list_cuis
                if (mmi_sign == 'MMI'):
                    cui_item = array[4]
                    list_cuis.append(cui_item)
                    count+=1
                if (count==(stop_value)):
                    print(datetime.datetime.now().replace(microsecond=0)-a)
                    return list_cuis
            

            
def mmi_lite_freetext(free_text_file = 'Utilities/paper_seed.txt', sty = False):
    #
    #
    #---------------------------------------------------------------------------------------------------------
    # The method maps text thanks to MetaMapLite and its ReST service. 
    #
    # Send request to ReST service and return response when received.
    # Example of received data:
    #
    #  url = 'https://ii-public1.nlm.nih.gov/metamaplite/rest/annotate'
    #  acceptfmt = 'text/plain'
    #  params = [('inputtext', 'Apnea\n'), ('docformat', 'freetext'),
    #               ('resultformat', 'json'), ('sourceString', 'all'),
    #               ('semanticTypeString', 'all')]
    #  resp = handle_request(url, acceptfmt, params)
    #  resp.text
    #  '[{"matchedtext":"Apnea",
    #   "evlist":[{"score":0,"matchedtext":"Apnea","start":0,"length":5,"id":"ev0",
    #               "conceptinfo":{"conceptstring":"Apnea",
    #                              "sources":["MTH","NCI_CTCAE_5","NCI","NCI_CTCAE_3"],
    #                              "cui":"C1963065","preferredname":"Apnea, CTCAE",
    #                              "semantictypes":["fndg"]}},
    #             {"score":0,"matchedtext":"Apnea","start":0,"length":5,"id":"ev0",
    #              "conceptinfo":{"conceptstring":"Apnea",
    #                             "sources":["LNC","MTH","HPO","NANDA-I","ICPC2P","CHV",
    #                                        "SNMI","SNM","NCI_FDA","LCH_NW","AOD","ICD9CM",
    #                                        "MDR","SNOMEDCT_US","CCPSS","WHO","NCI_NICHD",
    #                                        "CSP","RCDSA","MSH","ICD10CM","CST","OMIM",
    #                                        "NCI_CTCAE","ICPC2ICD10ENG","COSTAR","MEDCIN",
    #                                        "LCH","RCD","RCDAE","NCI","PSY","NDFRT","RCDSY",
    #                                        "DXP","ICNP"],
    #                             "cui":"C0003578","preferredname":"Apnea",
    #                             "semantictypes":["sosy"]}}],
    #    "docid":"00000000.tx","start":0,"length":5,"id":"en0","fieldid":"text"}]'
    #---------------------------------------------------------------------------------------------------------
    #
    #
    a = datetime.datetime.now().replace(microsecond=0)
    url = 'https://ii-public1.nlm.nih.gov/metamaplite/rest/annotate'
    acceptfmt = 'text/plain'
    cui_lite = []
    oov_lite = []
    with open(free_text_file, newline='') as txt:
    # Loop among the lines of the table
        if sty:
            for line in txt.readlines():
                
                params = [('inputtext', line+'\n'), ('docformat', 'freetext'),
                          ('resultformat', 'json'), ('sourceString', 'all'),
                          ('semanticTypeString', 'all')]
                headers = {'Accept' : acceptfmt}
                resp = requests.post(url, params, headers=headers) 
                
                if (json.loads(resp.text)):
                    for j in json.loads(resp.text)[0]['evlist']:
                        #print(i['conceptinfo']['cui'])
                        cui_lite.append((j['conceptinfo']['cui'], j['conceptinfo']['semantictypes']))
                else:
                    oov_lite.append(line)
                
            print(datetime.datetime.now().replace(microsecond=0)-a )
            return cui_lite, oov_lite
        
        else:
            for line in txt.readlines():
                
                params = [('inputtext', line+'\n'), ('docformat', 'freetext'),
                          ('resultformat', 'json'), ('sourceString', 'all'),
                          ('semanticTypeString', 'all')]
                headers = {'Accept' : acceptfmt}
                resp = requests.post(url, params, headers=headers) 
                
                if (json.loads(resp.text)):
                    for j in json.loads(resp.text)[0]['evlist']:
                        #print(i['conceptinfo']['cui'])
                        cui_lite.append(j['conceptinfo']['cui'])
                else:
                    oov_lite.append(line)
                
            print(datetime.datetime.now().replace(microsecond=0)-a )
            return cui_lite, oov_lite

        