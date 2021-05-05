import datetime, json, requests
import pandas as pd


def mmi_to_cui(stop_value = 400, mmi_file = 'paper_seed.txt.out'):
    #
    #
    #-------------------------------------------------------------------------------------------------
    # The method extracts CUIs from MMI list file coming from MetaMap, after metamapping text.
    #
    # It takes as input a stop_value which represents the seed's length and the path where the 
    # mapped text is located.
    # The method returns a list of first stop_value CUIs, considering only the MMI lines: for briefness,
    # the mapped concept with a CUI. 
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
            

            
def mmi_lite_freetext(free_text_file = 'paper_seed.txt'):
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