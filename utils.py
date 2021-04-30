import csv, datetime

def mmi_to_cui(stop_value = 400, mmi_file = 'paper_seed.txt.out'):
    #
    #
    #-------------------------------------------------------------------------------------------------
    # The method extracts CUIs from MMI list file coming from MetaMap, after metamapping text.
    #
    # It takes as input a stop_value which represents the seed's length and the path where the 
    # mapped text is located.
    # The method returns the first stop_value° CUIs, considering only the MMI lines: for briefness,
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
