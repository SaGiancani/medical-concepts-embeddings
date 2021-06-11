import csv
import pandas as pd


def csv_emb_to_txt(path_load='./Embeddings/cui2vec_pretrained.csv', path_save='./Embeddings/cui2vec_pretrained.txt'):
    #
    #
    #------------------------------------------------------------------------------------------------------------------
    # The method is thought to solve the problem raised by the .csv format used by http://cui2vec.dbmi.hms.harvard.edu/
    # with their cui2vec embedding. Gensim doesnt support csv format.
    # 
    # The method converts .csv embedding to a .txt one.
    # It takes as input the path of the .csv input embedding and the path of the output .txt embedding.
    #------------------------------------------------------------------------------------------------------------------
    #
    #
    csv_beam = pd.read_csv(path_load)

    with open(path_save, 'w') as file:
        file.write(''.join('%s %s\n' % (len(csv_beam.iloc[:,0]), (len(csv_beam.iloc[0])-1))))
        for i in range(len(list(csv_beam.iloc[:, 0]))):
            oldstr = str(list(csv_beam.iloc[i]))
            # The squared brackets coming from the list and the ', chars from the python strings are deleted
            newstr = oldstr.replace("]", "")
            newstr = newstr.replace(",", "")
            newstr = newstr.replace("[", "")
            newstr = newstr.replace("'", "")
            file.write(''.join('%s\n' % (newstr)))        
            
            
def extract_w2v_vocab(model):
    #
    #
    #---------------------------------------------------------------------------------------------------------
    #The method returns a list of the vocabulary of the w2v model, taking the last as input
    #---------------------------------------------------------------------------------------
    #
    #
    return list(model.vocab.keys())

        
def save_txt_dicts(my_dict, name_file):
    #
    #
    #---------------------------------------------------------------------------------------------------------
    # Creates a txt file filled with dictionaries. For readability, the key is highlighted by some '+' chars
    #
    # It takes as input 3 nested dictionaries to save in the txt file, and a string for naming the file.
    #---------------------------------------------------------------------------------------------------------
    #
    #
    with open(name_file, 'w') as fp:
        fp.write('\n'.join(' +++++%s+++++\n%s\n%s\n%s' % (x, y, k, z) for x, y, k, z in zip(my_dict.keys(),
                                                                                            my_dict.values().keys(),
                                                                                            my_dict.values().values().keys(), 
                                                                                            my_dict.values().values().values())))
