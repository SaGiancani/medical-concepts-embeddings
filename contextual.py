import datetime, torch
import numpy as np

def tokenize_words(tokenizer, sentences, start, stop, attention_mask = False):
    #
    #
    #------------------------------------------------------------------------------
    # The method gets as input the tokenizer object built with transformers library
    # and a list of sentences/composed words to convert into an embedding for 
    # BERT processing.
    #
    # It returns a dictionary of elements ready to use as query for BERT models.
    # 
    # The dictionary has two keys:
    # one, with values a list for the input_ids with the id of each token which
    # form the choosen sentence/composed word.
    # another one, with values for the attention_mask: it avoid to consider empty 
    # id elements.
    #------------------------------------------------------------------------------
    #
    #
    # Initializing timer
    a = datetime.datetime.now().replace(microsecond=0)
    
    # Initialize a checkpoint
    print('Start: '+str(start))
    print('Stop: '+str(stop))
    if len(sentences[start:stop])>10:
        checkpoint = round(len(sentences[start:stop])/10)
        print(checkpoint)
    
    # initialize dictionary to store tokenized sentences    
    inputs = {'input_ids': []}
 
    if attention_mask:
        inputs['attention_mask'] = []
    
    for i, sentence in enumerate(sentences[start:stop]):
        # Encode each sentence and append to dictionary:
        # The truncation is referred to dimension normalization of batches (useless for us)
        # padding = False allows to have tensor of different dimensions depending on the words  
        new_tokens = tokenizer.encode_plus(sentence,truncation=False,
                                           padding=False,return_tensors='pt')
        inputs['input_ids'].append(new_tokens['input_ids'][0])
        
        if attention_mask:
            inputs['attention_mask'].append(new_tokens['attention_mask'][0])
        
        # Print checkpoint
        if ((i+1)%checkpoint == 0):
            print(str(round(((i+1)/len(sentences[start:stop]))*100))+'% \n')
            
    # reformat list of tensors into single tensor
    # I cant stack the list into a tensor because by construction they have no the same dimension (padding = False)
    #inputs['input_ids'] = torch.stack(inputs['input_ids'])
    #inputs['attention_mask'] = torch.stack(inputs['attention_mask'])
    print(datetime.datetime.now().replace(microsecond=0)-a)
    return inputs

def context2static(model, inputs, vocabs, start, stop, name = 'bio_bert', n_layer= 5, log = None):
    #
    #
    #------------------------------------------------------------------------------------------------------------
    # The method gets as input a contextual embedding model (BERT, GPT, etc.), a list of input tokens for 
    # each sentence, a vocabulary of words (as a list) coming from static embedding, a start index value and a 
    # stop index value, the string name used for saving the new context2static embedding and the layer of model
    # choosen as vector.
    #
    # The method save automatically the extracted layer/vector, pooling it (with a mean) every token by dimension.
    # It writes the result on a txt file, ready for the processin on gensim library (static embeddings library)
    #
    # A straightforward saving strategy is used for avoiding sudden crashes of operations, with relative lost of 
    # data: this is requested by the long times of computation. This is a conservative approach, since it is 
    # slower but more efficient in terms of stored data.
    #------------------------------------------------------------------------------------------------------------
    #
    #
    a = datetime.datetime.now().replace(microsecond=0)
    vocabs = vocabs[start:stop]
    path_save = './Embeddings/words/'+ str(start)+'--'+str(stop)+name +'-l'+str(n_layer)+ '.txt'
    with open(path_save, 'w') as file:
        # The check is for the first row of static embeddings: they use to take the dimension of written matrix
        if start==0:
            dims = model(inputs[0].unsqueeze(0))[-1][n_layer].size()[2]
            file.write(''.join('%s %s\n' % (len(inputs), dims)))

        # Initialize a checkpoint for printing
        if len(inputs)>100:
            checkpoint = np.round(len(inputs)/100)
            print(checkpoint)
            if log is not None:
                log.info('Every ' + str(checkpoint) + ' elements, an info message is printed in the log')                
        else:
            checkpoint = 1
        # Using all the layers for output, the model returns a modeling_outputs object 
        # of 3 elements: we are interested to the last element, a tuple which contains all the layers outputs.
        for i, inp in enumerate(inputs):
            layer = model(inp.unsqueeze(0))[-1][n_layer]
            #tmp_arr.append(layer.detach().squeeze().numpy().mean(axis=0))
            newstr = str(layer.detach().squeeze().numpy().mean(axis=0).tolist())[1:-1]
            
            # The following approach is too computationally expensive
            #tmp = str(newstr)
            #tmp = ''
            #for j in newstr:
            #    tmp = tmp + str(round(j,6)) + ' '   
            #print(np.shape(layer.detach().squeeze().numpy()))
            
            newstr = newstr.replace(",", "")
            file.write(''.join('%s %s\n' % (vocabs[i], newstr))) 
            # Print checkpoint
            if ((i+1)%checkpoint == 0):
                print('The layer extraction is at ' + 
                      str(((i+1)/len(inputs))*100) +
                      '%: the '+ str(i+1) +'th element'+ '\n')
                if log is not None:
                    log.info('The layer extraction is at ' +
                             str(((i+1)/len(inputs))*100) +
                             '%: the '+ str(i+1) +'th element')
                    log.info(str(datetime.datetime.now().replace(microsecond=0)-a) + 
                             ' total time for ' + str(((i+1)/len(inputs))*100) + '%\n')
        print(datetime.datetime.now().replace(microsecond=0)-a)

