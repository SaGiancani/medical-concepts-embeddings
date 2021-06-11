import argparse, contextual, datetime, logging, sys, utils

from transformers import AutoModel, AutoTokenizer


def setup_custom_logger(name):
    PATH_LOGS = './logs/log_'
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(PATH_LOGS+str(datetime.datetime.now().replace(microsecond=0))+'.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger
'''
>>> logger = setup_custom_logger('myapp')
>>> logger.info('This is a message!')
2015-02-04 15:07:12 INFO     This is a message!
>>> logger.error('Here is another')
2015-02-04 15:07:30 ERROR    Here is another'''


def build_static_matrix(start_index, stop_index, name, layer):
    # Logging instation for next time printing
    logger = setup_custom_logger('myapp')
        
    # Load the vocabulary of PMC-w2v embedding: for avoiding the loading of model via gensim, 
    # a list with all the words of vocabulary was previously computed and stored
    #vocabs = utils.extract_w2v_vocab(w2v)
    a = datetime.datetime.now().replace(microsecond=0)
    vocabs = utils.inputs_load('PMC_w2v_vocabs')
    logger.info('PMC-w2v vocabulary (previously stored) is loaded in ' 
                +str(datetime.datetime.now().replace(microsecond=0)-a) + '\n')    
    
    # Load the contextual BioBERT embedding
    a = datetime.datetime.now().replace(microsecond=0)
    tokenizer = AutoTokenizer.from_pretrained('Embeddings/contextual/biobert-base-cased-v1.1', 
                                              output_hidden_states=True, cache_dir=None)
    model = AutoModel.from_pretrained('Embeddings/contextual/biobert-base-cased-v1.1',
                                      output_hidden_states=True, cache_dir=None)
    logger.info('The loading time for BioBERT is: ' +str(datetime.datetime.now().replace(microsecond=0)-a) + '\n')
    
    # Load tokenized input
    a = datetime.datetime.now().replace(microsecond=0)
    inputs = contextual.tokenize_words(tokenizer, vocabs, start_index, stop_index)
    logger.info('The tokenization time is: ' +str(datetime.datetime.now().replace(microsecond=0)-a) + '\n')
        
    # Build the context2static matrix
    a = datetime.datetime.now().replace(microsecond=0)
    tmp = contextual.context2static(model, inputs['input_ids'],
                                    vocabs, start_index, stop_index,
                                    name=name, n_layer = layer)
    logger.info('The context2static converting of matrix \"'+ str(start)+'--'+str(stop)+name +'\" time process is: ' +str(datetime.datetime.now().replace(microsecond=0)-a) + '\n')
    

if __name__ == '__main__':
    
    # Parsing values for fast and intuitive launch of the script: 
    # start_index, stop_index, name of the saved file and number of layer are inserted by command line.
    parser = argparse.ArgumentParser(description='Launching staticizing processing')
    parser.add_argument('--start', dest='start_index', type=int, 
                        required=True, help='The start index value')
    parser.add_argument('--stop', dest='stop_index', type=int,
                        required=True, help='The stop index value')
    parser.add_argument('--n', dest='name', type=str, default = 'bio_bert',
                        required=False, help='The choosen layer from where extracting the feature vector')
    parser.add_argument('--l', dest='layer', type=int,
                        required=True, help='The choosen layer from where extracting the feature vector')
    args = parser.parse_args()
    
    # Check on quality of inserted data
    assert args.start_index>=0, "The start index has to be bigger than 0"
    assert args.layer>=0, "The layer has to be bigger than 0"
    assert args.stop_index>args.start_index, "The stop index has to be bigger than start index"
    
    # Launch of the main method
    build_static_matrix(args.start_index, args.stop_index, args.name, args.layer)