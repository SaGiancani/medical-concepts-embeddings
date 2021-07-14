import argparse, contextual, datetime, utils

from itertools import product
from multiprocessing import Pool
from transformers import AutoModel, AutoTokenizer


def build_static_matrix(start_index, stop_index, name, layer):
    # Logging instation for next time printing
    logger = utils.setup_custom_logger('myapp')
        
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
                                    name=name, n_layer = layer, log=logger)
    logger.info('The context2static converting of matrix \"'+ str(start_index)+'--'+str(stop_index)+name +'\" time process is: ' +str(datetime.datetime.now().replace(microsecond=0)-a) + '\n')
    

if __name__ == '__main__':
    
    # Parsing values for fast and intuitive launch of the script: 
    # start_index, stop_index, name of the saved file and number of layer are inserted by command line.
    parser = argparse.ArgumentParser(description='Launching staticizing processing')
    parser.add_argument('--start', 
                        dest='start_index',
                        type=int, 
                        required=True,
                        help='The start index value')
    parser.add_argument('--stop',
                        dest='stop_index',
                        type=int,
                        required=True,
                        help='The stop index value')
    parser.add_argument('--n',
                        dest='name',
                        type=str,
                        default = 'bio_bert',
                        required=False,
                        help='The choosen layer from where extracting the feature vector')
    parser.add_argument('--l',
                        dest='layer',
                        type=int,
                        required=True,
                        help='The choosen layer from where extracting the feature vector')
    parser.add_argument('--p',
                        dest='processes',
                        type=int,
                        default = 1,
                        required=False,
                        help='The choosen number of parallel processes')
    args = parser.parse_args()
    
    # Check on quality of inserted data
    assert args.start_index>=0, "The start index has to be bigger than 0"
    assert ((args.stop_index-args.start_index)%args.processes)== 0, "Consider an interval of rows multiple of the number of chosen processes"
    assert args.layer>=0, "The layer has to be bigger than 0"
    assert args.stop_index>args.start_index, "The stop index has to be bigger than start index"
    assert args.processes>0, "Number of processes has to be more than 0" 
    
    
    # Launch of the main method
    if args.processes > 1:
        interval = int((args.stop_index - args.start_index)/args.processes)
        tmp = [(i*interval + args.start_index, (i*interval + args.start_index)+interval, args.name, args.layer) for i in range(args.processes)]
        print(tmp)
        with Pool(processes=args.processes) as pool:
            pool.starmap(build_static_matrix, tmp) 
        
    elif args.processes == 1:
        build_static_matrix(args.start_index, args.stop_index, args.name, args.layer)
