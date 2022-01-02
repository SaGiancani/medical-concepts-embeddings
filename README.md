# medical-concepts-embeddings

## Description

The repo provides some algorithms and strategies for the evaluation of medical embeddings, relying on UMLS metathesaurus.

It is organized in the following way: there are some subfolders __Embeddings__, __UMLS_data__, __Utilities__, __logs__.

- __Embeddings__ is the models' folder: into it are contained the analyzed embeddings. It is divided in three further subfolder, one for embedding type. A subfolder for CUI embeddings, one for classical textual embedding and another one for staticized BioBERT model. They require several gigabytes of storage.
- __UMLS_data__ contains the .RRF files coming from UMLS metathesaurus. It has to be filled with MRCONSO, MRREL, MRSTY, for the proper working of scripts. They require several gigabytes of storage and an UMLS license.
- __Utilities__ contains some useful variables, byproducts of processed UMLS data -for speeding up the computation and size-reducting the data-, or data ready for visualization.
- __logs__: here are stored the files for debugging and keeping tracks of backgrounds running processes.

The folders are followed by executable -and not- python scripts, one for each developed pipeline.
- __analogy_pipeline.py__ is the python executable for analogy measures pipeline. It is based on the analogical reasoning, formalized in _Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781 (2013)_. The script allows to handle the processing only on CUI models, plain text models, both, and implements several strategies for picking labels for each UMLS' concept.
- __contextual.py__ contains processing and accessory methods for __static_conversion.py__: it allows the staticization of contextual embeddings, in other words, BERT-family models. The script focus on BioBERT, according the biomedical soul of the project.
- __data_visualization.py__ contains accessory methods for the visualization of data: in particular there are several methods for tabling and plotting the outcome variables obtained using analogy, relatedness and relation_direction pipelines.
- __measures.py__ contains all the implemented measures and respectively formalized equations. The core formulas for analogy, relatedness (occurrence), and relation direction are present in it.
- __mmi_txt_to_cui.py__ provides utility methods for the processing of MetaMap and MetaMapLite data.
- __relatedness_pipeline.py__ similarly to __analogy_pipeline.py__, it implements the relatedness (aka occurrence) pipeline. The script allows to handle the processing only on CUI models, only on plain text models, or both. It allows even to choose the seed for occurrence testing (seed rel, seed paper or the union of both).
- __relation_direction_pipeline.py__ is the python executable for relation direction measures pipeline. It is based on the intuition formalized in _Bommasani, Rishi, Kelly Davis, and Claire Cardie. "Interpreting pretrained contextualized representations via reductions to static embeddings." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020_. In this case the main focus of the analysis is not genre, race or sexual orientation bias but the UMLS particular relationships are investigated.
- __static_conversion.py__ is based on _Bommasani, Rishi, Kelly Davis, and Claire Cardie. "Interpreting pretrained contextualized representations via reductions to static embeddings." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020_ and their staticization strategy for BERT-family models.
- __umls_tables_processing.py__ contains all the algorithms for the processing of UMLS .RRF used files. It implements methods for querying and manipulating UMLS biomedical information.
- __utils.py__ is a collection of utility methods, with methods employed all over the folder.

## Installation

Download this repo writing this line on your terminal:

`git clone https://github.com/SaGiancani/medical-concepts-embeddings`

Then create the environment med_concept_emb_env, activate and install requirements running the `setup.sh` bash script, with the following lines:

`cd medical-concepts-embeddings/`

`chmod u+x setup.sh`

`source setup.sh`

The scripts are coded on [gensim library](https://radimrehurek.com/gensim/) -already installed with the previous lines-, that requires language models in proper formats: .txt, .bin . 

Download and move language models for processing in the proper subfolders, descripted before.
