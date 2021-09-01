### Utilities folder

This folder contains preprocessed data used as variable by scripts: this has the goal to fasten the computation.

It is used for storing and loading data.

Some of the used variables are related to UMLS, and UMLS products usage is subject to restrictions.

To use the scripts, download and install UMLS in your machine.



On umls_tables_processing the 'Utilities/dict_conso' variable is loaded: this is the equivalent of running cui_strings method.
This variable is loaded, when not specifically fed, in `extracting_strings` and in `concepts_related_to_concept`

