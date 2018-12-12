# zeta_makro
Command-line tool for using zeta in experimental settings

# Why is this usefull?

Zeta was originally developed to find distinct words for authors or later distinct groups of texts, but always in contrast to another well-defined group of texts. Earlier implementations were based on this idea.  However, there are questions that require comparing a distinct group of texts with a more general group in order to identify features that describe this group more context-free. Therefore, zeta_makro distinguishes between focus group and comparison group. 



## Parameters

-path_to_metadata : Filepath to a Metadatatable in csv-format (sep=tab)
-path_to_files : Filepath to the folder containing files (tokenized, one token per line)
-output_filepath : Path to store the output
-target : Metadata criteria to build the focus group
-segmentlen : Length (in token) of compared chunks
-max_segments_per_group: maximal number of segments for each group
-counter_size : (even|full) even: size of comparision group is limited to size of focus group | full: no limit is set
-selection_mode : (random|balanced) random: the comparision group is build on a random sample across categories | balanced: if comparision group is build from different classes, these classes will be represented as equal as possible
