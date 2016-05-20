# BIST Parsers
## Graph & Transition based dependency parsers using BiLSTM feature extractors.

The techniques bedhind the parser are described in the paper [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](http://arxiv.org/abs/1603.04351). 

#### Required software

 * Python 2.7 interpreter
 * [PyCNN library](https://github.com/clab/cnn/tree/master/pycnn)

#### Train a parsing model

The software requires having a `training.conll` and `development.conll` files formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat).
For the faster graph-based parser change directory to `bmstparser` (1200 words/sec), and for the more accurate transition-based parser change directory to `barchybrid` (800 word/sec). The benchmark was performed on a Mac book pro with i7 processor. The graph-based parser acheives an accuracy of 93.8 UAS and the transition-based parser an accuracy of 94.4 UAS on the standard Penn Treebank dataset (Standford Dependencies). The model and param files achieving those scores are available for download ([Graph-based model](https://www.dropbox.com/sh/v9cbshnmb36km6v/AADgBS9hb9vy0o-UBZW9AbbKa/bestfirstorder.tar.gz?dl=0), [Transition-based model](https://www.dropbox.com/sh/v9cbshnmb36km6v/AACEPp3DLQeJnRA_QyPmll93a/bestarchybrid.tar.gz?dl=0)). The trained models include improvements beyond those described in the paper, to be published soon.

To train a parsing model with for either parsing architecture type the following at the command prompt:

    python src/parser.py --cnn-seed 123456789 --outdir [results directory] --train training.conll --dev development.conll --test test.conll --epochs 30 --lstmdims 125 --lstmlayers 2 [--extrn extrn.vectors] --bibi-lstm

We use the same external embedding used in [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](http://arxiv.org/abs/1505.08075) which can be downloaded from the authors [github repository](https://github.com/clab/lstm-parser/) and [directly here](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing).

If you are training a transition-based parser then for optimal results you should add the following to the command prompt `--k 3 --usehead --userl`. These switch will set the stack to 3 elements; use the BiLSTM of the head of trees on the stack as feature vectors; and add the BiLSTM of the right/leftmost children to the feature vectors.

Note 1: You can run it without pos embeddings by setting the pos embedding dimensions to zero (--pembedding 0).

Note 2: The reported test result is the one matching the highest development score.

Note 3: The parser calculates (after each iteration) the accuracies excluding punctuation symbols by running the `eval.pl` script from the CoNLL-X Shared Task and stores the results in directory specified by the `--outdir`.

Note 4: The external embeddings parameter is optional and better not used when train/predicting a graph-based model.

#### Parse data with your parsing model

The command for parsing a `test.conll` file formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat) with a previously trained model is:

    python src/parser.py --predict --outdir [results directory] --test test.conll [--extrn extrn.vectors] --model [trained model file] --params [param file generate during training]

The parser will store the resulting conll file in the out directory (`--outdir`).

Note 1: If you are using the arc-hybrid trained model we provided please use the `--extrn` flag and specify the location of the external embeddings file.

Note 2: If you are using the first-order trained model we provided please do not use the `--extrn` flag.

#### Citation

If you make use of this software for research purposes, we'll appreciate citing the following:

    @ARTICLE{2016arXiv160304351K,
      author = {{Kiperwasser}, E. and {Goldberg}, Y.},
      title = "{Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations}",
      journal = {ArXiv e-prints},
      archivePrefix = "arXiv",
      eprint = {1603.04351},
      primaryClass = "cs.CL",
      keywords = {Computer Science - Computation and Language},
      year = 2016,
      month = mar,
      adsurl = {http://adsabs.harvard.edu/abs/2016arXiv160304351K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact elikip@gmail.com


