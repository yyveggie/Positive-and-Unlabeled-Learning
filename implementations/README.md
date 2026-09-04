# Collected PU-learning implementations

Reference implementations of positive-unlabeled methods, gathered over several years and
merged here from an earlier repository. They are kept for reference and are independent of
`methods/`, which holds the implementations the experiments in this repository actually run.

The code is reproduced as collected. Where a file named its own source, that source is given
below; where none was recorded, the entry says so rather than guessing. Every method also has
a pseudocode document under `algorithms/` describing it and citing the paper it comes from.

Datasets are not included. Paths in the code point at a local `data/` directory; the UCI,
MNIST, LIBSVM and ijcnn1 files they expect are available from their original distributors.

| Directory | Files | Paper | Upstream source |
|---|---:|---|---|
| `Bagging` | 4 | Mordelet, Fantine, Vert, Jean-Philippe. A bagging SVM to learn from positive and unlabeled examples. Pattern Recognition Letters, 2014. | https://github.com/phuijse/bagging_pu, https://github.com/roywright/pu_learning |
| `Biased SVM` | 4 | Liu, Bing, Dai, Yang, Li, Xiaoli, Lee, Wee Sun, Yu, Philip S.. Building text classifiers using positive and unlabeled examples. Proceedings of the 3rd IEEE International Conference on Data Mining (ICDM), 2003. |  |
| `CR-SVM` | 1 | Li, Xiao-Li, Liu, Bing, Ng, See-Kiong. Negative training data can be harmful to text classification. Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2010. |  |
| `DAN` | 4 | Source not recorded in the collected files. |  |
| `DDI-PULearn` | 1 | Drug-drug interaction prediction. Source not recorded in the collected files. |  |
| `DEDPUL` | 25 | Ivanov, Dmitry. DEDPUL: difference-of-estimated-densities-based positive-unlabeled learning. Proceedings of the 19th IEEE International Conference on Machine Learning and Applications (ICMLA), 2020. | https://github.com/dimonenka/DEDPUL |
| `EB-SVM` | 1 | Source not recorded in the collected files. |  |
| `MixPUL` | 13 | Wei, Xu, Cheng, Wang, Ma, Yin, Chen. MixPUL: consistency-based augmentation for positive and unlabeled learning. arXiv:2004.09388, 2020. |  |
| `Modified PU-learning` | 1 | Source not recorded in the collected files. |  |
| `NB` | 1 | Naive Bayes baseline used by several of the two-step methods here. |  |
| `NB-SVM` | 1 | Source not recorded in the collected files. |  |
| `Original PU-learning` | 1 | Basile, Teresa M. A., Di Mauro, Nicola, Esposito, Floriana, Ferilli, Stefano, Vergari, Antonio. Density estimators for positive-unlabeled learning. New Frontiers in Mining Complex Patterns (NFMCP), 2018. |  |
| `PE-PUC` | 1 | Yu, Hang, Li, Xiao-Li. PE-PUC: a graph based PU-learning approach for text classification. Proceedings of the 5th International Conference on Machine Learning and Data Mining in Pattern Recognition (MLDM), 2007. |  |
| `PU-LEA NB` | 1 | Source not recorded in the collected files. |  |
| `PUSB` | 5 | Kato, Masahiro, Teshima, Takeshi, Honda, Junya. Learning from positive and unlabeled data with a selection bias. Proceedings of the 7th International Conference on Learning Representations (ICLR), 2019. |  |
| `PUbN` | 2 | Positive-unlabeled learning with a biased negative set. Source not recorded. |  |
| `Probability Estimation` | 1 | Elkan-Noto style posterior correction, adapted from the linked repository. | https://github.com/trokas/pu_learning |
| `Pulce` | 3 | Ienco, Dino, Pensa, Ruggero G.. Positive and unlabeled learning in categorical data. Neurocomputing, 2016. |  |
| `Pumil` | 8 | Multiple-instance PU learning. Source not recorded in the collected files. |  |
| `Pumml` | 8 | PU learning for materials discovery. | https://github.com/ncfrey/pumml |
| `Pywsl` | 7 | Copy of the pywsl weakly-supervised learning library, not a single method. | https://github.com/t-sakai-kure/pywsl |
| `RESVM` | 14 | Claesen, Marc, De Smet, Frank, Suykens, Johan A. K., De Moor, Bart. A robust ensemble approach to learn from positive and unlabeled data using SVM base models. Neurocomputing, 2015. | https://github.com/claesenm/resvm |
| `Roc-Clu-SVM` | 1 | Rocchio with clustering; distinct from Roc-SVM. |  |
| `Roc-SVM` | 1 | Li, Xiaoli, Liu, Bing. Learning to classify texts using positive and unlabeled data. Proceedings of the 18th International Joint Conference on Artificial Intelligence (IJCAI), 2003. |  |
| `S-EM` | 5 | Liu, Bing, Lee, Wee Sun, Yu, Philip S., Li, Xiaoli. Partially supervised classification of text documents. Proceedings of the 19th International Conference on Machine Learning (ICML), 2002. |  |
| `SAR-PU` | 31 | Bekker, Jessa, Robberechts, Pieter, Davis, Jesse. Beyond the selected completely at random assumption for learning from positive and unlabeled data. Proceedings of the European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), 2019. | https://github.com/ML-KULeuven/SAR-PU |
| `TIcE` | 1 | Bekker, Jessa, Davis, Jesse. Estimating the class prior in positive and unlabeled data through decision tree induction. Proceedings of the 32nd AAAI Conference on Artificial Intelligence, 2018. | https://dtai.cs.kuleuven.be/software/tice |
| `WMMD` | 7 | Convex formulation for learning from positive and unlabeled data. Source not recorded. |  |
| `kNN-RN` | 1 | Zhang, Bangzuo, Zuo, Wanli. Reliable negative extracting based on kNN for learning from positive and unlabeled examples. Journal of Computers, 2009. |  |
| `nnPU` | 10 | Kiryo, Ryuichi, Niu, Gang, du Plessis, Marthinus Christoffel, Sugiyama, Masashi. Positive-unlabeled learning with non-negative risk estimator. Advances in Neural Information Processing Systems (NeurIPS), 2017. |  |
| `nnPUSB` | 11 | Kato, Masahiro, Teshima, Takeshi, Honda, Junya. Learning from positive and unlabeled data with a selection bias. Proceedings of the 7th International Conference on Learning Representations (ICLR), 2019. |  |
| `pu_compress` | 4 | Source not recorded in the collected files. |  |

## Provenance and licensing

Several directories are copies of, or are derived from, third-party repositories. Their
original licences apply to that code, and the links above point at the upstream projects.
Any file that carried a licence header keeps it. Where a source could not be established
from the files themselves, the table says so; those entries should be treated as unattributed
until the origin is confirmed.
