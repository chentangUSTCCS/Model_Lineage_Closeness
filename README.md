# Model_Lineage_Closeness
This is the implementation of model_lineage_closeness_analysis_in_AAAI_2025.

It will take a few weeks to organize the source code and the benchmark we generated to assess the degree of modification between models, which will then be open-sourced and uploaded to this repository.

In addition, the content in the paper's appendix will also be organized and uploaded to this repository.



## Promising Application of Model Lineage Closeness
(1) Model copyright detection: when models are deployed to the cloud to gain revenue by providing services, model lineage closeness can detect if the deployed model is a pirated model of a model already on the cloud;

(2) Model retrieval: the open-source model zoo can leverage lineage closeness to manage model versions to achieve efficient target model retrieval in a large number of models;

(3) Model authentication: models are considered intellectual property and must be registered like patents. Model lineage closeness can be used to register new models;

(4) Model provenance: in forensic scenarios, model lineage closeness can provide the entire chain of model piracy modifications and determine the source of piracy.


## Future work
Looking ahead, our future research endeavors will focus on developing a more comprehensive measure of lineage closeness that can be applied to a broader range of model types. Currently, our proposed method is specifically designed for supervised classification models. Consequently, non-classification models, which lack a decision boundary, are not compatible with our approach.
