# Appendix

This section describes content and experiments that the paper did not have space to put down.

## Promising Applications
1. Model copyright detection: when models are deployed to the cloud to gain revenue by providing services, model lineage closeness can detect if the deployed model is a pirated model of a model already on the cloud;

2. Model retrieval: the open-source model zoo can leverage lineage closeness to manage model versions to achieve efficient target model retrieval in a large number of models;

3. Model authentication: models are considered intellectual property and must be registered like patents. Model lineage closeness can be used to register new models;

4. Model provenance: in forensic scenarios, model lineage closeness can provide the entire chain of model piracy modifications and determine the source of piracy.

## Descriptions of Different Modification Techniques
- **Fine-tuning** alters the source model's parameters by training for a few additional epochs on the part of the model parameters with the same training dataset. 

- **Pruning** is a common method for compressing models, which involves reducing the size of less-important parts of source model's parameters. 

- **Adversary training** aims to enhance the robustness of the source model intrinsically, thereby preventing adversarial attacks by trigger sets.

- **Quantization** is a commonly used technique for compressing the source model and reducing memory while maintaining its functionality, which facilitates the deployment of the source model on resource-constrained edge devices.

- **Transfer learning** aims to transfer the knowledge from the source model to a different but related task, thereby reducing the training overhead and yielding a well-fitting model since models trained for similar tasks usually share a common feature extraction process.

- **Knowledge distillation** distills knowledge from the source model to train a student model. There are two common approaches to knowledge distillation: feature-based and output-based. The feature-based distillation requires the student model to have the same structure as the source model in order to utilize the intermediate layers' outputs, while the output-based distillation sacrifices some of the performance to be free from the model structure.

## Visualization experiments supplement
We provide detailed decision boundary visualization results for different models: the source model, lineage models, and no lineage models in this section. The visualization method is using three different randomly sampled data points to visualize the decision boundary and decision area in the plane where these data points are located. And the black dots represent distinct data points, while regions of the same color correspond to the same decision area.

<img src="\fig\image-20241214151035888.png" alt="image-20241214151035888" style="zoom:50%;" />

As shown in Fig, we observe that when models have lineage, the decision boundaries are strongly similar in almost all the visualized decision areas. But when models have no lineage, there are significant differences between their decision boundaries. 

<img src="\fig\image-20241214151134842.png" alt="image-20241214151134842" style="zoom:50%;" />


Additionally, we present visualizations of the decision boundaries for lineage models that have undergone modifications using techniques from the first category, as illustrated in Fig. Our observations reveal that all modifications have an effect on the decision boundaries of the source model, but a majority of the decision boundaries remain similar. Fine-tuning and quantization, which do not alter the parameters of the source model's feature extraction, exhibit negligible impact on the decision boundaries. Similarly, pruning eliminates redundant parameters that have minimal influence on the model's feature extraction, resulting in an insignificant alteration to the decision boundary. While adversary training modifies the decision boundaries to enhance the source model's robustness against adversarial attacks, but the decision boundaries still retain partial similarity.

![image-20241214151205645](\fig\image-20241214151205645.png)


Furthermore, we visualized decision boundaries of models modified by the third category modification technique, as shown in Fig. We observed that in the feature-based distillation modification, decision boundaries undergo significant changes while retaining partial similarity. Conversely, in the output-based distillation modification, the decision boundaries predominantly differ. 

![image-20241214151253871](\fig\image-20241214151253871.png)

![image-20241214151304727](\fig\image-20241214151304727.png)


Upon investigating the decision boundaries of various model techniques, we identified two distinct types of decision boundaries. Among all the lineage models, we discovered a subset of decision boundaries that exhibit high sensitivity to any form of modification, regardless of its magnitude. This phenomenon is illustrated in Fig, where the decision boundaries within the white boxes differ across all lineage models. Conversely, another type of decision boundary is consistently observed across all types of models, including the source model, lineage models, and no lineage models. This characteristic is exemplified in Fig.\ref{fig:easy_to_similart}, where the decision boundaries within the white boxes remain identical across all models.

## Rethinking Other Model Lineage Works
After gaining a deeper understanding of the different types of lineage models as well as no lineage models, we rethink the theoretical soundness of related model lineage works.

Among these works, three are based on decision boundaries. IPGuard\cite{cao2021ipguard} generates adversarial samples near the decision boundary. However, as indicated by the visualization results presented earlier, the decision boundary is highly sensitive to model modifications, potentially rendering the effectiveness of adversarial samples uncertain. This issue is also observed in ModelDiff. Although UAP utilizes universal adversarial perturbations to enhance robustness, it still carries the risk of failure when confronted with modifications such as adversarial training and distillation, which visibly impact decision boundaries.

Additionally, two works focus on decision regions. CEM and MeTaFinger  propose generating inputs that yield the same output in the source model and lineage models, but differ in the no lineage models. Essentially, they aim to identify samples within decision regions where the source model and lineage models produce identical outputs, while the no lineage models produce divergent outputs. However, it is important to note that the definition of this decision region heavily relies on the set of models employed to generate the samples. Moreover, given that model modifications perturb the decision boundary, the decision region undergoes changes, potentially invalidating the generated samples.

The primary concern with these efforts lies in their reliance on the features surrounding the decision boundary, rather than the decision boundary itself. Consequently, their approaches lack sufficient robustness to accommodate changes in the decision boundary resulting from model modifications.


## Description of Benchmark
In this benchmark, we maintain the model structures used in the original benchmark, namely MobileNetv2 and ResNet18, as well as the dataset: Oxford Flowers 102, Stanford Dogs 120 and ImageNet. As presented in Table, the benchmark now encompasses a total of 136 models. Among these, there are 2 pretrained models that have been trained using the ImageNet dataset. Additionally, there are 108 lineage models derived from the pretrained models. These lineage models consist of 12 transferred models, 24 fine-tuned models, 24 pruned models, 12 adversary trained models, 24 quantized models, and 12 distilled models. Each lineage model is constructed based on one of the transferred models, employing different modification techniques as described in the "Configuration" column of Tab. Furthermore, there are 26 retrained models that have been trained from scratch.

![image-20241214151633840](\fig\image-20241214151633840.png)

## Quantization Analysis for Model Modification
In this section, we provide a fine-grained quantitative analysis of the effects of different modification techniques on the model lineage closeness.

### 1. Fine-tune
% 不同数量的epoch
We perform different epochs of fine-tuning for different model structures with different fine-tuning methods. The results are shown in Tab, from which we can see that fine-tuning the model with different epochs does not significantly affect the model lineage closeness. Since fine-tuning does not change the feature extraction layer, it does not have significant effects on the decision boundary, and thus for different epochs of fine-tuning, similar lineage closeness are obtained.

![image-20241214151755358](\fig\image-20241214151755358.png)


### 2. Prune
![image-20241214151830473](\fig\image-20241214151830473.png)

We measure the model lineage closeness at different pruning rates in two different models. To better observe the variation of the model lineage closeness, we also calculate the model accuracy, and the results are shown in Fig. The figure demonstrates that when the model accuracy decreases due to pruning, the model lineage closeness also decreases at the same time, which confirms the measuring correctness of our method, i.e., the greater the degree of modification to the model, the lower the model lineage closeness.

### 3. Adversary Training
% 不同数量的对抗样本
We modify the model by adversary training with different numbers of adversarial samples and measure the model lineage closeness. The results are shown in Fig, from which we can see that as the number of adversarial samples increases, the accuracy of the adversary-trained model is affected, thus making the model lineage closeness decrease along with it. This result shows the precise of our method.

![image-20241214151856939](\fig\image-20241214151856939.png)

### 4. Distillation
% 不同数量的epoch
We distill models with 1000 epochs and save models every 50 epochs. Then, we measure these model lineage closeness with the source model and record the corresponding model accuracy. The results are shown in Fig. Figures show that as the number of training epochs increases, the model accuracy increases, and the model lineage closeness gradually increases, which means that the distilled model and the source model are becoming more and more similar. This result corresponds to our perception of the distillation procession, which shows the correctness of our method.

![image-20241214151918873](\fig\image-20241214151918873.png)

### 5. Quantization
Quantization only changes the model parameters types to qint8/float16 to store and does not affect the model performance, thus not the model lineage. The lineage closeness of the quantized model and the source model is always 1.

### 6. Transfer Learning
We respectively transfer the source model by tuning 10\% layers, 50\% layers, and 100\% layers. Then we measure the lineage closeness among these models. We find that the greater the difference in the number of transferred layers, the lower the lineage closeness of the model, e.g., tuning 10\% layers models have higher lineage closeness(0.835 in MobileNetv2 and 0.787 in ResNet18) with tuning 50\% layers models than tuning 100\% layers models(0.514 in MobileNetv2 and 0.616 in ResNet18). Thus, our approach can well capture the differences in lineage closeness between models with different degrees of transfer learning.


## Ablation Study
We conduct ablation study to illustrate the soundness of our method in test set generation and lineage closeness measuring. All experiments are conducted on two dataset:  Oxford Flower 102(Flower102 for short) and Stanford Dogs 120(SDog120 for short).

![image-20241214161317223](\fig\image-20241214161317223.png)

Firstly, we assess the effectiveness of our method in removing invalid samples. As depicted in Fig, we observe that the removal of invalid samples leads to higher matching rates and smaller mean distances for models that possess lineage. Conversely, for models without lineage, the matching rate decreases, and the mean distance increases. This indicates that the disparity in closeness between models with and without lineage becomes more pronounced after the removal of invalid samples. The outcome demonstrates that our method achieves greater precision following the removal of invalid samples.

Secondly, we assess whether the measuring results are impacted by sampling discrepancies in data points. The results, as illustrated in Fig, indicate that there are no substantial differences in the matching rate and mean distance between lineage and no lineage models before and after the sampling of data points. This observation highlights the unbiased nature of our sampling method, as it does not significantly affect the measuring results.

![image-20241214161332098](\fig\image-20241214161332098.png)

![image-20241214161348310](\fig\image-20241214161348310.png)

Finally, we substantiate the precision of our method for measuring lineage closeness. The results, as depicted in Fig, indicate that after removing invalid samples, the discriminate nature of model lineage is enhanced. Specifically, models with lineage exhibit higher lineage closeness, whereas models without lineage demonstrate lower lineage closeness. Furthermore, we observe that the lineage closeness remains unaffected by the sampling of discrepancy data points. Collectively, these findings affirm the precision of our lineage closeness measuring method.

## Limitations
Firstly, it is important to note that our proposed method is specifically designed for supervised classification models. This limitation arises due to the requirement of calculating the adversarial distance from data points to the decision boundary. Consequently, non-classified models, which lack a decision boundary, are not compatible with our approach. Moving forward, our future research endeavors aim to develop a more comprehensive measure of lineage closeness that can be applied to a broader range of model types.

Secondly, when dealing with models that have different tasks, we address the issue by aligning the dimensions of the decision space through transfer learning, utilizing the dataset of the compared model. However, practical scenarios often present a challenge as the dataset of the compared model is frequently unavailable, rendering our current approach ineffective. To overcome this limitation, our future work aims to devise a lineage closeness measure that can be employed across various tasks, alleviating the reliance on specific datasets.

## Future Work
Looking ahead, our future research endeavors will focus on developing a more comprehensive measure of lineage closeness that can be applied to a broader range of model types. Currently, our proposed method is specifically designed for supervised classification models. Consequently, non-classification models, which lack a decision boundary, are not compatible with our approach.