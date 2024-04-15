# A Self-feedback Knowledge Elicitation Approach: Scientific Language Modeling for Chemical Reaction Predictions (SLM4CRP)

The task of chemical reaction predictions (CRPs) plays a pivotal role in advancing drug discovery and material science. However, its effectiveness is constrained by the vast and uncertain chemical reaction space and challenges in capturing reaction selectivity, particularly due to existing methods' limitations in exploiting the data's inherent knowledge. To address these challenges, we introduce a data-curated self-feedback knowledge elicitation approach. This method starts from iterative optimization of molecular representations and facilitates the extraction of knowledge on chemical reaction types (RTs). Then, we employ adaptive prompt learning to infuse the prior knowledge into the large language model (LLM). As a result, we achieve significant enhancements: a 14.2% increase in retrosynthesis prediction accuracy, a 74.2% rise in reagent prediction accuracy, and an expansion in the model's capability for handling multi-task chemical reactions. This research offers a novel paradigm for knowledge elicitation in scientific research and showcases the untapped potential of LLMs in CRPs.

![Overview of tasks and approaches.](figures/figure1.png)

**Paradigm of the Review**:
- **a. Chemical reaction prediction tasks**: showcasing three tasks along with examples.
- **b. Current LLM methods for CRPs**: indicating rational predictions but lacking in reactive validity.
- **c. Self-feedback knowledge elicitation for enhancing CRPs**: Self-feedback knowledge elicitation for enhancing CRPs}, demonstrating the enhancement of CRPs through the refinement of knowledge patterns, notably RTs, utilizing a self-feedback knowledge elicitation technique. Knowledge elicitation serves as a method of data curation for knowledge distillation, where RT is integrated into large language models via adaptive prompt learning, facilitating the planning of reaction pathways in CRPs.

**Note**: The dataset and model sections detail respective directories. Some data and checkpoints might not be available due to size constraints and permissions.

## SLM4CRP_with_RTs
The SLM4CRP_with_RTs dataset is a CRPs dataset featuring RT labels, developed from the Mol-Instruction. We introduce a novel knowledge elicitation approach integrating a self-feedback mechanism with data curation using LLMs. This dataset embodies domain-specific knowledge by combining reactants and products of chemical reactions with annotated RTs, demonstrating that domain-integrated data can enhance the capabilities of LLMs.

### Contents
- `forward_reaction_prediction.json`: Contains data for forward reaction prediction tasks.
- `retrosynthesis.json`: Includes data for retrosynthesis tasks.
- `reagent_prediction.json`: Features data for predicting reagents in chemical reactions.
- `reactions.json`: Serves multiple tasks involving different types of chemical reactions.

### Download links
- [SLM4CRP_with_RTs Dataset](https://huggingface.co/datasets/liupf/SLM4CRP_with_RTs)

## Review of Our Approach
![Three-stage training scheme of prompt-based knowledge elicitation](figures/figure2.png)

### Developments and Architectures of Models
- **Knowledge extraction**: We divide the datasets into training, validation, and testing sets. The training set inputs and outputs are clustered using LLM-RT embeddings, facilitating RT annotations. Annotation accuracy is enhanced through iterative tuning of clustering parameters and training with inputs and RTs, aiming to pinpoint the optimal cluster settings for maximum precision.
- **Data curation**: The trained LLM-RT model annotates RTs for the validation and testing datasets based on their input configurations.
- **Adaptive knowledge injection**: Adaptability scores are derived from the embeddings of inputs and instructions, guiding the selection of the most effective adaptive instructions.

## Code Structure

### `ckpts`
Directory containing checkpoints used for different purposes:
- **finetune_ckpts**: Checkpoints from fine-tuning processes.
- **text_ckpts**: Contains the checkpoints related to text models.
    - [Text+Chem T5](https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-base-augm): A model checkpoint for text and chemical data integration.

### `datasets/Mol`
This directory includes materials for working with molecular data:
- **SMILES/type**: Contains training data in SMILES format.
- **data_process.ipynb**: Notebook for preprocessing the dataset.

### `src`
Source code directory housing the implementation details:
- **`datasets`**: Code for constructing datasets.
    - `dataset_manager.py`: Creation of datasets for adaptive knowledge injection.
    - `dataset_manager_label.py`: Creation of datasets for knowledge elicitation.
- **`evaluations`**: Scripts for computing various evaluation metrics.
- **`models`**: Core models for the tasks.
    - `init.py`: Initializes model parameters and settings.
    - `model_manager.py`: Manages the loading and handling of models for knowledge injection.
    - `model_manager_label.py`: Manages the loading and handling of models for knowledge elicitation.
    - `chemT5.py`: Text+Chem T5.
- **`utils`**: Utility functions and initializations.
    - `init.py`: General utility tool initialization.
    - `xutils.py`: Advanced and specialized utility tool initialization.
- **`task_manager.py`**: Function to execute tasks related to adaptive knowledge injection.
- **`task_manager_label.py`**: Function to execute tasks related to knowledge elicitation.

### Detailed Parameter Explanations for Tasks
- `mode`: Select the operation mode. Options include `data_check`, `encoder_check`, `train`, and `eval`.
- `N`: Number of clusters.
- `reaction_type`: Specifies whether to include RT during training.
- `task`: Task type. Options include `forward`, `retro`, `reagent`, and `reactions`.
- `batch_size`: Set the batch size for operations.

## Knowledge Elicitation
<img src="figure3.png" alt="Performance of encoding vector self-feedback annotation and clustering" width="90%">
- **Accuracy of RT annotations across encoding vectors and clustering numbers**: We assess the annotation accuracy ($Acc$) among four encoding methods at various reasonable clustering numbers ($N$). The results reveal that the encoding method utilizing concatenated input-output vectors ($concat(input, output)_{vec}$) delivers the highest performance.
- **Clustering of test dataset vectors ($concat(input, output)_{vec}$)**: With $N$ set to 6 and 10, test dataset vectors are processed through a linear layer to reduce them to two dimensions, effectively visualizing the clustering outcomes.

## Case Studies
![Case studies of RT annotation](figures/figure4.png)

To validate the practical significance of RT annotation, we analyze samples filtered through the `concat(input, output)_{vec}` vector with `N=10` labeled results, focusing on samples with an RT label of 0. These instances typically involve simple atomic substitutions, verifying the predominance of substitution reactions in these cases. This analysis highlights the real-world relevance of our RT annotation method.

## References
```
[1] Xu Z, Li J, Yang Z, et al. SwinOCSR: end-to-end optical chemical structure recognition using a Swin Transformer[J]. Journal of Cheminformatics, 2022, 14(1): 1-13.
[2] Su B, Du D, Yang Z, et al. A molecular multimodal foundation model associating molecule graphs with natural language[J]. arXiv preprint arXiv:2209.05481, 2022.(https://arxiv.org/abs/2209.05481)
[3] Edwards C, Lai T, Ros K, et al. Translation between molecules and natural language[J]. arXiv preprint arXiv:2204.11817, 2022.
[4] Luo Y, Yang K, Hong M, et al. Molfm: A multimodal molecular foundation model[J]. arXiv preprint arXiv:2307.09484, 2023.
```
## Acknowledgments
The development of the ChEBI-20-MM dataset was inspired by the ChEBI-20 in molecule generation and captioning initiated by MolT5. Additional data information supplements are derived from PubChem. The initial model weights provided by Hugging Face are crucial components in our evaluation process.

## Citation
```
@misc{liu2024scientific,
      title={Scientific Language Modeling: A Quantitative Review of Large Language Models in Molecular Science}, 
      author={Pengfei Liu and Jun Tao and Zhixiang Ren},
      year={2024},
      eprint={2402.04119},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
