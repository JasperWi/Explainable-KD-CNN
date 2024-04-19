# Explainable-KD-CNN

This repository contains the code for my bachelor thesis, focusing on evaluating the influence of knowledge distillation on the explainability of convolutional neural networks for image classification tasks.


## Installation

To use the scripts in this repository, you'll need Python installed on your system. This repository uses Conda for managing dependencies. You can install the required packages using the following command:

    conda install --file requirements.txt


Additionally you will need to install the Quantus library using the following command:
    
    pip install git+https://github.com/understandable-machine-intelligence-lab/Quantus.git

To use the scripts in this repository, you will need to download the ImageNet dataset. You can find more information on how to download the dataset on their [official website](http://www.image-net.org/).

For the evaluation of the explainability of the models, you will need to additionally download the ImageNet_S{50} dataset from this [repository](https://github.com/LUSSeg/ImageNet-S).

## Usage

This repository contains the following two scripts:

1. `knowledge_distillation_training.py`: This script can be used to train a student model based on the MobileNetV3-small architecture on the ImageNet dataset using response-based knowledge distillation. The script will use a ResNet-50 teacher model to generate the soft targets for the student model in combination with the hard targets from the ImageNet dataset. For this purpose, the script will train multiple student models for different configurations of the influence of knowledge distillation loss on the total loss. The script will save the trained student model to the specified output directory.

    To use the script, you can run the following command:

        python knowledge_distillation_training.py --epochs <num_epochs> --learning_rate <lr> --temperature <temperature> --num_models <num_models> --output_directory <output_dir> --imageNet_path <imageNet_path>

    Arguments:
    - `--epochs`: Number of epochs to train the model. (Default: 160)
    - `--learning_rate`: Learning rate for the optimizer. (Default: 0.003)
    - `--temperature`: Temperature for the knowledge distillation. (Default: 50)
    - `--num_models`: Number of models to train for each configuration. (Default: 10)
    - `--output_directory`: Directory to save the trained models and training results. (Default: './output')
    - `--imageNet_path`: Path to the ImageNet dataset.

2. `assess_explainability`: This script can be used to assess the explainability of a trained model using the Quantus library. The script will generate explanations for the specified model using the specified images and save the explanations to the specified output directory.

    To use the script, you can run the following command:

        python assess_explainability.py --path_to_checkpoints <path_to_checkpoints> --path_to_data <path_to_data> --output_folder <output_folder> --num_models <num_models>


    Arguments:
    - `--path_to_checkpoints`: Path to the checkpoint files. (Default: './output')
    - `--path_to_data`: Path to the ImageNet_S dataset.
    - `--output_folder`: Path to the output folder. (Default: './output')
    - `--num_models`: Number of models to evaluate. (Default: 5)


## Acknowledgements

The procedure for knowledge distillation is adopted from the code of the paper by Hinton et al. [1]. Their original implemenation can be found under [https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/distillation](this link).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

<a id="1">[1]</a>
Hinton, G., Vinyals, O., & Dean, J. (2015).
Distilling the Knowledge in a Neural Network.
arXiv preprint arXiv:1503.02531.