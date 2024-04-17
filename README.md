# Explainable-KD-CNN

This repository contains the code for my bachelor thesis, focusing on evaluating the influence of knowledge distillation on the explainability of convolutional neural networks for image classification tasks.


## Installation

To use the scripts in this repository, you'll need Python installed on your system. This repository uses Conda for managing dependencies. You can install the required packages using the following command:

    conda install --file requirements.txt


Additionally you will need to install the Quantus library using the following command:
    
    pip install git+https://github.com/understandable-machine-intelligence-lab/Quantus.git


## Usage

This repository contains the following two scripts:

1. `knowledge_distillation_training.py`: This script can be used to train a student model based on the MobileNetV3-small architecture on the ImageNet dataset using response-based knowledge distillation. The script will use a ResNet-50 teacher model to generate the soft targets for the student model in combination with the hard targets from the ImageNet dataset. The script will save the trained student model to the specified output directory.

To use the script, you can run the following command:

    python knowledge_distillation_training.py --epochs <num_epochs> --learning_rate <lr> --temperature <temperature> --num_models <num_models> --output_directory <output_dir> --imageNet_path <imageNet_path>

Arguments:
- `--epochs`: Number of epochs to train the model.
- `--learning_rate`: Learning rate for the optimizer.
- `--temperature`: Temperature for the knowledge distillation.
- `--num_models`: Number of models to train for each configuration.
- `--output_directory`: Directory to save the trained models and training results.
- `--imageNet_path`: Path to the ImageNet dataset.
    

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.