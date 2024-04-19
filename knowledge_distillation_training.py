import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.models import resnet50, ResNet50_Weights
import csv
import argparse
import time


#####################################
#           Data Preparation        #
#####################################

def create_dataloader(selected_classes, imageNet_path):
    """
    Create data loaders for ImageNet dataset with only selected classes

    :param selected_classes: List of class indices that should be used for training
    :param imageNet_path: Path to the ImageNet dataset
    :return: data loader for the filtered training and test dataset
    """

    # Define the path to your ImageNet dataset
    data_dir = imageNet_path

    # Define the transformation pipeline for training data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing()  # Adding RandomErasing
    ])

    # Define the transformation pipeline for test data
    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create ImageFolder datasets
    train_dataset = ImageFolder(root=data_dir + '/ILSVRC2012_img_train', transform=train_transform)
    test_dataset = ImageFolder(root=data_dir + '/val', transform=test_transform)

    # Define batch size
    batch_size = 128

    # Filter samples from the training dataset
    train_indices = [idx for idx, (_, class_label) in enumerate(train_dataset.samples) if class_label in selected_classes]

    # Create a Subset of the training dataset with only selected classes
    filtered_train_dataset = Subset(train_dataset, train_indices)

    # Filter samples from the test dataset
    test_indices = [idx for idx, (_, class_label) in enumerate(test_dataset.samples) if class_label in selected_classes]

    # Create a Subset of the test dataset with only selected classes
    filtered_test_dataset = Subset(test_dataset, test_indices)

    # Create data loaders for the filtered datasets
    filtered_train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers = True)
    filtered_test_loader = DataLoader(filtered_test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers = True)
    return filtered_train_loader, filtered_test_loader

#####################################
#           Training Functions      #
#####################################


def test_accuracy(model, test_loader):
    """
    Calculate the accuracy of the model on the test dataset

    :param model: The model to evaluate
    :param test_loader: The data loader for the test dataset
    :return: The accuracy of the model on the test dataset
    """

    model.to("cuda")
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the inputs and labels to the GPU
            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            # Forward pass
            outputs = model(inputs)
            # Get the predicted class
            _, predicted = torch.max(outputs.data, 1)

            # Update the total and correct counts
            total += labels.size(0)

            # Check how many predictions are correct
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train_with_knowledge_distillation(teacher, student, train_loader, T, optimizer, ce_loss, distillation_factor):
    """
    Train the student model with a mix of knowledge distillation and the true label loss.

    :param teacher: The teacher model
    :param student: The student model
    :param train_loader: The data loader for the training dataset
    :param T: The temperature for the knowledge distillation
    :param optimizer: The optimizer for the student model
    :param ce_loss: The cross-entropy loss function
    :param distillation_factor: The factor for the knowledge distillation loss. 0 means only the true label loss is used, 1 means only the knowledge distillation loss is used and in between a weighted sum of both losses is used.
    :return: The average loss, the accuracy, the cross-entropy loss and the KD loss
    """

    # Log the start time
    elapsed_time = 0
    start_time = time.time()

    running_loss = 0.0
    total_correct = 0
    total_samples = 0
        
    for inputs, labels in train_loader:
        inputs, labels = inputs.to("cuda"), labels.to("cuda")

        optimizer.zero_grad()
        # Forward pass with the student model
        student_logits = student(inputs)
        
        # As long as KD is not disabled, calculate the logits of the teacher model
        if distillation_factor != 0:
            # Forward pass with the teacher model
            with torch.no_grad():
                teacher_logits = teacher(inputs)
        
            # Calculate the soft target loss
            soft_targets = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = torch.nn.functional.log_softmax(student_logits / T, dim=-1)
            soft_targets_loss = torch.nn.functional.kl_div(soft_prob, soft_targets, reduction='batchmean')

        # Calculate the true label loss
        label_loss = ce_loss(student_logits, labels)

        ce_loss_value = label_loss.item()
        kd_loss_value = 0

        # If only KD is used as loss, set the loss to the KD loss
        if distillation_factor == 1:
            loss = soft_targets_loss
            kd_loss_value = soft_targets_loss.item()
        # If both KD and the true label loss are used, then calculate the weighted sum of both losses
        elif distillation_factor != 0:
            # The value 40000 is used as a rough scaling factor to make the KD loss comparable to the true label loss
            loss = distillation_factor * soft_targets_loss * 40000 + (1 - distillation_factor) * label_loss
            kd_loss_value = soft_targets_loss.item() * 40000
        # If only the true label loss is used, set the loss to the true label loss
        else:
            loss = label_loss
        loss.backward()


        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1)
        
        # Update the weights
        optimizer.step()

        # Log the loss
        running_loss += loss.item()

    # Calculate accuracy
    total_correct += (torch.argmax(student_logits, 1) == labels).sum().item()
    total_samples += labels.size(0)
    accuracy = total_correct / total_samples * 100

    # Print the time per epoch
    elapsed_time = time.time() - start_time
    print(f"Time per Epoch: {elapsed_time:.2f}s")
    return running_loss / len(train_loader), accuracy, ce_loss_value, kd_loss_value

def train_student_models(output_dir, epochs, teacher, train_loader, test_loader, number_of_models, temperature, learning_rate, csv_file):
    """
    Train the student models with knowledge distillation, log the results and save the trained models.
    For this four different loss factors are used. The loss factor 0 means that only the true label loss is used.
    The loss factor 1 means that only the knowledge distillation loss is used. The loss factors 0.25 and 0.75 are used
    to calculate a weighted sum of both losses.

    :param output_dir: The directory where the trained models and the training results should be saved
    :param epochs: The number of epochs to train the models
    :param teacher: The teacher model
    :param train_loader: The data loader for the training dataset
    :param test_loader: The data loader for the test dataset
    :param number_of_models: The number of models to train for each configuration
    :param temperature: The temperature for the knowledge distillation
    :param learning_rate: The learning rate for the optimizer
    :param csv_file: The CSV file for logging the training results
    """


    for model_number in range(number_of_models):
        for loss_factor_kd in [0.5,0,0.25,0.75,1]:
            # Load the student model
            student =  mobilenet_v3_small()

            # Define the optimizer and the loss function
            optimizer = optim.RAdam(student.parameters(), lr=learning_rate)
            ce_loss = nn.CrossEntropyLoss()

            # Send both models to the GPU
            teacher.to("cuda")
            student.to("cuda")
            teacher.eval()  # Teacher set to evaluation mode
            student.train() # Student to train mode

            # Define the hyperparameters
            for epoch in range(epochs):
                loss, train_acc, ce_loss_value, kd_loss_value = train_with_knowledge_distillation(teacher, student, train_loader, temperature, optimizer, ce_loss, loss_factor_kd)
                val_acc = test_accuracy(student, test_loader)

                # Log results to CSV
                write_to_csv(csv_file, epoch, learning_rate, temperature, model_number, loss, train_acc, val_acc, loss_factor_kd, ce_loss_value, kd_loss_value)

            # Save the trained model
            model_filename = f'{output_dir}/model_distillation_{loss_factor_kd}_fold_{model_number}.pth'
            torch.save(student.state_dict(), model_filename)


#####################################
#           Logging Functions       #
#####################################

def prepare_csv_file(dir):
    """
    Prepare the CSV file for logging the training results

    :param dir: The directory where the CSV file should be saved
    """


    # Define CSV file name
    csv_file = dir + '/results.csv'

    # Open CSV file for writing the results
    with open(csv_file, mode='a', newline='') as file:
        # Create CSV writer
        csv_writer = csv.writer(file)
        # Write header
        csv_writer.writerow(['Epoch', 'Learning Rate', 'Temperature', 'Model Nr',  'Loss', 'Training Accuracy', 'Validation Accuracy', 'Distillation'])


def write_to_csv(dir, epoch, learning_rate, temperature, model_number, loss, train_acc, val_acc, loss_factor_kd, ce_loss_value, kd_loss_value):
    """
    Write the training results to the CSV file

    :param dir: The directory where the CSV file should be saved
    :param epoch: The current epoch
    :param learning_rate: The learning rate
    :param temperature: The temperature for the knowledge distillation
    :param model_number: The model number
    :param loss: The loss value
    :param train_acc: The training accuracy
    :param val_acc: The validation accuracy
    :param loss_factor_kd: The factor for the knowledge distillation loss
    :param ce_loss_value: The cross-entropy loss value
    :param kd_loss_value: The knowledge distillation loss value
    """

    csv_file = dir + '/results.csv'
    with open(csv_file, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([epoch+1, learning_rate, temperature, model_number, loss, train_acc, val_acc, loss_factor_kd])
    print(f"Epoch {epoch+1}, Learningrate: {learning_rate},Temperature: {temperature}, ModelNR: {model_number}, Loss: {loss}, Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}, Distillation: {loss_factor_kd}, CE_Loss: {ce_loss_value}, KD_Loss: {kd_loss_value}")


#####################################
#           Main Function           #
#####################################

def main():
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')

    # Overview over all arguments:
    # 1. --epochs: number of epochs to train the each model
    # 2. --learning_rate: learning rate for the optimizer
    # 3. --temperature: temperature for the knowledge distillation
    # 4. --num_models: number of models to train for each configuration
    # 5. --output_directory: directory to save the trained models and the training results
    # 6. --imageNet_path: path to the ImageNet dataset

    parser.add_argument('--epochs', type=int, default=160, help='Number of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate for the optimizer')
    parser.add_argument('--temperature', type=float, default=50, help='Temperature for the knowledge distillation')
    parser.add_argument('--num_models', type=int, default=10, help='Number of models to train for each configuration')
    parser.add_argument('--output_directory', type=str, default='./output', help='Directory to save the trained models and the training results')
    parser.add_argument('--imageNet_path', type=str, default='C:/Users/Admin/Desktop/Thesis', help='Path to the ImageNet dataset')

    # Read the command line arguments
    epochs = parser.parse_args().epochs
    temperature = parser.parse_args().temperature
    learning_rate = parser.parse_args().learning_rate
    num_models = parser.parse_args().num_models
    output_directory = parser.parse_args().output_directory
    imageNet_path = parser.parse_args().imageNet_path


    # Define the indeces of the selected classes from the ImageNet dataset that are used for training
    selected_classes = [1, 3, 11, 31, 222, 277, 284, 295, 301, 325, 330, 333, 342, 368, 386, 388, 404, 412, 418, 436, 449, 466, 487, 492, 502, 510, 531, 532, 574, 579, 606, 617, 659, 670, 695, 703, 748, 829, 846, 851, 861, 879, 883, 898, 900, 914, 919, 951, 959, 992]

    # Create data loaders for the filtered datasets
    train_loader, test_loader = create_dataloader(selected_classes, imageNet_path)

    # Load the pre-trained ResNet-50 model
    teacher = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Prepare the CSV file
    prepare_csv_file(output_directory)

    # Start training loop
    train_student_models(output_directory, epochs, teacher, train_loader, test_loader, num_models, temperature, learning_rate, output_directory)

if __name__ == "__main__":
    main()