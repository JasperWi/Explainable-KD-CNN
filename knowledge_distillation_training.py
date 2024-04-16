import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import KFold
import csv

# Define the indices of the 10 classes you want to use (assuming they are consecutive)
selected_classes = [1, 3, 11, 31, 222, 277, 284, 295, 301, 325, 330, 333, 342, 368, 386, 388, 404, 412, 418, 436, 449, 466, 487, 492, 502, 510, 531, 532, 574, 579, 606, 617, 659, 670, 695, 703, 748, 829, 846, 851, 861, 879, 883, 898, 900, 914, 919, 951, 959, 992]
non_selected_classes = set(range(1000)) - set(selected_classes)

def create_dataloader():
    # Define the path to your ImageNet dataset
    data_dir = 'C:/Users/Admin/Desktop/Thesis'

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

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    it = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    #print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

import time


def train_knowledge_distillation(teacher, student, train_loader, epochs, T, soft_target_loss_weight, ce_loss_weight, optimizer, ce_loss, disti):
    elapsed_time = 0
    start_time = time.time()

    ce_loss_value = 0
    kd_loss_value = 0

    for epoch in range(epochs):
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            optimizer.zero_grad()

            student_logits = student(inputs)
            
            if disti != 0:
                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
            
                soft_targets = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = torch.nn.functional.log_softmax(student_logits / T, dim=-1)

                soft_targets_loss = torch.nn.functional.kl_div(soft_prob, soft_targets, reduction='batchmean')

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            #print(str(label_loss.item()) + " : " + str(soft_targets_loss.item()))
            ce_loss_value = label_loss.item()
            kd_loss_value = 0
            # Weighted sum of the two losses
            if disti == 1:
                loss = soft_targets_loss
                kd_loss_value = soft_targets_loss.item()
            elif disti != 0:
                loss = ce_loss_weight * label_loss + soft_target_loss_weight * soft_targets_loss * 40000
                kd_loss_value = soft_targets_loss.item() * 40000
            else:
                loss = label_loss
            loss.backward()



            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1)
            
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            total_correct += (torch.argmax(student_logits, 1) == labels).sum().item()
            total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        elapsed_time = time.time() - start_time
        print(f"Time per Epoch: {elapsed_time:.2f}s")
        start_time = time.time()
    return running_loss / len(train_loader), accuracy * 100, ce_loss_value, kd_loss_value



def main():
    filtered_train_loader, filtered_test_loader = create_dataloader()

    teacher = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Define CSV file name
    csv_file = './FINFINFIN/run.csv'

    # Open CSV file for writing
    with open(csv_file, mode='a', newline='') as file:
        # Create CSV writer
        csv_writer = csv.writer(file)
        # Write header
        csv_writer.writerow(['Epoch', 'Learning Rate', 'Temperature', 'Model Nr',  'Loss', 'Training Accuracy', 'Validation Accuracy', 'Distillation'])

    for nr in range(7,30):
        for learning_rate in [0.003]:
            for disti in [0.5,0,0.25,0.75,1]:
                for temperature in [50]:
                    student =  mobilenet_v3_small()
                    optimizer = optim.RAdam(student.parameters(), lr=learning_rate)
                    ce_loss = nn.CrossEntropyLoss()

                    teacher.to("cuda")
                    student.to("cuda")
                    teacher.eval()  # Teacher set to evaluation mode
                    student.train() # Student to train mode

                    for rep in range(16):
                        loss, train_acc, ce_loss_value, kd_loss_value = train_knowledge_distillation(teacher, student, filtered_train_loader, 10, temperature, disti, 1-disti, optimizer, ce_loss, disti)
                        val_acc = test(student, filtered_test_loader, "cuda")

                        # Log results to CSV
                        with open(csv_file, mode='a', newline='') as file:
                            # Create CSV writer
                            csv_writer = csv.writer(file)
                            csv_writer.writerow([(rep+1)*10, learning_rate, temperature, nr, loss, train_acc, val_acc, disti])
                        print(f"Epoch {(rep+1)*10}/{16*10}, Learningrate: {learning_rate},Temperature: {temperature}, ModelNR: {nr}, Loss: {loss}, Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}, Distillation: {disti}, CE_Loss: {ce_loss_value}, KD_Loss: {kd_loss_value}")
                    model_filename = f'./FINFINFIN/model_distillation_{disti}_fold_{nr}.pth'
                    torch.save(student.state_dict(), model_filename)

if __name__ == "__main__":
    main()