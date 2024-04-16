import torch
import quantus
import torch
import torchvision
from torchvision import transforms
import numpy as np
import gc
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import csv
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights
import os
from PIL import Image
import cv2

# Define the indices of the 10 classes you want to use (assuming they are consecutive)
selected_classes = [1, 3, 11, 31, 222, 277, 284, 295, 301, 325, 330, 333, 342, 368, 386, 388, 404, 412, 418, 436, 449, 466, 487, 492, 502, 510, 531, 532, 574, 579, 606, 617, 659, 670, 695, 703, 748, 829, 846, 851, 861, 879, 883, 898, 900, 914, 919, 951, 959, 992]
selected_original_classes = [
    450, 443, 387, 500, 141, 62, 95, 163, 622, 645,
    188, 157, 78, 185, 24, 169, 230, 752, 907, 266,
    689, 887, 914, 762, 979, 243, 529, 315, 792, 227,
    659, 918, 829, 260, 584, 306, 939, 287, 304, 944,
    312, 220, 874, 958, 795, 240, 932, 320, 999, 913
]
non_selected_classes = set(range(1000)) - set(selected_classes)

def create_dataloader():
    # Define the path to your ImageNet dataset
    data_dir = 'C:/Users/Admin/Desktop/Thesis2\ImageNetS50'
    #data_dir = 'C:/Users/Admin/Desktop/Thesis'

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #'/validation-segmentation_original'
    test_dataset = ImageFolder(root=data_dir + '/validation-segmentation_original', transform=test_transform)

    # Define batch size
    batch_size = 616

    filtered_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    return filtered_test_loader

def create_plain_dataloader():
    data_dir = 'C:/Users/Admin/Desktop/Thesis2\ImageNetS50'
        # Define the transformation pipeline for test data

    plain_test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    batch_size = 616
    plain_test_dataset = ImageFolder(root=data_dir + '/validation-segmentation', transform=plain_test_transform)

    plain_filtered_test_loader = DataLoader(plain_test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return plain_filtered_test_loader


def show_images(x_batch, y_batch, a_batch):
    num_images = len(x_batch)

    # Define the layout of subplots
    rows = num_images
    cols = 3  # Three columns for image, label, and Grad-CAM view

    # Create a figure and set the size based on the number of images
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3*num_images))

    for i in range(num_images):
        # Plot the original image
        normalized_image = x_batch[i].transpose(1, 2, 0)# Transpose to (224, 224, 3) and normalize
        axes[i, 0].imshow(normalized_image, alpha=1, vmin=0, vmax=1)
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Image')

        # Display the label
        axes[i, 1].text(0.5, 0.5, f"Label: {y_batch[i]}", ha='center', va='center', fontsize=8, transform=axes[i, 1].transAxes)
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Label')

        # Plot the Grad-CAM view
        normalized_cam_view = a_batch[i].transpose(1, 2, 0)#a_batch[i] / np.max(a_batch[i])  # Normalize Grad-CAM view
        axes[i, 2].imshow(normalized_cam_view, cmap='jet', alpha=1, vmin=0, vmax=1)
        #axes[i, 2].imshow(normalized_image, alpha=0.5)  # Overlay original image
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Grad-CAM View')

    plt.tight_layout()
    plt.show()

def test(s,x,a):
        if np.sum(s) == 0:
            print("1")
            return np.nan

        # Prepare shapes.
        a = a.flatten()
        s = s.flatten().astype(bool)

        # Compute ratio.
        size_bbox = float(np.sum(s))
        size_data = np.prod(x.shape[1:])
        ratio = size_bbox / size_data

        # Compute inside/outside ratio.
        inside_attribution = np.sum(a[s])
        total_attribution = np.sum(a)
        inside_attribution_ratio = float(inside_attribution / total_attribution)
        return inside_attribution_ratio
        """
        if not ratio <= self.max_size:
            warn.warn_max_size()
        if inside_attribution_ratio > 1.0:
            warn.warn_segmentation(inside_attribution, total_attribution)
            return np.nan
        if not self.weighted:
            return inside_attribution_ratio
        else:
            return float(inside_attribution_ratio * ratio)"""
        
def save_images(array, output_folder='resultss'):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    array*=255
    # Iterate through the images and save them
    for i in range(array.shape[0]):
        image_array = np.squeeze(array[i])  # Remove singleton dimension
        image = Image.fromarray(image_array.astype(np.uint8))

        # Save the image with a unique filename
        image_filename = os.path.join(output_folder, f"image_{i + 1}.png")
        image.save(image_filename)

        print(f"Image {i + 1} saved: {image_filename}")


def get_pixels_between_percentiles(heatmap, lower_percentile, upper_percentile):
    heatmap_flat = heatmap.flatten()
    sorted_indices = np.argsort(heatmap_flat)
    num_pixels = len(sorted_indices)
    lower_index = int(lower_percentile / 100 * num_pixels)
    upper_index = int(upper_percentile / 100 * num_pixels)
    pixel_indices = np.unravel_index(sorted_indices[lower_index:upper_index], heatmap.shape)
    return pixel_indices

def obscure_pixels(original_image, pixel_coords):
    obscured_image = original_image.copy()
    for x, y in zip(*pixel_coords):
        obscured_image[:, x, y] = 0
    return obscured_image



def asses(CamVersion, model, x_batch, y_batch, x_load, s_batch):
    y_batch = [selected_classes[i] for i in y_batch]

    target_layers = [model.features[-1]] #[model.layer4[-1]] #
    input_tensor = x_load[0].unsqueeze(0)
    cam = CamVersion(model=model, target_layers=target_layers, use_cuda=True)
    a_batch = np.zeros((len(x_load), *x_load[0].shape[1:]))

    for i, single_image in enumerate(x_load):
        if i == 616:
            break
        input_tensor = single_image.unsqueeze(0)
        targets = [ClassifierOutputTarget(y_batch[i])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True)[0, :]
        if grayscale_cam.max() == 0:
            grayscale_cam[0,0] += 0.001
        a_batch[i] = grayscale_cam

    a_batch = a_batch[:616]

    print("saliency maps created")

    #show_images(plain_x_batch[[0,10,20,30,40,50]], y_batch[[0,10,20,30,40,50]], a_batch[[0,10,20,30,40,50]])

    for index in range(616):
        if np.isinf(x_batch[index]).any():
            print(index, "Infs x_batch")
        elif x_batch[index].max() <= 0:
            print(index, x_batch[index].max(), "x_batch")
        if np.isinf(y_batch[index]).any():
            print(index, "Infs y_batch")
        if np.isinf(a_batch[index]).any():
            print(index, "Infs a_batch")
        elif a_batch[index].max() <= 0:
            print(index, a_batch[index].max(), "a_batch")

        if np.isnan(x_batch[index]).any():
            print(index, "Nans x_batch")
        if np.isnan(y_batch[index]).any():
            print(index, "Nans y_batch")
        if np.isnan(a_batch[index]).any():
            print(index, "Nans a_batch")
        
    #x_batch, s_batch, y_batch = x_batch.to("cuda"), s_batch.to("cuda"), y_batch.to("cuda")

    x1_batch = np.concatenate([x_batch[:323,:,:,:], x_batch[323 + 1:,:,:,:]],axis=0)
    a1_batch = np.concatenate([a_batch[:323,:,:], a_batch[323 + 1:,:,:]],axis=0)
    y1_batch = np.concatenate([y_batch[:323], y_batch[323 + 1:]],axis=0)
    s1_batch = np.concatenate([s_batch[:323,:,:,:], s_batch[323 + 1:,:,:,:]],axis=0)

    """     Selectivity_Score = quantus.RegionPerturbation(
    patch_size=14,
    regions_evaluation=10,
    perturb_baseline="uniform",  
    normalise=True,
    )(model=model,
    x_batch=x1_batch,
    y_batch=y1_batch,
    a_batch=a1_batch,
    s_batch=s1_batch,
    device="cuda")
    Selectivity_Score = [np.corrcoef(x, np.arange(len(x)))[0, 1] for x in Selectivity_Score]
    print("Selectivity_Score") """

    prediction_values = []
    for index, image in enumerate(x1_batch):
        predictions = []
        input_tensors = []
        with torch.no_grad():
            preds = torch.nn.functional.softmax(model(torch.tensor(image, dtype=torch.float32).unsqueeze(0).to("cuda")), dim=-1)[0]
            predictions.append(preds[y1_batch[index]].item())
        for percetage in range(0, 100, 10):
            pixel_coords = get_pixels_between_percentiles(a1_batch[index], percetage, percetage +10)
            obscured_image = obscure_pixels(image, pixel_coords)
            with torch.no_grad():
                predictions.append(torch.nn.functional.softmax(model(torch.tensor(obscured_image, dtype=torch.float32).unsqueeze(0).to("cuda")), dim=-1)[0][y1_batch[index]].item())
        prediction_values.append(predictions)
    print("Prediction Decline")
    gc.collect()
    
    if CamVersion == GradCAM:
        AttributionLocalisation_score = quantus.AttributionLocalisation(
        disable_warnings=True
        )(model=model,
        x_batch=x1_batch,
        y_batch=y1_batch,
        a_batch=a1_batch,
        s_batch=s1_batch,
        device="cuda")
        print("AttributionLocalisation_score")
        gc.collect()

        PointingGame_score = quantus.PointingGame(
        disable_warnings=True
        )(model=model,
        x_batch=x1_batch,
        y_batch=y1_batch,
        a_batch=a1_batch,
        s_batch=s1_batch,
        device="cuda")
        print("PointingGame_score")
        gc.collect()

        RelevanceRankAccuracy_score = quantus.RelevanceRankAccuracy(
        disable_warnings=True
        )(model=model,
        x_batch=x1_batch,
        y_batch=y1_batch,
        a_batch=a1_batch,
        s_batch=s1_batch,
        device="cuda")
        print("RelevanceRankAccuracy_score")
        gc.collect()

        """RelevanceMassAccuracy_Score = quantus.RelevanceMassAccuracy(
        disable_warnings=True
        )(model=model,
        x_batch=x1_batch,
        y_batch=y1_batch,
        a_batch=a1_batch,
        s_batch=s1_batch,
        device="cuda")
        print("RelevanceMassAccuracy_Score")
        gc.collect()"""

        TopKIntersection_Score = quantus.TopKIntersection(
        disable_warnings=True
        )(model=model,
        x_batch=x1_batch,
        y_batch=y1_batch,
        a_batch=a1_batch,
        s_batch=s1_batch,
        device="cuda")
        print("TopKIntersection_Score")
        gc.collect()
    else:
        AttributionLocalisation_score = 0
        PointingGame_score = 0
        RelevanceRankAccuracy_score = 0
        #RelevanceMassAccuracy_Score = 0
        TopKIntersection_Score = 0

    Monotonicity_Score = quantus.MonotonicityCorrelation(
    nr_samples=5,
    features_in_step=3136,
    perturb_baseline="uniform",
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    similarity_func=quantus.similarity_func.correlation_spearman,
    disable_warnings=True
    )(model=model,
    x_batch=x1_batch,
    y_batch=y1_batch,
    a_batch=a1_batch,
    s_batch=s1_batch,
    device="cuda")
    print("Monotonicity_Score")
    gc.collect()

    FaithfullnessEstimate_Score = quantus.FaithfulnessEstimate(
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    similarity_func=quantus.similarity_func.correlation_pearson,
    features_in_step=224*4,  
    perturb_baseline="black",
    disable_warnings=True
    )(model=model,
    x_batch=x1_batch,
    y_batch=y1_batch,
    a_batch=a1_batch,
    s_batch=s1_batch,
    device="cuda")
    print("FaithfullnessEstimate_Score")
    gc.collect()

    

    pointing_game_mean = np.nanmean(PointingGame_score)
    attribution_localisation_mean = np.nanmean(AttributionLocalisation_score)
    relevance_rank_accuracy_mean = np.nanmean(RelevanceRankAccuracy_score)
    #relevance_mass_accuracy_mean = np.nanmean(RelevanceMassAccuracy_Score)
    top_k_intersection_mean = np.nanmean(TopKIntersection_Score)
    monotonicity_mean = np.nanmean(Monotonicity_Score)
    faithfulness_estimate_mean = np.nanmean(FaithfullnessEstimate_Score)
    #selectivity_mean = np.nanmean(Selectivity_Score)

    return (
        pointing_game_mean,
        attribution_localisation_mean,
        relevance_rank_accuracy_mean,
        #relevance_mass_accuracy_mean,
        top_k_intersection_mean,
        monotonicity_mean,
        faithfulness_estimate_mean,
        prediction_values
        #selectivity_mean
    )


def main():
    test_loader = create_dataloader()
    plain_test_loader = create_plain_dataloader()

    data_iter = iter(test_loader)
    x_load, _ = next(data_iter)

    # Initialize empty lists to store batches
    all_x_batches = []
    all_y_batches = []

    # Loop through the entire dataset
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()
        
        # Append the current batch to the lists
        all_x_batches.append(x_batch)
        all_y_batches.append(y_batch)
        break

    # Concatenate all batches into a single array
    x_batch = np.concatenate(all_x_batches, axis=0)
    y_batch = np.concatenate(all_y_batches, axis=0)

    # Initialize empty lists to store batches
    plain_all_x_batches = []

    # Loop through the entire dataset
    for plain_x_batch, _ in plain_test_loader:
        plain_x_batch = plain_x_batch.cpu().numpy()
        
        # Append the current batch to the lists
        plain_all_x_batches.append(plain_x_batch)
        break

    #print(all_y_batches[0])
    #print(all_x_batches[0])
    

    # Concatenate all batches into a single array
    plain_x_batch = np.concatenate(plain_all_x_batches, axis=0)

    plain_x_batch *= 255

    s_batch = np.zeros((616, 1,224, 224))

    print(plain_x_batch.shape)
    print(y_batch.shape)

    for index in range(616):
        for x in range(224):
            for y in range(224):
                if int(plain_x_batch[index,0,x,y])-1  == y_batch[index]:
                    s_batch[index,0,x,y] = 1.0


    # for i in range(0,200,10):
    #     # Transpose axes to bring color channels to the last dimension
    #     normalized_image = ((x_batch[i] - np.min(x_batch[i])) / (np.max(x_batch[i]) - np.min(x_batch[i])) * 255).astype(np.uint8).transpose(1, 2, 0)

    #     # Convert BGR to RGB format
    #     rgb_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)

    #     # Save the image
    #     cv2.imwrite('./images/'+str(i) + 'output_image'+str(y_batch[i])+'.jpg', rgb_image)
    #show_images(s_batch[[0,100,200,300,400,500]], y_batch[[0,100,200,300,400,500]], x_batch[[0,100,200,300,400,500]])


    with open('./FINFINFIN/asses_model3.csv', mode='a', newline='') as file:
        # Create CSV writer
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Distillation','PointingGame_score', 'AttributionLocalisation_score', 'RelevanceRankAccuracy_score', 'TopKIntersection_Score', 'Monotonicity_Score', 'FaithfullnessEstimate_Score', 'CamVersion'])

    with open('./FINFINFIN/predictions.csv', mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Distillation', 'Model_Index', 'CamVersion', 'Image_Nr'] + [str(i)+"%" for i in range(-10, 100, 10)])



    print("data loading complete")

    for disti2 in [0,25,50,75,100]:
        disti = disti2 / 100
        models = []
        for fold in range(5,15): #a loop from 
            try:
                model = mobilenet_v3_small()
                if disti == 1 or disti == 0:
                    checkpoint  = torch.load('./FINFINFIN/model_distillation_'+str(int(disti))+'_fold_'+str(fold)+'.pth')
                else:
                    checkpoint  = torch.load('./FINFINFIN/model_distillation_'+str(disti)+'_fold_'+str(fold)+'.pth')
                model.load_state_dict(checkpoint)
                model.eval()
                model.to("cuda")
                models.append(model)
            except FileNotFoundError:
                print("Model checkpoint not found for fold", fold)
                break
        for CamVersion in [XGradCAM, AblationCAM, HiResCAM, GradCAM, GradCAMPlusPlus, EigenCAM]: #AblationCAM,HiResCAM,EigenCAM,GradCAMPlusPlus
            for model_index, model in enumerate(models):
                PointingGame_score, AttributionLocalisation_score, RelevanceRankAccuracy_score, TopKIntersection_Score, Monotonicity_Score, FaithfullnessEstimate_Score, prediction_values  = asses(CamVersion, model, x_batch, y_batch, x_load, s_batch)
                with open('./FINFINFIN/asses_model3.csv', mode='a', newline='') as file:
                    # Create CSV writer
                    csv_writer = csv.writer(file)
                    csv_writer.writerow([disti, PointingGame_score, AttributionLocalisation_score, RelevanceRankAccuracy_score, TopKIntersection_Score, Monotonicity_Score, FaithfullnessEstimate_Score, CamVersion.__name__])
                    print(str(disti) , ",", str(PointingGame_score), "," , str(AttributionLocalisation_score), ",", str(RelevanceRankAccuracy_score), "," ,",", CamVersion.__name__)
                with open('./FINFINFIN/predictions.csv', mode='a', newline='') as file:
                    # Create CSV writer
                    csv_writer = csv.writer(file)
                    # Write each row of the 2D array preceded by its index
                    for index, row in enumerate(prediction_values):
                        # Prepend index to the row
                        row_with_index = [disti, model_index, CamVersion.__name__, index] + row
                        # Write the row to the CSV file
                        csv_writer.writerow(row_with_index)



if __name__ == "__main__":
    main()
