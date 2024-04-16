import torch
import quantus
import torch
from torchvision import transforms
import numpy as np
import gc
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import csv
import argparse

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

def calculate_prediction_drop(model, x_batch, y1_batch, a_batch):
    prediction_values = []
    for index, image in enumerate(x_batch):
        predictions = []
        with torch.no_grad():
            preds = torch.nn.functional.softmax(model(torch.tensor(image, dtype=torch.float32).unsqueeze(0).to("cuda")), dim=-1)[0]
            predictions.append(preds[y1_batch[index]].item())
        for percetage in range(0, 100, 10):
            pixel_coords = get_pixels_between_percentiles(a_batch[index], percetage, percetage +10)
            obscured_image = obscure_pixels(image, pixel_coords)
            with torch.no_grad():
                predictions.append(torch.nn.functional.softmax(model(torch.tensor(obscured_image, dtype=torch.float32).unsqueeze(0).to("cuda")), dim=-1)[0][y1_batch[index]].item())
        prediction_values.append(predictions)

def create_saliency_maps(CamVersion, model, x_batch, y_batch):
    # Selecting the last convolutional layer as the target layer to apply Grad-CAM to
    target_layers = [model.features[-1]]


    # Initialize the Grad-CAM object
    cam = CamVersion(model=model, target_layers=target_layers, use_cuda=True)

    # Create the variable to store the saliency maps
    a_batch = np.zeros((len(x_batch), *x_batch[0].shape[1:]))

    # Create the saliency maps for each image in the batch
    for i, single_image in enumerate(x_batch):
        input_tensor = single_image.unsqueeze(0)
        targets = [ClassifierOutputTarget(y_batch[i])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True)[0, :]

        # Avoid division by zero in in case the saliency map is all zeros
        if grayscale_cam.max() == 0:
            grayscale_cam[0,0] += 0.001
        a_batch[i] = grayscale_cam

    return a_batch

def run_quantus_evaluation_scripts(model, x_batch, y_batch, s_batch, a_batch):

    # Calculate the prediction drop when important pixels are removed
    print("Calculating the prediction drop")
    prediction_values = calculate_prediction_drop(model, x_batch, y_batch, a_batch)

    print("Calculating the attribution localisation score")
    AttributionLocalisation_score = quantus.AttributionLocalisation(
    disable_warnings=True
    )(model=model,
    x_batch=x_batch,
    y_batch=y_batch,
    a_batch=a_batch,
    s_batch=s_batch,
    device="cuda")

    print("Calculating the pointing game score")
    PointingGame_score = quantus.PointingGame(
    disable_warnings=True
    )(model=model,
    x_batch=x_batch,
    y_batch=y_batch,
    a_batch=a_batch,
    s_batch=s_batch,
    device="cuda")

    print("Calculating the relevance rank accuracy score")
    RelevanceRankAccuracy_score = quantus.RelevanceRankAccuracy(
    disable_warnings=True
    )(model=model,
    x_batch=x_batch,
    y_batch=y_batch,
    a_batch=a_batch,
    s_batch=s_batch,
    device="cuda")

    print("Calculating the top k intersection score")
    TopKIntersection_Score = quantus.TopKIntersection(
    disable_warnings=True
    )(model=model,
    x_batch=x_batch,
    y_batch=y_batch,
    a_batch=a_batch,
    s_batch=s_batch,
    device="cuda")

    print("Calculating the monotonicity score")
    Monotonicity_Score = quantus.MonotonicityCorrelation(
    nr_samples=5,
    features_in_step=3136,
    perturb_baseline="uniform",
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    similarity_func=quantus.similarity_func.correlation_spearman,
    disable_warnings=True
    )(model=model,
    x_batch=x_batch,
    y_batch=y_batch,
    a_batch=a_batch,
    s_batch=s_batch,
    device="cuda")

    print("Calculating the faithfulness estimate score")
    FaithfullnessEstimate_Score = quantus.FaithfulnessEstimate(
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    similarity_func=quantus.similarity_func.correlation_pearson,
    features_in_step=224*4,  
    perturb_baseline="black",
    disable_warnings=True
    )(model=model,
    x_batch=x_batch,
    y_batch=y_batch,
    a_batch=a_batch,
    s_batch=s_batch,
    device="cuda")

    # Calculate the mean of the scores across all samples
    pointing_game_mean = np.nanmean(PointingGame_score)
    attribution_localisation_mean = np.nanmean(AttributionLocalisation_score)
    relevance_rank_accuracy_mean = np.nanmean(RelevanceRankAccuracy_score)
    top_k_intersection_mean = np.nanmean(TopKIntersection_Score)
    monotonicity_mean = np.nanmean(Monotonicity_Score)
    faithfulness_estimate_mean = np.nanmean(FaithfullnessEstimate_Score)

    return (
        pointing_game_mean,
        attribution_localisation_mean,
        relevance_rank_accuracy_mean,
        top_k_intersection_mean,
        monotonicity_mean,
        faithfulness_estimate_mean,
        prediction_values
    )

def prepare_data(path_to_data, selected_classes):
    original_images_loader = create_dataloader_orginal_images(path_to_data)
    segmentation_map_loader = create_dataloader_segmentation_maps(path_to_data)

    # Load the entire dataset of original images
    x_batch, y_batch = next(iter(original_images_loader))

    # Convert the tensors to numpy arrays
    x_batch = x_batch.cpu().numpy()
    y_batch = y_batch.cpu().numpy()

    # Load the entire dataset of segmentation maps
    x_batch_segmentations, _ = next(iter(segmentation_map_loader))    

    # Convert the tensors to numpy arrays
    x_batch_segmentations = x_batch_segmentations.cpu().numpy()

    # Convert the segmentation maps to binary masks
    x_batch_segmentations *= 255

    # Create the binary masks for the labeled class of each image
    s_batch = np.zeros((616, 1,224, 224))
    for index in range(616):
        for x in range(224):
            for y in range(224):
                if int(x_batch_segmentations[index,0,x,y])-1  == y_batch[index]:
                    s_batch[index,0,x,y] = 1.0


    y_batch = [selected_classes[i] for i in y_batch]

    # Filter out classes where the segmentation map is all zeros
    x_batch = np.concatenate([x_batch[:323,:,:,:], x_batch[323 + 1:,:,:,:]],axis=0)
    y_batch = np.concatenate([y_batch[:323], y_batch[323 + 1:]],axis=0)
    s_batch = np.concatenate([s_batch[:323,:,:,:], s_batch[323 + 1:,:,:,:]],axis=0)

    return y_batch, s_batch, x_batch


def create_dataloader_orginal_images(path_to_data):

    # Define the transformation for the original images
    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set batch size so that the entire dataset is loaded at once
    batch_size = 616

    # Create a DataLoader for the test dataset of original images
    test_dataset = ImageFolder(root=path_to_data + '/validation-segmentation_original', transform=test_transform)
    filtered_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    return filtered_test_loader

def create_dataloader_segmentation_maps(path_to_data):

    # Define the transformation for the segmentation maps
    plain_test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Set batch size so that the entire dataset is loaded at once
    batch_size = 616
 
    # Create a DataLoader for the test dataset of segmentation maps
    plain_test_dataset = ImageFolder(root=path_to_data + '/validation-segmentation', transform=plain_test_transform)
    plain_filtered_test_loader = DataLoader(plain_test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return plain_filtered_test_loader

def prepare_csv_files(output_folder):
    # Create the CSV file for the score values of the explainability metrics
    with open(output_folder + '/explainability_metrics.csv', mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Distillation','PointingGame_score', 'AttributionLocalisation_score', 'RelevanceRankAccuracy_score', 'TopKIntersection_Score', 'Monotonicity_Score', 'FaithfullnessEstimate_Score', 'CamVersion'])

    # Create the CSV file for how the prediction values change when important pixels are removed
    with open(output_folder + '/prediction_value_drops.csv', mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Distillation', 'Model_Index', 'CamVersion', 'Image_Nr'] + [str(i)+"%" for i in range(-10, 100, 10)])

def save_results(output_folder, kd_factor, model_index, CamVersion, PointingGame_score, AttributionLocalisation_score, RelevanceRankAccuracy_score, TopKIntersection_Score, Monotonicity_Score, FaithfullnessEstimate_Score, prediction_values):
    with open(output_folder + 'explainability_metrics.csv', mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([kd_factor, PointingGame_score, AttributionLocalisation_score, RelevanceRankAccuracy_score, TopKIntersection_Score, Monotonicity_Score, FaithfullnessEstimate_Score, CamVersion.__name__])
        print(str(kd_factor) , ",", str(PointingGame_score), "," , str(AttributionLocalisation_score), ",", str(RelevanceRankAccuracy_score), "," ,",", CamVersion.__name__)

    with open(output_folder + 'prediction_value_drops.csv', mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        for index, row in enumerate(prediction_values):
            row_with_index = [kd_factor, model_index, CamVersion.__name__, index] + row
            csv_writer.writerow(row_with_index)

def load_model(path_to_checkpoints, num_models, kd_factor):
    models = []
    # Load all models for the current knowledge distillation factor
    for model_num in range(num_models):
        try:
            model = mobilenet_v3_small()

            if kd_factor == 1 or kd_factor == 0:
                checkpoint  = torch.load(path_to_checkpoints + '/model_distillation_'+str(int(kd_factor))+'_fold_'+str(model_num)+'.pth')
            else:
                checkpoint  = torch.load(path_to_checkpoints + 'model_distillation_'+str(kd_factor)+'_fold_'+str(model_num)+'.pth')
            model.load_state_dict(checkpoint)

            # Set the model to evaluation mode and move it to the GPU
            model.eval()
            model.to("cuda")

            models.append(model)

        except FileNotFoundError:
            print("Model checkpoint not found for knowledge distillation factor", kd_factor, "and model number", model_num)
            break
    
    return models


def main():
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Assess the explainability of a model using Grad-CAM and the Quantus metrics')

    # Overview over all arguments:
    # 1. --path_to_checkpoints: Path to the checkpoint files
    # 2. --path_to_data: Path to the ImageNet dataset
    # 3. --output_folder: Path to the output folder
    # 4. --num_models: Number of models to evaluate

    # Add the arguments to the parser
    parser.add_argument('--path_to_checkpoints', type=str, default='./output', help='Path to the checkpoint files')
    parser.add_argument('--path_to_data', type=str, default='C:/Users/Admin/Desktop/Thesis2/ImageNetS50', help='Path to the ImageNet dataset')
    parser.add_argument('--output_folder', type=str, default='./output', help='Path to the output folder')
    parser.add_argument('--num_models', type=int, default=5, help='Number of models to evaluate')

    # Save the arguments to variables
    path_to_checkpoints = parser.parse_args().path_to_checkpoints
    path_to_data = parser.parse_args().path_to_data
    output_folder = parser.parse_args().output_folder
    num_models = parser.parse_args().num_models


    # Define the indices of the 10 classes you want to use (assuming they are consecutive)
    selected_classes = [1, 3, 11, 31, 222, 277, 284, 295, 301, 325, 330, 333, 342, 368, 386, 388, 404, 412, 418, 436, 449, 466, 487, 492, 502, 510, 531, 532, 574, 579, 606, 617, 659, 670, 695, 703, 748, 829, 846, 851, 861, 879, 883, 898, 900, 914, 919, 951, 959, 992]
    selected_original_classes = [450, 443, 387, 500, 141, 62, 95, 163, 622, 645,188, 157, 78, 185, 24, 169, 230, 752, 907, 266,689, 887, 914, 762, 979, 243, 529, 315, 792, 227,659, 918, 829, 260, 584, 306, 939, 287, 304, 944,312, 220, 874, 958, 795, 240, 932, 320, 999, 913]
    non_selected_classes = set(range(1000)) - set(selected_classes)

    # Load the testing data samples
    y_batch, s_batch, x_batch = prepare_data(path_to_data, selected_classes)

    # Prepare the CSV files that will store the results
    prepare_csv_files(output_folder)

    print("data loading complete")

    for kd_factor in [0, 0.25, 0.5, 0.75, 1.0]:
        models = load_model(path_to_checkpoints, num_models, kd_factor)
        
        for CamVersion in [XGradCAM, AblationCAM, HiResCAM, GradCAM, GradCAMPlusPlus, EigenCAM]:
            for model_index, model in enumerate(models):
                a_batch = create_saliency_maps(CamVersion, model, x_batch, y_batch)
                PointingGame_score, AttributionLocalisation_score, RelevanceRankAccuracy_score, TopKIntersection_Score, Monotonicity_Score, FaithfullnessEstimate_Score, prediction_values  = run_quantus_evaluation_scripts(CamVersion, model, x_batch, y_batch, s_batch, a_batch)
                save_results(output_folder, kd_factor, model_index, CamVersion, PointingGame_score, AttributionLocalisation_score, RelevanceRankAccuracy_score, TopKIntersection_Score, Monotonicity_Score, FaithfullnessEstimate_Score, prediction_values)


if __name__ == "__main__":
    main()
