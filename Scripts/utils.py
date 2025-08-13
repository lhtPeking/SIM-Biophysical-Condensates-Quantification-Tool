import os
import shutil
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import regionprops_table

class Statistical_Analysis:
    @staticmethod
    def Welch_ANOVA_test():
        pass

class Condensate_Analysis:
    def __init__(self, folder):
        self.folder = folder
        self.subfolder_list = [os.path.join(folder, f) for f in os.listdir(folder)]
        
        self.expression_list = None
        self.masked_figure_path_list = None
        
        self.selected_figures = None # list , every element is the path of recon_masked_path
        
    def calculate_expression_level(self):
        expression_list = [] # tuple list, every tuple has the form (recon_masked_path, expression_level)
        masked_figure_path_list = [] # list list, every son list has the form [ori_masked_path, recon_masked_path, 2D_mask_ori_path, 2D_mask_recon_path]
        for subfolder in self.subfolder_list:
            for file in os.listdir(subfolder):
                if file.lower().startswith("ori_mask"):
                    
                    recon_mask_path = None
                    two_d_ori_mask_path = None
                    two_d_recon_mask_path = None
                    
                    suffix = file.split("_")[-1]
                    for f in os.listdir(subfolder):
                        if f.endswith("ori_mask_" + suffix):
                            recon_mask_path = os.path.join(subfolder, f)
                        if f.endswith("mask2d_ori_" + suffix):
                            two_d_ori_mask_path = os.path.join(subfolder, f)
                        if f.endswith("mask2d_recon_" + suffix):
                            two_d_recon_mask_path = os.path.join(subfolder, f)
                    masked_figure_path_list.append([os.path.join(subfolder, file), recon_mask_path, two_d_ori_mask_path, two_d_recon_mask_path])
                    
                    if None in (recon_mask_path, two_d_ori_mask_path, two_d_recon_mask_path):
                        raise FileNotFoundError(f"Missing required mask files for {file}")
                    
        # calculate expression level using ori_masked_path
        for cell in masked_figure_path_list:
            ori_image = tiff.imread(cell[0])
            ori_mask = tiff.imread(cell[2])
            
            if ori_mask.shape != ori_image.shape[1:]:
                raise ValueError(f"Image shape of ori image {ori_image.shape} != Mask shape {ori_mask.shape}")
            
            mask_bool = (ori_mask == 255)
            Z = ori_image.shape[0]
            mask_bool_3d = np.broadcast_to(mask_bool, (Z,) + mask_bool.shape)
            selected_pixels = ori_image[mask_bool_3d]
            mean_value = np.mean(selected_pixels)
            
            expression_list.append((cell[1], mean_value))
            
        self.expression_list = expression_list
        self.masked_figure_path_list = masked_figure_path_lists
            
        return expression_list, masked_figure_path_list
    
    def calculate_condensate_property(self, percentile=0.01, identify_parameter=None):
        all_results = []

        for tiff_path in self.selected_figures:
            image = tiff.imread(tiff_path)

            # Thresholding based on given percentile
            non_zero_pixels = image[image > 0]
            threshold = np.percentile(non_zero_pixels, (1 - percentile) * 100)
            mask = image >= threshold
            masked_image = np.where(mask, image, 0)

            # Connected component labeling (3D, 26-connectivity)
            structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
            labels, num = ndimage.label(masked_image > 0, structure=structure)

            # Extract particle properties (geometry + intensity)
            props = regionprops_table(
                labels,
                intensity_image=image,  # Use original image for intensity measurements
                properties=[
                    "label",
                    "area",                  # Number of voxels
                    "bbox",                  # Bounding box
                    "centroid",              # Centroid coordinates
                    "equivalent_diameter",   # Diameter of a sphere with same volume
                    "extent",                # Volume fraction within bounding box
                    "max_intensity",
                    "mean_intensity",
                    "min_intensity",
                    "solidity",              # Compactness (relative to convex hull)
                    "eccentricity",          # Shape elongation (0 = sphere)
                    "orientation",           # Main axis orientation (radians)
                    "major_axis_length",
                    "minor_axis_length"
                ]
            )

            df = pd.DataFrame(props)

            # Convert to real-world units if voxel size is provided
            if identify_parameter and "voxel_size" in identify_parameter:
                voxel_size = identify_parameter["voxel_size"]  # e.g. (z, y, x) in μm
                df["volume_um3"] = df["area"] * np.prod(voxel_size)
                df["equivalent_diameter_um"] = df["equivalent_diameter"] * voxel_size[0]  # assuming isotropic
            else:
                df["volume_um3"] = np.nan
                df["equivalent_diameter_um"] = np.nan

            # Store file name for reference
            df["file_name"] = tiff_path
            all_results.append(df)

        # Merge results from all files
        final_df = pd.concat(all_results, ignore_index=True)

        return final_df
    
    def visualization(self):
        pass