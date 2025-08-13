import os
import shutil
import numpy as np
import tifffile
import matplotlib.pyplot as plt

class Condensate_Analysis:
    def __init__(self, folder):
        self.folder = folder
        self.subfolder_list = [os.path.join(folder, f) for f in os.listdir(folder)]
        
    def calculate_expression_level(self):
        expression_list = [] # tuple list, every tuple has the form (recon_masked_path, experssion_level)
        return Dict