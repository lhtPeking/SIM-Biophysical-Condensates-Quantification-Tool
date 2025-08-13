import os
import shutil
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

def _to_2d_grayscale(arr):
    a = np.asarray(arr)
    if a.ndim == 2:
        return a.astype(np.float64)
    elif a.ndim == 3:
        return np.mean(a, axis=0).astype(np.float64)
    else:
        raise TypeError(f"Invaid figure shape. Program only receive 2/3 dim figure, now get dim={a.dim}")

def _fft_magnitude(img2d):
    img = np.nan_to_num(img2d, nan=0.0, posinf=0.0, neginf=0.0)
    img = img - np.mean(img)
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    mag = np.abs(Fshift)
    mag = np.log1p(mag)
    
    # normalize
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)
    return mag

def FFT_check(files, original_data_folder, dest_folder):
    for name in files: # name is actually path
        print(f'Processing: {name}')
        try:
            img = tifffile.imread(name)
            img2d = _to_2d_grayscale(img)
            fft_mag = _fft_magnitude(img2d)

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.title('Reconstructed Figure')
            plt.imshow(img2d, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('FFT Spectrum (Linear)')
            plt.imshow(fft_mag, cmap='gray')
            plt.axis('off')

            plt.suptitle(os.path.basename(name))
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

            while True:
                respond = input("Move to destination folder? [Y/N, Q to quit]: ").strip().lower()
                if respond == 'y':
                    file_name = os.path.basename(name)
                    filename_without_ext = os.path.splitext(file_name)[0]
                    reconstructed_file_name = f"{filename_without_ext}_recon.tif"
                    
                    target_folder_path = os.path.join(dest_folder, filename_without_ext)
                    target_path = os.path.join(target_folder_path, reconstructed_file_name)
                    
                    os.makedirs(target_folder_path, exist_ok=True)
                    
                    # counter = 1
                    # while os.path.exists(target_path):
                    #     target_path = f"{target_path}_{counter}"
                    #     counter += 1
                    
                    shutil.move(name, target_path)
                    
                    match_sequence = filename_without_ext
                    print("match_sequence:", match_sequence)
                    
                    for f in os.listdir(original_data_folder):
                        # print("Base name of f:", os.path.basename(f))
                        file_path = os.path.join(original_data_folder, f)
                        
                        ori_filename_without_ext = os.path.splitext(os.path.basename(f))[0]
                        
                        if ori_filename_without_ext == match_sequence:
                            original_file_name = f"{filename_without_ext}_ori.tif"
                            target_path_origin = os.path.join(target_folder_path, original_file_name)
                            # print("Move Original File to:", target_path_origin)
                            shutil.move(file_path, target_path_origin)
                            print("Original File Moved.")
                            
                    print(f"Moved to: {target_folder_path}")
                    break
                elif respond == 'n':
                    print("Skipped moving.")
                    break
                elif respond == 'q':
                    print("Quit signal received.")
                    plt.close('all')
                    return
                else:
                    print("Invalid input. Please enter Y / N / Q.")

        except Exception as e:
            print(f"[Error] Failed to process {name}: {e}")
        finally:
            plt.close('all')

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    original_data_folder = filedialog.askdirectory(title="Please Select the Original Figures' Folder") # Example: GFP Prox1_tif
    root.destroy()
    
    # original_file_paths = [os.path.join(original_data_folder, f)
    #               for f in os.listdir(original_data_folder)
    #               if f.lower().endswith('.tif')]
    
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Please Select the Reconstructed SIM Figures' Folder") # Example: SI_1.518_GFP_Prox1
    root.destroy()
    
    file_paths = [os.path.join(folder_path, f)
                  for f in os.listdir(folder_path)
                  if f.lower().endswith('.tif')]
    
    root = Tk()
    root.withdraw()
    dest_folder = filedialog.askdirectory(title="Please Select the Storage Folder")
    root.destroy()
    
    dest_subfolder = os.path.join(dest_folder, os.path.basename(folder_path))
    os.makedirs(dest_subfolder, exist_ok=True)
    
    FFT_check(file_paths, original_data_folder, dest_subfolder)