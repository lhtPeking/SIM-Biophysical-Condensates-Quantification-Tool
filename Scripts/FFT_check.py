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

def FFT_check(files, dest_folder):
    for name in files:
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
                    target_path = os.path.join(dest_folder, os.path.basename(name))
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = f"{target_path}_{counter}"
                        counter += 1
                    shutil.move(name, target_path)
                    print(f"Moved to: {target_path}")
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
    folder_path = filedialog.askdirectory(title="Please Select the Reconstructed SIM Figures' Folder")
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
    
    FFT_check(file_paths, dest_subfolder)