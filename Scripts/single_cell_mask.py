import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import json
import tifffile as tfi
from matplotlib.widgets import PolygonSelector
from skimage.draw import polygon2mask
from skimage.transform import resize


def _read_stack_ZHW(tif_path):
    arr = tfi.imread(tif_path) # (Z, H, W)
    return arr, arr.dtype

def _save_stack(path, arr, dtype):
    arr_to_save = arr.astype(dtype, copy=False) # numpy float64 transforms to original datatype
    tfi.imwrite(path, arr_to_save, imagej=False)

def _draw_single_polygon_mask(base_img_2d, title="Draw polygon. Press 'Enter' to complete."):
    # return bool mask (H, W) and vertex list [(x, y), ...]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(base_img_2d, cmap="gray")
    ax.set_title(title)
    ax.set_axis_off()
    
    verts_container = {"verts": None}
    def onselect(verts):
        verts_container["verts"] = verts
    selector = PolygonSelector(ax,
                               onselect,
                               useblit=True,
                               props=dict(color='white', linewidth=2, alpha=0.8))
    
    done = {"flag": False}
    def on_key(event):
        if event.key in ("enter", "return"):
            done["flag"] = True
            plt.close(fig)
    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    selector.disconnect_events()

    if not done["flag"] or verts_container["verts"] is None or len(verts_container["verts"]) < 3:
        raise RuntimeError("Unsuccessful. At least 3 vertexes, press 'Enter' to complete.")

    H, W = base_img_2d.shape
    # PolygonSelector returns (x, y) but polygon2mask needs (row, col)=(y, x)
    verts_xy = verts_container["verts"]
    verts_yx = [(y, x) for (x, y) in verts_xy]
    mask = polygon2mask((H, W), np.array(verts_yx, dtype=float))
    return mask.astype(bool), verts_xy

def draw_mask_and_analysis(subfolder):
    # This subfolder contains 1 *recon.tif and 1 *ori.tif
    """
    - Find *recon.tif and *ori.tif in the given subfolder
    - Ask how many masks to draw
    - Interactively draw each mask (press Enter to finish one) on the maximum projection of the recon image in the XY plane
    - Apply each mask to all Z slices of the recon image; then resize it with nearest-neighbor interpolation to match the XY dimensions of the ori image, and apply it to all Z slices of the ori image
    - Save each mask separately as:
        recon_mask_XX.tif, ori_mask_XX.tif (pixels outside the mask set to zero)
        mask2d_recon_XX.tif, mask2d_ori_XX.tif (binary 2D masks)
        mask_XX_vertices.json (polygon vertex coordinates for reproduction)
    """
    print("Processing:", subfolder)
    for f in os.listdir(subfolder):
        if f.lower().endswith(".json"):
            print("This subfolder is already processed.")
            return
    
    candidates = os.listdir(subfolder)
    recon_path = [os.path.join(subfolder, f) for f in candidates if f.lower().endswith("recon.tif")]
    ori_path   = [os.path.join(subfolder, f) for f in candidates if f.lower().endswith("ori.tif")]
    if len(recon_path) != 1 or len(ori_path) != 1:
        raise FileNotFoundError(f"In {subfolder} *recon.tif or *ori.tif not found.")
    
    # (Z, H, W)
    recon, recon_dtype = _read_stack_ZHW(recon_path)   # in our case: (9, 1024, 1024)
    ori,   ori_dtype   = _read_stack_ZHW(ori_path)     # in our case: (135, 512, 512)

    Zr, Hr, Wr = recon.shape
    # print(ori.shape, len(ori.shape))
    if len(ori.shape) != 3:
        print(f"The ori file in this subfolder has incorrect shape: {ori.shape}")
        return
    Zo, Ho, Wo = ori.shape

    # max projection
    recon_mip = recon.max(axis=0)
    print(recon_mip.shape, recon_mip.dtype, recon_mip.min(), recon_mip.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    disp_img = (recon_mip - recon_mip.min()) / (recon_mip.max() - recon_mip.min())
    ax.imshow(disp_img, cmap="gray")
    ax.set_title(subfolder)
    ax.set_axis_off()
    plt.show(block=False)

    try:
        n = int(input("Enter the number of masks to draw:").strip())
        plt.close(fig)
    except Exception:
        raise ValueError("The input should be integer.")
    if n <= 0:
        raise ValueError("Mask number should larger than 1.")

    print(f"Starting to draw {n} masks one by one. Press Enter after finishing each.")

    # output
    def out(name): return os.path.join(subfolder, name)

    for i in range(1, n + 1):
        idx = f"{i:02d}"
        title = f"Draw mask {i}/{n}: Left-click to place polygon points, press Enter to finish."
        mask_r_xy, verts_xy = _draw_single_polygon_mask(recon_mip, title=title)  # (Hr, Wr)

        # broadcast
        mask_r_zhw = np.broadcast_to(mask_r_xy[np.newaxis, ...], (Zr, Hr, Wr))
        recon_masked = recon * mask_r_zhw.astype(recon.dtype)

        _save_stack(out(f"recon_mask_{idx}.tif"), recon_masked, recon_dtype)
        tfi.imwrite(out(f"mask2d_recon_{idx}.tif"), (mask_r_xy.astype(np.uint8) * 255), imagej=False)

        # nearest mapping
        mask_o_xy = resize(
            mask_r_xy.astype(np.uint8),
            (Ho, Wo),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(bool)

        # broadcast
        mask_o_zhw = np.broadcast_to(mask_o_xy[np.newaxis, ...], (Zo, Ho, Wo))
        ori_masked = ori * mask_o_zhw.astype(ori.dtype)

        _save_stack(out(f"ori_mask_{idx}.tif"), ori_masked, ori_dtype)
        tfi.imwrite(out(f"mask2d_ori_{idx}.tif"), (mask_o_xy.astype(np.uint8) * 255), imagej=False)

        with open(out(f"mask_{idx}_vertices.json"), "w", encoding="utf-8") as f:
            json.dump({"vertices_xy": verts_xy}, f, ensure_ascii=False, indent=2)

        print(f"Mask {i} completed: Saved recon_mask_{idx}.tif / ori_mask_{idx}.tif and related files.")


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    data_folder = filedialog.askdirectory(title="Please Select One Unprocessed but Fourier Checked Folder") # eg: 2_Frequency_Domain_Checked/SI_1.518_GFP
    root.destroy()
    
    for subfolder in os.listdir(data_folder):
        draw_mask_and_analysis(os.path.join(data_folder, subfolder))