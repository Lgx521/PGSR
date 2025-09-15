## How to Use

1.  **Save the script**: Save the code above as a file named `convert_from_json.py`.

2.  **Prerequisites**:

      * You need `numpy`, `Pillow`, and `scipy` installed (`pip install numpy Pillow scipy`).
      * COLMAP and ImageMagick must be installed and accessible in your system's PATH, or you must provide the full path to their executables.

3.  **Directory Structure**: Your data for a scene should be organized as follows. The script assumes your source images are in a subdirectory named `input`.

    ```
    /path/to/your/scene/
    ├── input/
    │   ├── r_0.png
    │   ├── r_1.png
    │   └── ...
    └── transforms.json
    ```

4.  **Run the script**: Execute the script from your terminal, pointing to your scene directory and the `transforms.json` file.

    ```bash
    python scripts/preprocess/convert_with_tfs.py --data_path /home/sgan/PGSR/data/antique_gt_pose_10/scene --transforms_json /home/sgan/PGSR/data/antique_gt_pose_10/scene/transforms.json --resize
    ```

      * Add the `--resize` flag if you want the downsampled `images_2`, `images_4`, and `images_8` folders.
      * Use `--colmap_executable /path/to/colmap` if it's not in your PATH.
      * Use `--magick_executable /path/to/mogrify` if it's not in your PATH.

After running, your scene directory will be populated with the `database.db` file, the `sparse` directory containing the model generated from your JSON, and the `images` folder, ready for training.


### Training
```bash
(pgsr) sgan@sgan-ubuntu:~/PGSR$ python train.py -s ./data/antique_gt_pose/scene_4 -m .data/all_output/imgs_4_test --max_abs_split_points 0 --opacity_cull_threshold 0.05
```