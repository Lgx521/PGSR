import os
import json
import shutil
import logging
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
import math

def get_image_size(image_path):
    """Reads image dimensions without loading the full image."""
    with Image.open(image_path) as img:
        return img.size

def convert_nerf_pose_to_colmap(c2w_nerf):
    """
    Converts a NeRF camera-to-world matrix to a COLMAP world-to-camera
    quaternion and translation vector.

    Args:
        c2w_nerf (np.ndarray): A 4x4 numpy array representing the camera-to-world
                               transform matrix in NeRF's coordinate system.

    Returns:
        tuple: A tuple containing:
            - q (list): A list [w, x, y, z] representing the world-to-camera rotation quaternion.
            - t (np.ndarray): A numpy array [x, y, z] representing the world-to-camera translation vector.
    """
    # NeRF uses convention [right, up, backwards], COLMAP uses [right, down, forwards]
    # We need to apply a rotation to the camera coordinate system.
    # This is equivalent to post-multiplying the c2w matrix by a conversion matrix.
    c2w_colmap = c2w_nerf @ np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    # Invert to get world-to-camera (extrinsics)
    w2c_colmap = np.linalg.inv(c2w_colmap)

    # Extract rotation matrix and translation vector
    R = w2c_colmap[:3, :3]
    t = w2c_colmap[:3, 3]

    # Convert rotation matrix to quaternion (SciPy returns [x, y, z, w])
    q_xyzw = Rotation.from_matrix(R).as_quat()

    # Rearrange to COLMAP's [w, x, y, z] format
    q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    return q_wxyz, t


def init_colmap_from_json(args):
    """
    Initializes a COLMAP project using camera poses from a transforms.json file.
    """
    # Define executable paths
    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
    magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
    use_gpu = 1 if not args.no_gpu else 0

    scene_path = args.data_path
    json_path = args.transforms_json
    image_dir_name = "input" # Assumes images are in an 'input' subfolder of the scene
    image_path = os.path.join(scene_path, image_dir_name)

    print(f"Processing scene: {scene_path}")

    # --- 1. Load transforms.json ---
    with open(json_path, 'r') as f:
        transforms = json.load(f)

    # --- 2. Clean up previous runs ---
    print("Cleaning up old files...")
    os.makedirs(os.path.join(scene_path, "sparse"), exist_ok=True)
    os.system(f"rm -rf {scene_path}/images/*")
    os.system(f"rm -rf {scene_path}/sparse/*")
    if os.path.exists(os.path.join(scene_path, 'database.db')):
        os.remove(os.path.join(scene_path, 'database.db'))

    # --- 3. Create text-based COLMAP model ---
    print("Creating COLMAP model from transforms.json...")
    sparse_txt_path = os.path.join(scene_path, "sparse_txt")
    os.makedirs(sparse_txt_path, exist_ok=True)

    # Find all image files and sort them to ensure consistent ordering
    image_files = sorted([f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        logging.error(f"No images found in {image_path}. Exiting.")
        exit(1)

    # Get image dimensions from the first image
    width, height = get_image_size(os.path.join(image_path, image_files[0]))

    # a) Create cameras.txt
    camera_angle_x = transforms['camera_angle_x']
    focal_length = 0.5 * width / math.tan(0.5 * camera_angle_x)

    with open(os.path.join(sparse_txt_path, "cameras.txt"), "w") as f:
        # Using PINHOLE model: CAMERA_ID, MODEL, WIDTH, HEIGHT, FX, FY, CX, CY
        f.write(f"1 PINHOLE {width} {height} {focal_length} {focal_length} {width/2} {height/2}\n")

    # b) Create images.txt
    # Map file_path from JSON to actual image filenames
    frame_map = {frame['file_path']: frame for frame in transforms['frames']}

    with open(os.path.join(sparse_txt_path, "images.txt"), "w") as f:
        for i, image_name in enumerate(image_files):
            image_id = i + 1
            file_path_key = os.path.splitext(image_name)[0]

            if file_path_key in frame_map:
                c2w = np.array(frame_map[file_path_key]['transform_matrix'])
                q, t = convert_nerf_pose_to_colmap(c2w)

                # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                f.write(f"{image_id} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {image_name}\n\n")
            else:
                print(f"Warning: No transform found for image {image_name}")

    # c) Create empty points3D.txt
    with open(os.path.join(sparse_txt_path, "points3D.txt"), "w") as f:
        pass # Empty file

    # --- 4. Convert text model to binary ---
    print("Converting text model to binary...")
    model_converter_cmd = (
        f"{colmap_command} model_converter "
        f"--input_path {sparse_txt_path} "
        f"--output_path {os.path.join(scene_path, 'sparse')} "
        f"--output_type BIN"
    )
    exit_code = os.system(model_converter_cmd)
    if exit_code != 0:
        logging.error(f"Model conversion failed with code {exit_code}. Exiting.")
        shutil.rmtree(sparse_txt_path)
        exit(exit_code)
    shutil.rmtree(sparse_txt_path)


    # --- 5. Feature extraction and matching (required for triangulation) ---
    print("Running feature extraction...")
    feat_extraction_cmd = (
        f"{colmap_command} feature_extractor "
        f"--database_path {os.path.join(scene_path, 'database.db')} "
        f"--image_path {image_path} "
        f"--ImageReader.camera_model PINHOLE "
        f"--ImageReader.single_camera 1 "
        f"--SiftExtraction.use_gpu {use_gpu}"
    )
    exit_code = os.system(feat_extraction_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    print("Running feature matching...")
    feat_matching_cmd = (
        f"{colmap_command} exhaustive_matcher "
        f"--database_path {os.path.join(scene_path, 'database.db')} "
        f"--SiftMatching.use_gpu {use_gpu}"
    )
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # --- 6. Triangulate points using known poses ---
    print("Triangulating points...")
    point_triangulator_cmd = (
        f"{colmap_command} point_triangulator "
        f"--database_path {os.path.join(scene_path, 'database.db')} "
        f"--image_path {image_path} "
        f"--input_path {os.path.join(scene_path, 'sparse')} "
        f"--output_path {os.path.join(scene_path, 'sparse')}"
    )
    exit_code = os.system(point_triangulator_cmd)
    if exit_code != 0:
        logging.error(f"Point triangulation failed with code {exit_code}. Exiting.")
        exit(exit_code)
        
    # --- 7. Image undistortion (produces the 'images' folder) ---
    # This step is crucial as it normalizes the images according to the created pinhole model.
    print("Undistorting images...")
    img_undist_cmd = (
        f"{colmap_command} image_undistorter "
        f"--image_path {image_path} "
        f"--input_path {os.path.join(scene_path, 'sparse')} "
        f"--output_path {scene_path} "
        f"--output_type COLMAP"
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
        exit(exit_code)


    # --- 8. Optional Resizing ---
    if args.resize:
        print("Copying and resizing images...")
        distorted_images_path = os.path.join(scene_path, "images")
        
        for factor in [2, 4, 8]:
            out_dir = os.path.join(scene_path, f"images_{factor}")
            os.makedirs(out_dir, exist_ok=True)
            
            for file in os.listdir(distorted_images_path):
                source_file = os.path.join(distorted_images_path, file)
                destination_file = os.path.join(out_dir, file)
                shutil.copy2(source_file, destination_file)
                
                resize_percentage = 100 / factor
                resize_cmd = f"{magick_command} mogrify -resize {resize_percentage}% {destination_file}"
                exit_code = os.system(resize_cmd)
                if exit_code != 0:
                    logging.error(f"Resize to {resize_percentage}% failed for {file} with code {exit_code}. Exiting.")
                    exit(exit_code)

if __name__ == '__main__':
    parser = ArgumentParser("COLMAP converter from transforms.json")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the scene directory (e.g., ./data/my_scene)')
    parser.add_argument('--transforms_json', type=str, required=True, help='Path to the transforms.json file')
    parser.add_argument("--no_gpu", action='store_true', help='Disable GPU for SIFT extraction/matching')
    parser.add_argument("--colmap_executable", default="", type=str, help='Path to the COLMAP executable')
    parser.add_argument("--resize", action="store_true", help='Resize images after processing')
    parser.add_argument("--magick_executable", default="", type=str, help='Path to the ImageMagick mogrify executable')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    init_colmap_from_json(args)

    print("Done.")