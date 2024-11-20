import os
import cv2
import copy
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_images(input_files_dir: str) -> list:
    filenames = [path for path in glob.glob(f"{input_files_dir}/*jpg")] # Extract the paths of the training
    return [cv2.imread(filename) for filename in filenames]

def show_image(img: np.array) -> None:  
    cv2.imshow("image", img)
    cv2.waitKey(0) # Screen time
    cv2.destroyAllWindows()
    
def write_image(output_files_dir, save_name ,counter_image, img):
    cv2.imwrite(f"{output_files_dir}/{save_name}_{counter_image}.jpg",img)

def get_chessboard_points(chessboard_shape, dx, dy):
    """
    Calculates the 3D coordinates of the chessboard corners.

    Args
    ----
    - chessboard_shape: tuple with the number of inner corners in the chessboard (columns, rows).
    - dx: distance between corners in the x-axis.
    - dy: distance between corners in the y-axis.
    
    Returns
    ----
    - points: numpy array with the 3D coordinates of the chessboard corners.
    """
    
    points = []
    
    for i in range(chessboard_shape[1]):
        for j in range(chessboard_shape[0]):
            points.append(np.array([i*dy, j*dx, 0]))
            
    return np.array(points, dtype = np.float32)

def get_camera_parameters(input_dir:str, pattern_size: tuple, dx:int, dy:int, save_images = False, show_results = False, fine_adjust = True, fisheye = False, output_dir = "output_images", save_name = "detected", image_number = -1) -> tuple | None:
    """
    Estimates the camera parameters (intrinsics, extrinsics, distortion coefficients) using a set of images of a chessboard pattern.

    Args
    ----
    - input_dir: path to the directory containing the images of the chessboard pattern.
    - pattern_size: tuple with the number of inner corners in the chessboard (columns, rows).
    - dx: x length of the corners in the chessboard.
    - dy: y length of the corners in the chessboard.
    - save_images: boolean indicating whether the images with the detected corners should be saved.
    - show_results: boolean indicating whether the results should be printed.
    - fine_adjust: boolean indicating whether the corners should be refined.
    - fisheye: boolean indicating whether the fisheye model should be used.
    - output_files_dir: path to the directory where the images with the detected corners should be saved.
    - save_name: name of the images with the detected corners.
    - image_number: number of images to be used for the calibration. If -1, all images are used.
    
    Returns
    ----
    - rms: root mean squared reprojection error.
    - intrinsics: camera matrix.
    - extrinsics: list with the rotation and translation vectors for each image.
    - distortion: distortion coefficients.
    - rotations: list with the rotation matrices for each image.
    - translations: list with the translation vectors for each image."""

    imgs = load_images(input_dir) # Load the images as matrixes of pixeles

    corners = [cv2.findChessboardCorners(img, pattern_size) for img in imgs] # Find the corners of the chessboard in the images [True/False, [corner_points]]
    
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    
    if fine_adjust:
        corners_copy = copy.deepcopy(corners)

        # To refine corner detections it is needed the function cv2.cornerSubPix which requires the images in grayscale
        imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

        # (5, 5) is the size of the window to search for refined corners, (-1, -1) allows to value all points in the window
        refined_corners = [cv2.cornerSubPix(i, cor[1], (5, 5), (-1, -1), subpix_criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)] 
        
    else:
        refined_corners = [cor[1] if cor[0] else [] for cor in corners]
        
    corners = [cor for cor in corners if cor[0]] # Delete the images where no info was extracted

    cleaned_refined_corners = []
    undefined_images = []

    for i in range (len(refined_corners)): # Delete the images where no info was extracted
        corner = refined_corners[i]
        
        if len(corner) != 0:
            cleaned_refined_corners.append(corner)
            
        else:
            undefined_images.append(i)

    for i in sorted(undefined_images, reverse=True):
        imgs.pop(i) # In case there are many wrong images, adjust the deleted indexes

    if image_number > 0 and 0 <= len(cleaned_refined_corners) - image_number: # Adjust the number of images to be used
        for _ in range(len(cleaned_refined_corners) - image_number):
            cleaned_refined_corners.pop(-1)
            imgs.pop(-1)
            
    elif image_number == 0:
        print("A camera can not be adjusted with less than 1 image.")
        
        return None
        
    elif image_number != -1:
        print("The number of images to be used is greater than the number of images available.")
        
        return None
            
    refined_corners = cleaned_refined_corners
      
    if save_images:
        imgs_copy = copy.deepcopy(imgs)
        _ = [cv2.drawChessboardCorners( imgs_copy[i], pattern_size, refined_corners[i], corners[i][0])  for i in range(len(refined_corners)) if corners[i][0]]# draws on imgs_copy
        os.makedirs(f"{output_dir}", exist_ok = True)
        
        for i in range(len(imgs_copy)):
            write_image(output_dir, save_name, i, imgs_copy[i]) 
        
    chessboard_points = [get_chessboard_points(pattern_size, dx, dy) for _ in range(len(refined_corners))] # As many chessboard points as images with detected corners

    if not fisheye:
        rms, intrinsics, distortion, rotations, traslations = cv2.calibrateCamera(chessboard_points, refined_corners, imgs[-1].shape[::-1][1:], None, None)
                        
    else: # Fisheye calibration
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW # +cv2.fisheye.CALIB_CHECK_COND
        intrinsics = np.zeros((3, 3))
        distortion = np.zeros((4, 1))
        rotations = [np.zeros((1, 1, 3), dtype=np.float64) for _ in refined_corners]
        traslations = [np.zeros((1, 1, 3), dtype=np.float64) for _ in refined_corners]
        
        rms, _, _, _, _ = cv2.fisheye.calibrate(chessboard_points, refined_corners, imgs[-1].shape[::-1][1:], intrinsics, distortion, rotations, traslations, calibration_flags, subpix_criteria)
    
    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec.reshape(-1, 1))), rotations, traslations))

    if show_results:
        print("Root mean squared reprojection error:\n", rms)
        print("Intrinsics:\n", intrinsics)
        print("Extrinsics:\n", extrinsics)
        print("Distortion coefficients:\n", distortion)

            
    return rms, intrinsics, extrinsics, distortion

def correct_fisheye(input_dir:str, intrinsics:np.array, distortion:np.array, output_dir = "output", save_name = "corrected") -> None:
    """
    Corrects the distortion of a set of images using the fisheye model.
    
    Args
    ----
    - input_dir: path to the directory containing the images to be corrected.
    - intrinsics: camera matrix.
    - distortion: distortion coefficients.
    - output_dir: path to the directory where the corrected images should be saved.
    - save_name: name of the corrected images."""
    
    imgs = load_images(input_dir) # Load the images as matrixes of pixeles
    dim = imgs[0].shape[:2][::-1] # Dim  of each image
    
    # Create the undistortion map
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(intrinsics, distortion, np.eye(3), intrinsics, dim, cv2.CV_16SC2)
    
    # Apply the undistortion map to each image. Enlarge picture: INTER_LINEAR, INTER_CUBIC. Shrink picture: INTER_AREA
    corrected_imgs = [cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC) for img in imgs]

    os.makedirs(f"{output_dir}", exist_ok = True)
    
    for i in range(len(corrected_imgs)):
        write_image(output_dir, save_name, i, corrected_imgs[i]) 
        
if __name__ == "__main__":
    
    output_folder = "processed_data/calibration"
    
    _, intrinsics, _, distortion = get_camera_parameters("raw_data", (6,9), 34, 34, True, True, True, False, output_folder , "calibrated")
    
    df = [intrinsic.tolist() for intrinsic in intrinsics]
    
    df = pd.DataFrame(df, columns = ["A", "B", "C"])
    
    df.to_csv(f"{output_folder}/intrinsics.csv", index = False)
    
    
    
