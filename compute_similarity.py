# Load Libraries
import cv2
import numpy as np
import itertools

# Function to compute similarity score
def compute_similarity(image1_path, image2_path, blur_kernel=(5, 5), blur_sigma=1.0):

    # Read images and convert to grayscale
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise
    img1 = cv2.GaussianBlur(img1, blur_kernel, blur_sigma)
    img2 = cv2.GaussianBlur(img2, blur_kernel, blur_sigma)

    # Initialize ORB detector
    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31
    )

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Handle cases with no features detected
    if des1 is None or des2 is None:
        return 0.0

    # Feature matching with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Calculate normalized score
    max_possible = min(len(des1), len(des2))
    raw_score = len(matches)
    norm_score = raw_score / max_possible if max_possible > 0 else 0.0

    return np.clip(norm_score, 0.0, 1.0)

# Load images
image_files = [f'{i}.png' for i in range(1, 7)]
for i, (img1, img2) in enumerate(itertools.combinations(image_files, 2), start=1):
    similarity_score = compute_similarity(img1, img2, blur_kernel=(13, 13), blur_sigma=7)
    print(f'Pair {i}: {img1} and {img2} - Similarity score: {similarity_score}')