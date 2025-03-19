from pathlib import Path
import numpy as np
import menpo.io as mio
from menpo.visualize import print_progress
from menpofit.aam import HolisticAAM
from menpo.feature import normalize
import os
import dlib
import cv2
from menpo.io import export_landmark_file
from menpo.shape import PointCloud

# Paths
landmark_model_path = "/workspace/shared_with_host/shape_predictor_68_face_landmarks.dat"
image_folder = "/home/jovyan/data/celeba/samples"  
landmarks_folder = os.path.join(image_folder, "landmarks")  
os.makedirs(landmarks_folder, exist_ok=True)

# Load dlib face detector & shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_model_path)


for img_name in os.listdir(image_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape  

        faces = detector(gray)
        if len(faces) == 0:
            print(f"No face detected in {img_name}")
            continue

        
        landmarks = predictor(gray, faces[0])
        pts = np.array([[p.y, w - p.x] for p in landmarks.parts()], dtype=np.float32)  # Rotate 90 CCW

        # Save as Menpo .pts format
        pointcloud = PointCloud(pts)
        pts_filename = os.path.splitext(img_name)[0] + ".pts"
        pts_path = os.path.join(landmarks_folder, pts_filename)
        export_landmark_file(pointcloud, pts_path, extension=".pts")

        print(f"Saved landmarks for {img_name} as {pts_path}")


path_to_lfpw = Path('/home/jovyan/data/celeba/samples')
landmarks_folder = path_to_lfpw / "landmarks"


training_images = []
for image in mio.import_images(path_to_lfpw, verbose=True):
    img_name = image.path.stem
    pts_path = landmarks_folder / f"{img_name}.pts"

    if pts_path.exists():
        landmarks = mio.import_landmark_file(pts_path)
        image.landmarks['PTS'] = landmarks['PTS']  
        training_images.append(image)

print(f"Loaded {len(training_images)} images with landmarks.")

# Train AAM
aam = HolisticAAM(
    training_images,
    group='PTS',
    verbose=True,
    holistic_features=normalize,
    diagonal=120,
    scales=(0.5, 1.0),
    max_shape_components=None,
    max_appearance_components=None
)

import pickle


with open("/home/jovyan/trained_aam.pkl", "wb") as f:
    pickle.dump(aam, f)

print("AAM model saved successfully!")
