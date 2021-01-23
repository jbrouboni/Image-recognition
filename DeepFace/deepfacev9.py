import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import json
import itertools
from deepface import DeepFace
from pathlib import Path

#home = str(Path.home())
#print("HOME_FOLDER is ",home)

detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt" , "res10_300x300_ssd_iter_140000.caffemodel")

def annotate_image(image_name):
  image = cv2.imread(image_name)
  base_img = image.copy()
  original_size = base_img.shape
  target_size = (300, 300)
  image = cv2.resize(image, target_size)
  aspect_ratio_x = (original_size[1] / target_size[1])
  aspect_ratio_y = (original_size[0] / target_size[0])
  imageBlob = cv2.dnn.blobFromImage(image = image)
  detector.setInput(imageBlob)
  detections = detector.forward()
  column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
  detections_df = pd.DataFrame(detections[0][0], columns = column_labels)
  detections_df = detections_df[detections_df['is_face'] == 1]
  detections_df = detections_df[detections_df['confidence']>0.90]
  detections_df['left'] = (detections_df['left'] * 300).astype(int)
  detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
  detections_df['right'] = (detections_df['right'] * 300).astype(int)
  detections_df['top'] = (detections_df['top'] * 300).astype(int)
  y_list=list()         
  for i, instance in detections_df.iterrows():
    confidence_score = str(round(100*instance["confidence"], 2))+" %"
    left = instance["left"]; right = instance["right"]
    bottom = instance["bottom"]; top = instance["top"]
    detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y) ,
    int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
    box_coords=[left,right,bottom,top]
    obj = DeepFace.analyze(image_name,actions = ['gender', 'race']);
    x = {"Name":image_name,
        "label_name": obj["dominant_race"]+" " +obj["gender"],
        "score": confidence_score,
        "bounding_box":box_coords
          }
    #Conversion en JSON:
    y = json.dumps(x)
    y_list.append(y)
  return y_list

def annotate(imnames):
  file = open(imnames, "r") # Ouvrir le fichier en lecture seule
  lines = file.readlines()
  file.close()
  # Itérer sur les lignes
  fichier = open("DataDeepFace.txt", "w")
  for line in lines:
    if line == lines[-1]:
      photo_filename = line
    else:
      photo_filename = line[:-1]
    print(photo_filename)
    data=annotate_image(photo_filename)
    for value in data:
        fichier.write("\n"+value)
  fichier.close()

#imnames="imnames.txt"
imnames= sys.argv[1]
annotate(imnames)