
!pip install deepface;
!pip install opencv-python;
from deepface import DeepFace
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import json
import itertools
from pathlib import Path

#home = str(Path.home())
#print("HOME_FOLDER is ",home)

#If there is an error message like facial_expression_model_weights.h5 will be downloaded...
'''Access denied with the following error:

    Too many users have viewed or downloaded this file recently. Please
    try accessing the file again later. If the file you are trying to
    access is particularly large or is shared with many people, it may
    take up to 24 hours to be able to view or download the file. If you
    still can't access a file after 24 hours, contact your domain
    administrator.'''

#The solution 

'''Pre-trained weights are in Google Drive and it seems limit is exceeded. However, you can still download the pre-trained weight file manually from the link specified in the console log.

The URL will be specified in the console log: https://drive.google.com/uc?id=13iUHHP3SlNg53qSuQZDdHDSDNdBP9nwy

Then, you should copy the pre-trained weight to HOME_FOLDER/.deepface/weights folder whereas your HOME_FOLDER can be found as shown below. 
Then, deepface will not download the pre-trained weight anymore. 
BTW, if the drive url downloads a zip, then you should unzip it here'''

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

imnames= sys.argv[1]
annotate(imnames)