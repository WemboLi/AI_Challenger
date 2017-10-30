import json
from pprint import pprint
import sys
import pandas as pd
import os
import cv2

# csv_path = "/media/wenbo/20161109_21/DL/AI-Challenger/scene_classification/data/ai_challenger_scene_train_20170904/scene_classes.csv"

# img_path = "/media/wenbo/20161109_21/DL/AI-Challenger/scene_classification/data/ai_challenger_scene_train_20170904/scene_train_images_20170904" 


if len(sys.argv) < 5:
  sys.exit('Usage: %s json-name out-name image-dir' % sys.argv[0])

json_file = sys.argv[1]
out_file = sys.argv[2]
csv_path = sys.argv[3]
img_path = sys.argv[4]


df = pd.read_csv(csv_path, header = None, usecols = [0, 2])

print df

# create folder for each class
base_dir = os.path.dirname(csv_path)
# base_dir = 

classes_folders = [None] * df.shape[0]

for i in range(df.shape[0]):
   folder = os.path.join(base_dir, str(df.iloc[i, 1]).replace("/", "_"))

   if not os.path.isdir(folder):
      print "creating folder %s" % folder
      os.mkdir(folder)
   classes_folders[df.iloc[i, 0]] = folder

# print classes_folders

with open(json_file, 'r') as in_file:
  data = json.load(in_file)

# pprint(data["image_id"])

fp = open(out_file, 'w')

for i in range(len(data)):
   image_id = data[i]['image_id']
   label_id = int(data[i]['label_id'])
   fp.write('{} {}\n'.format(image_id, label_id))
   
   rImage_path = os.path.join(img_path, image_id)
   print 'Reading image {}'.format(rImage_path)
   image_data = cv2.imread(rImage_path)
   
   cv2.imwrite( os.path.join(classes_folders[label_id], image_id), image_data )
      
