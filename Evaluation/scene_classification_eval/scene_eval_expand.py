import sys
import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scene_eval import __load_data


def __eval_result_extend(submit_dict, ref_dict, miss_file):
   right_count = 0
   
   log_file1 = open(miss_file, 'w')
   y_true = []
   y_pred = []

   result = {}
   for(key, value) in ref_dict.items():
       if key not in set(submit_dict.keys()):
           result['warning'].append('lacking image %s in your submission file \n' % key)
           print('warnning: lacking image %s in your submission file' % key)
           continue
        
       if value in submit_dict[key][:3]:
           right_count += 1
       else:
           print 'wrong classify: {}->{},{},{}\n'.format(value, submit_dict[key][0], submit_dict[key][1], submit_dict[key][2])
           log_file1.write('{}->{},{},{}\n'.format(value, submit_dict[key][0], submit_dict[key][1], submit_dict[key][2]))
       
       y_true.append(value)
       y_pred.append(submit_dict[key][0])

   result['score'] = str(float(right_count)/max(len(ref_dict), 1e-5))
   
   print result 
   return y_true, y_pred, result
    
def classifaction_report_csv(report, csv_file):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(csv_file)


if __name__ == '__main__':
  
 # sys.path.insert(0, data_root)
  if(len(sys.argv) < 5):
     print 'usage python classification_json.py deploy.prototxt .caffemodel .npy json_file image_folder'
     sys.exit(-1)

  json_ref = sys.argv[1]
  json_submit = sys.argv[2]
  miss_file = sys.argv[3]
  report_file = sys.argv[4]

  submit_dict, ref_dict = __load_data(json_submit, json_ref)
# prin the report of classification
  y_true, y_pred, result = __eval_result_extend(submit_dict, ref_dict, miss_file)
  labels = [i for i in range(80)]
  targets = [str(i) for i in labels]
    
  report = classification_report(y_true, y_pred, target_names = targets)
   
  print report 
  classifaction_report_csv(report, report_file)
