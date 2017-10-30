import json
from pprint import pprint
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
  sys.exit('Usage: %s json-name out-name' % sys.argv[0])

json_file = sys.argv[1]
out_file = sys.argv[2]

with open(json_file, 'r') as in_file:
  data = json.load(in_file)

# pprint(data["image_id"])

stats = np.zeros(80, dtype = np.int)

for i in range(len(data)):
      # fp.write('{} {}\n'.format(data[i]['image_id'], data[i]['label_id']))
     stats[int(data[i]['label_id'])] += 1


# draw a bar graph of the class item number
fig, ax = plt.subplots()
index = np.arange(80)
bar_width = 0.15

rects1 = plt.bar(index, stats, bar_width, color = 'b', label = '#scenes')


plt.xlabel('scene index')
plt.ylabel('scene_number') 
plt.legend()

plt.show()

# write into a txt file
with open(out_file, 'w') as fp:
   for i in range(stats.shape[0]):
       fp.write('{} {}\n'.format(i, stats[i]))

