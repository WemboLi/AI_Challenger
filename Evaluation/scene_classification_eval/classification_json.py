import sys
import caffe
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data_root = "/data/scene_classification/"

# sys.path.insert(0, data_root)
if(len(sys.argv) < 5):
  print 'usage python classification_json.py deploy.prototxt .caffemodel .npy json_file image_folder'
  sys.exit()

model_def = sys.argv[1]
model_weights = sys.argv[2]
mean_file = sys.argv[3]
json_file = sys.argv[4]
folder_path = sys.argv[5]

caffe.set_mode_cpu()

net = caffe.Net(model_def, model_weights, caffe.TEST)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(mean_file)
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 224x224


y_true = []
y_pred = [] 

files = []
with open(json_file, 'r') as fp:
    data = json.load(fp)
    for i in range(len(data)):
        y_true.append(int(data[i]['label_id']))
        files.append(os.path.join(folder_path, data[i]['image_id']))
#json_lines = []
#files.extend(sorted(glob.glob(image_dir + '/*.jpg')))
#files.extend(sorted(glob.glob(image_dir + '/*JPEG')))


for i in range(len(files)):
    dict_pred = {}
 #  image_id = data[i]['image_id']
    image = caffe.io.load_image(files[i])
    transformed_image = transformer.preprocess('data', image)

# copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image
### perform classification
    output = net.forward()
    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
    print 'predicted class is:', output_prob.argmax(), 'true class is:', y_true[i]
# sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:3]  # reverse sort and take five largest items
    #print 'probabilities and labels:'
    #zip(output_prob[top_inds], labels[top_inds])

    # dict_pred['image_id'] = os.path.basename(files[i])
    # dict_pred['label_id'] = top_inds
    # json_lines.append(dict_pred)
     
    y_pred.append(top_inds[0])

# with open(out_file, 'w') as fp:
    # json.dump(json_lines, fp)

labels = [i for i in range(80)]
targets = [str(i) for i in labels]

# draw the consufion matrix to illustrate accuracy of model
cm = confusion_matrix(y_true, y_pred, labels = labels)
print cm

# prin the report of classification
print(classification_report(y_true, y_pred, target_names = targets))

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)

plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)

ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
# plt.show()
fig.savefig('demo.png', dpi = fig.dpi)
