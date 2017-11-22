import sys
import caffe
import os
import json
import glob
import numpy as np


def generate_results(model_def, model_weights, mean_file, result_file, image_dir):

    caffe.set_mode_gpu()
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

    files = []
    json_lines = []
    files.extend(sorted(glob.glob(image_dir + '/*.jpg')))
    files.extend(sorted(glob.glob(image_dir + '/*JPEG')))

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
       print 'predicted class is:', output_prob.argmax()
# sort top five predictions from softmax output
       top_inds = output_prob.argsort()[::-1][:3]  # reverse sort and take five largest items
       dict_pred['image_id'] = os.path.basename(files[i])
       dict_pred['label_id'] = top_inds.tolist()
       json_lines.append(dict_pred)
     
    with open(result_file, 'w') as fp:
       json.dump(json_lines, fp)

if __name__ == '__main__':
  # sys.path.insert(0, data_root)
  if(len(sys.argv) < 5):
     print 'usage python classification_json.py deploy.prototxt .caffemodel .npy json_file image_folder'
     sys.exit(-1)

  model_def = sys.argv[1]
  model_weights = sys.argv[2]
  mean_file = sys.argv[3]
  result_file = sys.argv[4]
  image_dir = sys.argv[5]

  generate_results(model_def, model_weights, mean_file, result_file, image_dir)
 
