import caffe
import numpy as np
import sys

if len(sys.argv) != 3:
   print "Usage: python convert_protomean.py proto.mean out.npy"
   sys.exit()

print 'Start blob parse'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
print arr.shape
out = arr[0]
np.save( sys.argv[2], out )
