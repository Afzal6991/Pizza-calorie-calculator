


import os, sys

import tensorflow as tf

import numpy as np


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#image_path = sys.argv[0]

image_path ="test hamburger.jpg"



image_data = tf.gfile.FastGFile(image_path, 'rb').read()



label_lines = [line.rstrip() for line 
               in tf.gfile.GFile("retrained_labels.txt")]



with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    
    graph_def.ParseFromString(f.read())
    
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
   
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    
    
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        
        score = predictions[0][node_id]
        
        print('%s (score = %.5f)' % (human_string, score*100))
        
maxx=top_k[0]
    
label_calor=label_lines[maxx]
#x=[200,345,245,123,565,776,874,236,454,232,435,456,232,67]
cc=np.array(([200,345,245,123,565,776,87,23,454,232,435,456,232,67,333,888]), dtype=np.uint8)

print('calories are')

m=cc[maxx]

print(m)
        