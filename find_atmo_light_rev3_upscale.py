import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tf_guidefilter import *


def find_atmo_light(x):
    '''
    This is a dark channel prior method for finding global atmospheric light using tensorflow Operations. 
    We only use the bottom 62% of the image to try and exclude bright sections of sky
    '''
    cx=230
    
    x=x[0,cx:620]
    r,g,b=tf.split(x,3,axis=2)
    min_dc=tf.minimum(tf.minimum(r,g),b)
    
    darkvec=tf.reshape(min_dc,(1,(620-cx)*940))
    k=np.floor((620-cx)*940/1000).astype('int32')
    imvec=tf.reshape(x,((620-cx)*940,3))
    indices=tf.nn.top_k(darkvec,k=k).indices
    atmo=tf.gather(imvec,indices)
    A=tf.subtract(tf.reduce_mean(atmo,axis=1),0)
    A=tf.concat((tf.fill([620,940,1],A[0][0]),tf.fill([620,940,1],A[0][1]),tf.fill([620,940,1],A[0][2])),axis=2)
    return A

