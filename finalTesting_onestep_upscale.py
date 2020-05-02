#r# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 11:44:52 2017

@author: Cameron Hodges
"""
import h5py
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow_addons.layers import Maxout 
import time
from PIL import Image
import os
import matplotlib.pyplot as plt
import scipy.misc
import cv2
from find_atmo_light_rev3_upscale import *
from tf_guidefilter_upscale import *

from skimage.measure import compare_ssim as ssim

from skimage.transform import resize

#%%

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray  
#Input directories

#%%

#%%

cutOff=float(0.05)

gf_ind="y"



#%%

def loadSourceList(file_path):
    listing = os.listdir(file_path)
    num_samples=np.size(listing)
    #print (num_samples)
    imCompath=[]
    for file in range(num_samples):
        imCompath+=[file_path+listing[file]]
    return np.array(imCompath)

def load_images(file_name):
    if train_batch_size == 1:
        file_name=[file_name] 
    else: 
        file_name=file_name 
    data_list=np.ones((len(file_name),620,940,3))
    k=0
    for i in range(len(file_name)):
        data=Image.open('data/'+file_name[i])
        imgsize=data.size
        data_list[i,:,:]=(np.array(data.resize((940,620)))).reshape((1,620,940,3))
        
        
    return imgsize, data_list/255

hazy=os.listdir("data")


#%%
tf.reset_default_graph()
#Start Generator Code
name=None

def conv2d(x,W,wd,name):
    return tf.nn.conv2d(x, W, strides=[1,wd[0],wd[1],1], padding='SAME',name=name)

def maxpool2d(x,wd,name):
    return tf.nn.max_pool(x,ksize=(1,wd[0],wd[1],1), strides=[1,wd[0],wd[1],1], padding='SAME',name=name)

with tf.name_scope('G_weights'):
    G_weights={'G_W_conv1':tf.Variable(tf.random_normal([5,5,3,16],mean=0, stddev=0.003)),
             'G_W_conv2a':tf.Variable(tf.random_normal([3,3,16,16],mean=0, stddev=0.003)),
             'G_W_conv2b':tf.Variable(tf.random_normal([5,5,16,16],mean=0, stddev=0.003)),
             'G_W_conv2c':tf.Variable(tf.random_normal([7,7,16,16],mean=0, stddev=0.003)),
             'G_W_conv4':tf.Variable(tf.random_normal([6,6,48,48],mean=0, stddev=0.003)),
             'G_out':tf.Variable(tf.random_normal([48, 1],mean=0, stddev=0.003))}
    tf.summary.histogram("G_weights_c1", G_weights['G_W_conv1'])
    tf.summary.histogram("G_weights_c2a", G_weights['G_W_conv2a'])
    tf.summary.histogram("G_weights_c2b", G_weights['G_W_conv2b'])
    tf.summary.histogram("G_weights_c2c", G_weights['G_W_conv2c'])
    tf.summary.histogram("G_weights_c4", G_weights['G_W_conv4'])
    tf.summary.histogram("G_weights_out", G_weights['G_out'])
            
with tf.name_scope('G_biases'):    
    G_biases={'G_W_conv1':tf.Variable(tf.zeros([16])),
             'G_W_conv2a':tf.Variable(tf.zeros([16])),
             'G_W_conv2b':tf.Variable(tf.zeros([16])),
             'G_W_conv2c':tf.Variable(tf.zeros([16])),
             'G_W_conv4':tf.Variable(tf.zeros([48])),
             'G_out':tf.Variable(tf.zeros([1]))}

saver_G=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="G_"))

def generator_neural_network(x,train_batch_size):

    
    def core_network(x):
        with tf.name_scope('G_core_network'):

            #conv1a=conv1
            x=tf.subtract(x,tf.constant(0.5))
            conv1=conv2d(x,G_weights['G_W_conv1'],(5,5),'L_G_conv1')
            
            conv1a=Maxout(16,-1).call(conv1)
            conv2a=conv2d(conv1a,G_weights['G_W_conv2a'],(3,3),'L_G_conv2a')
            conv2a=tf.nn.dropout(conv2a, keep_prob=0.8)
            conv2a=maxpool2d(conv2a,(2,2),'L_G_conv2a_max')
            conv2b=conv2d(conv1a,G_weights['G_W_conv2b'],(5,5),'L_G_conv2b')
            conv2b=tf.nn.dropout(conv2b, keep_prob=0.8)
            conv2b=maxpool2d(conv2b,(2,2),'L_G_conv2b_max')
            conv2c=conv2d(conv1a,G_weights['G_W_conv2c'],(7,7),'L_G_conv2c')
            conv2c=tf.nn.dropout(conv2c, keep_prob=0.8)
            conv2c=maxpool2d(conv2c,(2,2),'L_G_conv2c_max')
        
            mrg=tf.concat([conv2a, conv2b, conv2c], axis=3)
            
            conv4=conv2d(mrg,G_weights['G_W_conv4'],(6,6),'L_G_conv4')
            
            conv4a=tf.minimum(tf.maximum(conv4,tf.constant(0,'float')),tf.constant(1,'float'))
            fc = tf.reshape(conv4a ,[-1,1*48])
            output=tf.matmul(fc, G_weights['G_out'])+G_biases['G_out']
            #output=tf.minimum(tf.maximum(output,tf.constant(0,'float')),tf.constant(1,'float'))
            
            
            return output
    
    def t_negCorrection(dehzed,rgb):
        pvc_1_chmin=tf.reduce_min((dehzed),axis=3)
        pvc_1_chmax=tf.reduce_max((dehzed),axis=3)
        
        inRangecondition=tf.reshape(tf.logical_and(tf.greater(pvc_1_chmin,tf.constant(-1e-5,'float')),tf.less(pvc_1_chmax,tf.constant(1.00000001,'float'))),[1,620,940,1])
        inRangecondition=tf.concat((inRangecondition,inRangecondition,inRangecondition),axis=3)
        correctedImage=tf.where(inRangecondition,dehzed,rgb)
        return correctedImage
          
      
    def reconstruct_image(patches,num_ch):
        image_h, image_w=620, 940
        pad = [[0, 0], [0, 0]]
        patch_h = 20
        patch_w = 20
        patch_ch = num_ch
        p_area = 400
        h_ratio = 31
        w_ratio = 47
         
        image = tf.reshape(patches, [1, h_ratio, w_ratio, p_area, patch_ch])
        #print(image)
        image = tf.split(image, p_area, 3)
        
        image = tf.stack(image, 0)
        #print(image)
        image = tf.reshape(image, [p_area, h_ratio, w_ratio, patch_ch])
        #print(image)
        image = tf.batch_to_space_nd(image, [patch_h, patch_w], pad)
        return image
     
    def reformTransmap(transmap):
        txL=[]
        for k in range(1457):
            txL+=[tf.fill([20,20],transmap[0,k])]
        txL=tf.stack(txL,axis=0)
        txF=reconstruct_image(txL,1)
        #txF=smoothFilter(txF)
        return txF     
      
    def generate_patches(image):
        patch_h, patch_w = 20,20
        pad = [[0, 0], [0, 0]]
        image_h = 620
        image_w = 640
        image_ch = 3
        p_area = patch_h * patch_w
        patches = tf.space_to_batch_nd([image[0]], [patch_h, patch_w], pad)
        #print(patches.shape)
        
        patches = tf.split(patches, p_area, 0)
        
        patches = tf.stack(patches, 3)
        #print(patches)
        patches = tf.reshape(patches, [-1, patch_h, patch_w, image_ch])
        return patches
     
        
    def dehaze_single(input_im, trans_map):

        input_im=tf.reshape(input_im,[1,620,940,3])
        
        airlight=tf.reshape(find_atmo_light(input_im),[1,620,940,3])
        #observed=transmap*true+airlight(1-trans)
        #true=(observed-airlight(1-trans))/trans
        filter_im=tf.subtract(tf.constant(1.0),tf.reshape(tf.image.rgb_to_grayscale(input_im),[620,940]))
        #filter_im=tf.reshape(tf.image.rgb_to_grayscale(input_im),[380,620])
        if gf_ind=="y":
            transMap=tf.reshape(reformTransmap(trans_map),[620,940])
            transMap=tf.reshape(t_guidedFilter(filter_im,transMap,40,1e-5),[1,620,940,1])
            #transMap=guided_filter(transMap, filter_im, 70, 1e-5)
        if gf_ind=="n":
            transMap=tf.reshape(reformTransmap(trans_map),[1,620,940,1])

        
        #transMap=tf.reshape(tf.subtract(tf.constant(1.0),t_guidedFilter(transMap, filter_im,40,1e-5)),[1,380,620,1])
        transMap=tf.concat((transMap,transMap,transMap),axis=3)
        transMap=tf.reshape(tf.maximum(tf.reshape(transMap,[-1]),tf.constant(cutOff)),[-1,620,940,3])#Best if 0.6 with neg correction
        final_image=tf.add(tf.divide(tf.subtract(input_im,airlight),transMap),airlight)
        #final_image=t_negCorrection(final_image,input_im)

        return final_image  
        
        
    def dehaze_tensor(in_values):
        with tf.name_scope('Dehaze_tensor'):
            input_im, trans_map = in_values
            batch_size=input_im.shape[0].value
            output_batch=dehaze_single(input_im[0], trans_map[0])
            for l in range(batch_size-1):
                output_batch=tf.concat([output_batch,dehaze_single(input_im[l+1], trans_map[l+1])],axis=0)
                print(l+1)
                
            return output_batch
    #with tf.name_scope('input_image'):
        #input_image=tf.placeholder('float',[None,380,620,3])
    
    
    
    with tf.name_scope('complete_network'):
        siamese_in=[]
        
        #x_trans=tf.subtract(x,tf.constant(0.5))
        for i in range(train_batch_size): 
            patches=tf.reshape(generate_patches(x[i:i+1]),[1457,20,20,3])
            output_dense=tf.reshape(core_network(patches),[-1,1457])
            #output_dense_n=tf.maximum(output_dense,tf.constant(20.0,'float'))
            siamese_in+=[dehaze_single(x[i:i+1], output_dense)]
            
            
        siamese_in=tf.reshape(tf.stack(siamese_in,axis=0),[-1,620,940,3])

       
        return siamese_in
#%%
train_batch_size=1    
with tf.name_scope(name):
    x1=tf.placeholder('float',[train_batch_size,620,940,3],name='x1')#Always clean Image

    #y=tf.placeholder('float',[None],name='y')
with tf.name_scope("generator_network"):
    dehz=generator_neural_network(x1,train_batch_size)



   
#%%



with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())
        
    
    save_path_G=saver_G.restore(sess, "G_ C_model_1_ecud_min.ckpt") 


    

    
    for n in range(len(os.listdir("data"))):
        #n=1   
        ti=time.time()
        originalsize, inputImage=load_images(hazy[n])
        pvc =sess.run(dehz, feed_dict={x1: inputImage})
        pvc=pvc.reshape((620,940,3))
        
        pvc = resize(pvc, (originalsize[1], originalsize[0]), anti_aliasing=True)
        pvc=np.concatenate((np.expand_dims(pvc[...,2],-1),np.expand_dims(pvc[...,1],-1),np.expand_dims(pvc[...,0],-1)),axis=2)
        cv2.imwrite(str('output/'+hazy[n].split('/')[-1]), pvc * 255.0,[cv2.IMWRITE_JPEG_QUALITY, 100])
        #    np.savetxt(fname="C:\\Users\\Cameron\\\Dropbox\\Masters Research\\Dehazing Code Base\\loss_reg\\finalPerformance-"+weightSelect+".csv",X=print_loss_reg,fmt='%s',delimiter=",")
        print('image - '+str(n+1)+' completed'+"Time: "+"%.2f" %  round(time.time()-ti,2))
        
    
    
