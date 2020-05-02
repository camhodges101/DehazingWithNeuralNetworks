# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:38:29 2018

@author: Cameron Hodges
"""


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%
    
def t_boxFilter(img, r):
    rows, cols = img.shape[0].value, img.shape[1].value
        
    tr=np.array(r).reshape((1))
    tr=tf.reshape(tf.constant(r),[1])
    
    
    imCum=tf.cumsum(img,axis=0)
    imDst=imCum[r : 2*r+1, :]
    imDst=tf.concat((imDst,tf.subtract(imCum[2*r+1 : rows, :],imCum[0 : rows-2*r-1, :])),axis=0)
    imDst=tf.concat((imDst,tf.subtract(tf.reshape(tf.tile(imCum[rows-1, :], tr),[r,-1]),imCum[rows-2*r-1 : rows-r-1, :])),axis=0)
    
    imCum=tf.cumsum(imDst,axis=1)
    imDst=imCum[:,r : 2*r+1]
    imDst=tf.concat((imDst,tf.subtract(imCum[:,2*r+1 : cols],imCum[:,0 : cols-2*r-1])),axis=1)
    a=tf.reshape(tf.tile(imCum[:,cols-1], tr),[r,620])
    a=tf.transpose(a)
    
    imDst=tf.concat((imDst,tf.subtract(a,imCum[:,cols-2*r-1 : cols-r-1])),axis=1)
    
    return imDst
         

def t_guidedFilter(I, p, r, eps):
    rows, cols = I.shape[0].value, I.shape[1].value
    eps=tf.constant(eps)
    N = t_boxFilter(tf.ones((rows, cols)),r)
    
    meanI = tf.divide(t_boxFilter(I, r),N)
    meanp = tf.divide(t_boxFilter(p, r),N)
    
    meanIp = tf.divide(t_boxFilter(tf.multiply(I,p),r),N)
    
    covIP=tf.subtract(meanIp,tf.multiply(meanI,meanp))
    meanII=tf.divide(t_boxFilter(tf.multiply(I,I),r),N)
    
    varI=tf.subtract(meanII,tf.multiply(meanI,meanI))
    
    a=tf.divide(covIP,tf.add(varI,eps))
    b=tf.subtract(meanp,tf.multiply(a,meanI))
    
    meanA=tf.divide(t_boxFilter(a,r),N)
    meanB=tf.divide(t_boxFilter(b,r),N)

    q=tf.add(tf.multiply(meanA,I),meanB)
    return q

    

