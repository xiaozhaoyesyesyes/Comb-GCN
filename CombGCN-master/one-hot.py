import tensorflow as tf
import pandas as pd
import numpy as np
data=pd.read_csv("D:\python-code\dim_0",header=None)
 #必须添加header=None，否则默认把第一行数据处理成列名导致缺失
data=data.values.tolist()

tf.compat.v1.disable_v2_behavior()

var1 = tf.one_hot(indices=data,depth=15,axis=1)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    a = sess.run([var1])
    b=np.array(a).reshape(-1,1)
np.savetxt('sample.csv', b, delimiter=",")