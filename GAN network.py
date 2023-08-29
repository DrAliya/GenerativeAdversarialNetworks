#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://github.com/uclaacmai/Generative-Adversarial-Network-Tutorial/blob/master/Generative%20Adversarial%20Networks%20Tutorial.ipynb
# https://github.com/T-Almeida/GAN-study
import pandas as pd
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv(r"C:\Users\Admin\Documents\ALIYA\1. Aliya Thesis\DATASET\1. NSL KDD\mywork\nslKDDNoheaders.csv", header=None)


# In[5]:


df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

# display 5 rows
df[0:5]


# In[6]:


x_columns = df.columns.drop('outcome')
x = df[x_columns].values
dummies = pd.get_dummies(df['outcome']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values


# In[7]:


from sklearn.model_selection import train_test_split

# Create a test/train split.  25% test
# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)


# In[8]:


x_train.shape


# In[9]:


# define some functions that will help us with creating CNNs in Tensorflow

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[10]:


def discriminator(x_train, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()
        #First Conv and Pool Layers
        W_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = avg_pool_2x2(h_conv1)

        #Second Conv and Pool Layers
        W_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = avg_pool_2x2(h_conv2)

        #First Fully Connected Layer
        W_fc1 = tf.get_variable('d_wfc1', [7 * 7 * 16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #Second Fully Connected Layer
        W_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))

        #Final Layer
        y_conv=(tf.matmul(h_fc1, W_fc2) + b_fc2)
    return y_conv


# In[11]:


def generator(z, batch_size, z_dim, reuse=False):
    with tf.variable_scope('generator') as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()
        g_dim = 64 #Number of filters of first layer of generator 
        c_dim = 1 #Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
        s = 28 #Output size of the image
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16) #We want to slowly upscale the image, so these values will help
                                                                  #make that change gradual.

        h0 = tf.reshape(z, [batch_size, s16+1, s16+1, 25])
        h0 = tf.nn.relu(h0)
        #Dimensions of h0 = batch_size x 2 x 2 x 25

        #First DeConv Layer
        output1_shape = [batch_size, s8, s8, g_dim*4]
        W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, 
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv1
        H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
        H_conv1 = tf.nn.relu(H_conv1)
        #Dimensions of H_conv1 = batch_size x 3 x 3 x 256

        #Second DeConv Layer
        output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim*2]
        W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, 
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv2
        H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
        H_conv2 = tf.nn.relu(H_conv2)
        #Dimensions of H_conv2 = batch_size x 6 x 6 x 128

        #Third DeConv Layer
        output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim*1]
        W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, 
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv3
        H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
        H_conv3 = tf.nn.relu(H_conv3)
        #Dimensions of H_conv3 = batch_size x 12 x 12 x 64

        #Fourth DeConv Layer
        output4_shape = [batch_size, s, s, c_dim]
        W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, 
                                         strides=[1, 2, 2, 1], padding='VALID') + b_conv4
        H_conv4 = tf.nn.tanh(H_conv4)
        #Dimensions of H_conv4 = batch_size x 28 x 28 x 1

    return H_conv4


# In[12]:


from tensorflow.keras import Model,Input
from tensorflow.keras.layers import Dense,Activation,LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

#Dimention of random vector using for sampling
z = 100


g_input = Input(shape=[z], name="g_input")
H = Dense(128)(g_input)
H = Activation('relu')(H)
#H = Dense(256)(g_input)
#H = Activation('relu')(H)
H = Dense(41)(H)
g_output = Activation('sigmoid')(H)
"""
#More advancer network
g_input = Input(shape=[z])
H = Dense(256)(g_input)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)
H = Dense(512)(H)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)
H = Dense(1024)(H)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)
H = Dense(mnist_flat_size)(H)
g_output = Activation('tanh')(H)
"""
generator = Model(g_input,g_output, name="generator")
#No need to compile because he will use train with adversarial method
generator.summary()


# In[13]:


d_optimizer = Adam(lr=0.001)


d_input = Input(shape=[41],  name="d_input")
H = Dense(128)(d_input)
H = Activation('relu')(H)
H = Dense(1)(H)
d_output = Activation('sigmoid')(H)

"""
d_input = Input(shape=[mnist_flat_size],  name="d_input")
H = Dense(512)(d_input)
H = LeakyReLU(alpha=0.2)(H)
H = Dense(128)(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dense(1)(H)
d_output = Activation('sigmoid')(H)
"""
discriminator = Model(d_input,d_output, name="discriminator")
discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)
discriminator.summary()


# In[14]:


def make_trainable(net, val=True):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

# Freeze weights in the discriminator for adversarial training
make_trainable(discriminator, False)

gan_optimizer = Adam(lr=0.001)

# Build stacked GAN model to train the generator
gan_input = Input(shape=[z], name="gan_z_input")
gan_fake_samples = generator(gan_input)
gan_output = discriminator(gan_fake_samples)

gan = Model(gan_input, gan_output, name="generator_adversarial")
gan.compile(loss='binary_crossentropy', optimizer=gan_optimizer)
gan.summary()


# In[15]:


# https://www.wouterbulten.nl/blog/tech/getting-started-with-generative-adversarial-networks/
from keras.models import Sequential,Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def discriminator():
    
    net = Sequential()
    input_shape = (121, 1)
    dropout_prob = 0.4

    net.add(Conv1D(64, strides=2, input_shape=input_shape, padding='same'))
    net.add(LeakyReLU())
    
    net.add(Conv2D(128, strides=2, padding='same'))
    net.add(LeakyReLU())
    net.add(Dropout(dropout_prob))
    
    net.add(Conv2D(256, strides=2, padding='same'))
    net.add(LeakyReLU())
    net.add(Dropout(dropout_prob))
    
    net.add(Conv2D(512, strides=1, padding='same'))
    net.add(LeakyReLU())
    net.add(Dropout(dropout_prob))
    
    net.add(Flatten())
    net.add(Dense(1))
    net.add(Activation('sigmoid'))
    
    return net


# In[16]:


def generator():
    
    net = Sequential()
    dropout_prob = 0.4
    
    net.add(Dense(7*256, input_dim=100))
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    net.add(Reshape((7,256)))
    net.add(Dropout(dropout_prob))
    
    net.add(UpSampling1D())
    net.add(Conv1D(128, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    
    net.add(UpSampling1D())
    net.add(Conv1D(64, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    
    net.add(Conv1D(32, padding='same'))
    net.add(BatchNormalization(momentum=0.9))
    net.add(LeakyReLU())
    
    net.add(Conv1D(1, padding='same'))
    net.add(Activation('sigmoid'))
    
    return net


# In[ ]:


optim_discriminator = optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-10)
model_discriminator = Sequential()
model_discriminator.add(discriminator)
model_discriminator.compile(loss='binary_crossentropy', optimizer=optim_discriminator, metrics=['accuracy'])


# In[ ]:


optim_adversarial = Adam(lr=0.0004, clipvalue=1.0, decay=1e-10)
model_adversarial = Sequential()
# model_adversarial.add(net_generator)

# Disable layers in discriminator
for layer in net_discriminator.layers:
    layer.trainable = False

# model_adversarial.add(net_discriminator)
model_adversarial.compile(loss='binary_crossentropy', optimizer=optim_adversarial, metrics=['accuracy'])


# In[9]:


# define the standalone discriminator model using GAN training hacks 
def define_discriminator(in_shape=(128,1)):
    model = Sequential()
    # input layer with image size of 128x128, since its a colored image it has 3 channels
    model.add(Conv1D(16, 3, padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 64x64 using strides of 2,2 and use of LeakyReLU
    model.add(Conv1D(8, 3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 32x32
    model.add(Conv1D(16, 3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 16x16
    model.add(Conv1D(8, 3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # now the image size is down to 16 x 16
    # classifier 
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # compile model learning rate is higher than generator 2e-3
    # use adam optimizer 
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
 
# define the standalone generator model using GAN training hacks
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 16x16 image
    n_nodes = 256 * 16
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16, 256)))
    # upsample to 32x32, use of strides and LeakyReLU
    model.add(Conv1DTranspose(128, 4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 64x64
    model.add(Conv1DTranspose(128,4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 128x128
    model.add(Conv1DTranspose(128, 4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer, use of tanh as per hacks
    model.add(Conv1D(3,3, activation='tanh', padding='same'))
    return model

# combined generator and discriminator model, for updating the generator
# this is alogical GAN model using above defined generator and discriminator 
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # initialize with a sequential model
    model = Sequential()
    # generator and discriminator are initialized later in the code to g_model and d_model respectively 
    # add the generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    # use of adam optimizer, learning rate is lower than discriminator 2e-4
    opt = Adam(lr=0.00002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# In[ ]:


# read images from df_4 dataframe into x_train
x_train = []
def load_real_samples():
    for f, breed in tqdm(df_4.values):
        try:
            img = image.load_img(('/storage/train/{}'.format(f)), target_size=(128, 128))
            # convert to float32
            arr1 = image.img_to_array(img, dtype = 'float32')
            # scale images to [-1,1] from [0,255]
            arr = (arr1 - 127.5) / 127.5
            x_train.append(arr)
        except:
            pass
    return x_train
    
# select real samples
# images are loaded into dataset variable using load_real_samples() function
# below function will randomly select images from dataset and spit out X and y 
# X contains images, y contains lables. (model is not trained with labels as all the images are of same label)
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, len(dataset), n_samples)
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))   
    # retrieve selected images
    X = []  
    f = 0 
    for f in range (len(ix)):
        X.append(dataset[ix[f]])
    #print(len(X))
    return X, y 
    
# Use Gaussian Latent Space
# generate points in latent space as input for the generator
# The latent space defines the shape and distribution of the input to the generator model used to generate new images.
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

