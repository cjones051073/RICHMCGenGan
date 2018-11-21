#! /usr/bin/env python3

from __future__ import print_function, division
import sys, os

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import InputLayer, Concatenate, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model

import keras.backend as K

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import RICH

import argparse

from tqdm import tqdm

class WGAN():
  
  def __init__(self):

    # read arguments

    parser = argparse.ArgumentParser(description='Job Parameters')

    parser.add_argument( '--name', type=str, default="PPtR1R2HitsR12EntryExit" )
    
    parser.add_argument( '--inputvars', type=str, nargs='+',
                         default = [  'TrackP', 'TrackPt'
                                      ,'NumRich1Hits', 'NumRich2Hits'
                                      ,'TrackRich1EntryX', 'TrackRich1EntryY'
                                      ,'TrackRich1ExitX', 'TrackRich1ExitY'
                                      ,'TrackRich2EntryX', 'TrackRich2EntryY'
                                      ,'TrackRich2ExitX', 'TrackRich2ExitY' ] )
    
    parser.add_argument( '--outputvars', type=str, nargs='+',
                         default = ['RichDLLe', 'RichDLLk', 'RichDLLmu',
                                    'RichDLLp', 'RichDLLd', 'RichDLLbt'] )

    parser.add_argument( '--datadir', type=str, default="/home/jonesc/Projects/RICHMCGenGan/data" )
    parser.add_argument( '--outputdir', type=str, default="/home/jonesc/Projects/RICHMCGenGan/output" )

    parser.add_argument( '--nepochs', type=int, default="100000" )

    parser.add_argument( '--datareadsize', type=int, default="4000000" )

    parser.add_argument( '--batchsize', type=int, default="50000" )

    parser.add_argument( '--ncriticpergen', type=int, default="15" )

    parser.add_argument( '--ncriticlayers', type=int, default="10" )
    parser.add_argument( '--ngeneratorlayers', type=int, default="10" )
    parser.add_argument( '--leakrate', type=float, default="0.0" )
    parser.add_argument( '--dropoutrate', type=float, default="0.0" )
    
    self.args,unparsed = parser.parse_known_args()
    print( "Job arguments", self.args )

    # parameters

    # amount of noise to add to training data
    self.NOISE_DIMENSIONS = 64

    # Total input dimensions of generator (including noise)
    self.GENERATOR_DIMENSIONS = self.NOISE_DIMENSIONS + len(self.args.inputvars)

    # Num inputs to critic
    self.CRITIC_DIMENSIONS = len(self.args.inputvars) + len(self.args.outputvars)

    # optimiser
    optimizer = RMSprop( lr = 0.00005 )

    # Build and compile the critic
    self.critic = self.build_critic()
    self.critic.compile( loss      = self.wasserstein_loss,
                         optimizer = optimizer,
                         metrics   = ['accuracy'] ) 
    
    # Build the generator
    self.generator = self.build_generator()

    # Physics inputs
    phys_inputs  = Input(shape=(len(self.args.inputvars),))
    print("phys_inputs", phys_inputs )
    
    # Noise inputs
    noise_inputs = Input(shape=(self.NOISE_DIMENSIONS,))
    print("noise_inputs", noise_inputs )

    # generator inputs
    gen_inputs = [ phys_inputs, noise_inputs ]
    
    # Generator outputs
    gen_outputs = self.generator( gen_inputs )
    print( "gen_outputs", gen_outputs )

    # For the combined model we will only train the generator
    #self.critic.trainable = False

    # The critic takes generated outputs + physics inputs as input
    critic_output = self.critic( [ phys_inputs, gen_outputs ] )
    print( "critic_output", critic_output )

    # The combined model  (stacked generator and critic)
    self.combined = Model( gen_inputs, critic_output )
    self.combined.compile(loss      = self.wasserstein_loss,
                          optimizer = optimizer,
                          metrics   = ['accuracy'] )

  def wasserstein_loss(self, y_true, y_pred):
    return K.mean( y_true * y_pred )

  def build_generator(self):

    # physics inputs
    phys_input = Input( shape=(len(self.args.inputvars),) )
    phys_dense = Dense(64,)(phys_input)

    # noise inputs
    noise_input = Input( shape=(self.NOISE_DIMENSIONS,) )
    noise_dense = Dense(64,)(noise_input)

    # merge the two inputs
    layers = concatenate([phys_dense,noise_dense])

    # add the inner layers
    for i in range(0,self.args.ngeneratorlayers) :
      layers = Dense(128, activation='relu' )(layers)
      if self.args.leakrate    > 0 : layers = LeakyReLU(self.args.leakrate)(layers)
      if self.args.dropoutrate > 0 : layers = Dropout(self.args.dropoutrate)(layers)

    # output layer
    layers =  Dense(len(self.args.outputvars))(layers)

    # build the model and return
    generator = Model( inputs=[ phys_input, noise_input ], outputs = layers )
    print( "Building Generator, #inputs=", generator.inputs )
    generator.summary()
    return multi_gpu_model( generator, gpus=3 )

  def build_critic(self):

    # https://keras.io/getting-started/functional-api-guide/

    # physics inputs
    phys_input = Input( shape=(len(self.args.inputvars),) )
    phys_dense = Dense(64,)(phys_input)

    # DLL inputs
    dlls_input = Input( shape=(len(self.args.outputvars),) )
    dlls_dense = Dense(64,)(dlls_input)

    # merge the two inputs
    layers = concatenate([phys_dense,dlls_dense])

    # add the inner layers
    for i in range(0,self.args.ncriticlayers) :
      layers = Dense(128, activation='relu' )(layers)
      if self.args.leakrate    > 0 : layers = LeakyReLU(self.args.leakrate)(layers)
      if self.args.dropoutrate > 0 : layers = Dropout(self.args.dropoutrate)(layers)

    # output layer (one variable)
    layers = Dense(1)(layers)

    # build the model and return
    critic = Model( inputs=[ phys_input, dlls_input ], outputs = layers )
    print( "Building Critic, #inputs=", critic.inputs )
    critic.summary()
    return multi_gpu_model( critic, gpus=3)

  def train(self):

    # Load the physics inputs
    phys_input_raw = RICH.createLHCbData( self.args.inputvars,  
                                          self.args.datareadsize, 'KAONS',
                                          self.args.datadir )
    print( "phys_input_raw", phys_input_raw.shape, '\n', phys_input_raw.tail() )
    # Load the target outputs
    dlls_output_raw = RICH.createLHCbData( self.args.outputvars,  
                                          self.args.datareadsize, 'KAONS',
                                          self.args.datadir )
    print( "dlls_output_raw", dlls_output_raw.shape, '\n', dlls_output_raw.tail() )

    # scale the data
    print( "Scaling the data" )
    ndataforscale = min( 1000000, phys_input_raw.shape[0] )
    phys_scaler = RICH.getScaler( phys_input_raw.iloc[0:ndataforscale] )
    phys_input  = pd.DataFrame( phys_scaler.transform(phys_input_raw),
                                columns = phys_input_raw.columns, dtype=np.float32 )
    print( "phys_input", phys_input.shape, '\n', phys_input.tail() )
    dlls_scaler = RICH.getScaler( dlls_output_raw.iloc[0:ndataforscale] )
    dlls_output = pd.DataFrame( dlls_scaler.transform(dlls_output_raw),
                                columns = dlls_output_raw.columns, dtype=np.float32 )
    print( "dlls_input", dlls_output.shape, '\n', dlls_output.tail() )
    
    # Adversarial ground truths
    valid = -np.ones((self.args.batchsize, 1))
    fake  =  np.ones((self.args.batchsize, 1))
    
    for epoch in tqdm(range(self.args.nepochs)):
      
      for _ in range(self.args.ncriticpergen):
        
        # ---------------------
        #  Train Critic
        # ---------------------
        
        # Select a random batch of input
        idx = np.random.randint( 0, phys_input.shape[0], self.args.batchsize )
        phys_batch = phys_input.values[idx]
        dlls_batch = dlls_output.values[idx]
        
        # Sample noise as generator input
        noise_batch = np.random.normal(0, 1, (self.args.batchsize, self.NOISE_DIMENSIONS))
          
        # Generate a batch of new DLL values
        dlls_gen = self.generator.predict( [ phys_batch, noise_batch ] )
          
        # Train the critic
        #self.critic.trainable = True
        #d_loss_real = self.critic.train_on_batch( [phys_batch,dlls_batch], valid)
        #d_loss_fake = self.critic.train_on_batch( [phys_batch,dlls_gen], fake)
        #d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
          
        # Clip critic weights
        #for l in self.critic.layers:
        #  weights = l.get_weights()
        #  weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
        #  l.set_weights(weights)


      # ---------------------
      #  Train Generator
      # ---------------------
      
      #self.critic.trainable = False
      #g_loss = self.combined.train_on_batch( [phys_batch,noise_batch], valid )
        
      # Plot the progress
      #print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))


if __name__ == '__main__':
  
  wgan = WGAN()
  wgan.train()
