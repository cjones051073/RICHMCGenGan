#! /usr/bin/env python

import os, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import RICH

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(1234)

# The dimension of the input, including random noise
input_dim  = 64 

# names of variables to extract for training data
train_names = [ 'NumRich1Hits', 'NumRich2Hits', 'TrackP', 'TrackPt' 
                #,'NumPVs'
                #,"NumLongTracks"
                #,"TrackChi2PerDof", "TrackNumDof"
                #,'TrackVertexX', 'TrackVertexY', 'TrackVertexZ' 
                #,'TrackRich1EntryX', 'TrackRich1EntryY' 
                #,'TrackRich1ExitX', 'TrackRich1ExitY',
                #,'TrackRich2EntryX', 'TrackRich2EntryY'
                #,'TrackRich2ExitX', 'TrackRich2ExitY' 
]
# Train on random input only
#train_names = [ ]

# names for target data
target_names = [ 'RichDLLe', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLd', 'RichDLLbt' ]
output_dim = len(target_names)

# amount of noise to add to training data
n_noise_data = input_dim - len(train_names) 

# Adam optimizer
def get_optimizer():
    from keras.optimizers import Adam, RMSprop
    return Adam( lr=0.0002, beta_1=0.5 )
    #return RMSprop(lr=0.00005)

def get_generator(optimizer):

    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.layers.advanced_activations import LeakyReLU
    from keras import initializers
    
    generator = Sequential()
    generator.add(Dense(256, input_dim=input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(output_dim, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    generator.summary()

    return generator

def discriminator_loss( y_true, y_pred ) :
    import keras.backend as K
    # Wasserstein 
    return K.mean( y_true * y_pred )
    # Cramer

def get_discriminator(optimizer):

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout
    from keras.layers.advanced_activations import LeakyReLU
    from keras import initializers
    import keras.layers as ll

    SIZE   = 128
    LAYERS = 10

    weight_init = initializers.RandomNormal( mean=0, stddev=0.02 )

    discriminator = Sequential()

    discriminator.add( Dense( SIZE, 
                              input_dim=output_dim,
                              activation='relu',
                              kernel_initializer=weight_init ) )
                            
    for i in range(LAYERS-1) :
        discriminator.add( Dense( SIZE, activation='relu' ) )

    discriminator.add(Dense(1, activation='sigmoid'))
    
    #discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #discriminator.compile(loss=d_loss, optimizer=optimizer, metrics=['accuracy'])

    discriminator.compile(loss=discriminator_loss, optimizer=optimizer, metrics=['accuracy'])

    discriminator.summary()

    return discriminator

def get_gan_network(discriminator, random_dim, generator, optimizer):
    from keras.models import Model
    from keras.layers import Input

    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False

    # gan input
    gan_input = Input(shape=(input_dim,))

    # the output of the generator
    x = generator(gan_input)

    # get the output of the discriminator 
    gan_output = discriminator(x)

    # create the gan
    gan = Model( inputs=gan_input, outputs=gan_output  )
    gan.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'] )

    gan.summary()

    return gan

def getTrainBatch( train_input, batch_i ):
    
    # Get the real physics inputs
    phys_input = train_input[batch_i]
   
    # generate some additional random input data
    noise_input = np.random.normal( 0, 1, size=[len(batch_i),n_noise_data] )
    
    # add physics and noise input
    all_input = np.concatenate( (phys_input,noise_input), axis=1 )

    return all_input

def train( epochs=1, batches_per_epoch=100, batch_size=256, maxData=-1 ):

    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import platform

    plots_dir = "plots/"+platform.node()+"/LHCbGen/"
    if     os.path.exists(plots_dir) : shutil.rmtree(plots_dir) 
    if not os.path.exists(plots_dir) : os.makedirs(plots_dir)
    
    # Get the training and testing data
    all_input  = RICH.createLHCbData( train_names,  maxData, 'KAONS' )
    all_target = RICH.createLHCbData( target_names, maxData, 'KAONS' )

    # Split into train and test samples
    train_input,  test_input  = train_test_split( all_input,  random_state = 1234 )
    train_target, test_target = train_test_split( all_target, random_state = 4321 ) 

    # plots of the target output
    RICH.plots( "output_raw", all_target, plots_dir )

    # make some plots of the raw inputs
    RICH.plots( "inputs_raw", all_input, plots_dir )

    # normalise the data
    print( "Normalising the data" )
    if len(all_input.columns) > 0 :
        input_scaler = RICH.getScaler( all_input  )
        train_input  = pd.DataFrame( input_scaler.transform(train_input), columns = train_input.columns )
        test_input   = pd.DataFrame( input_scaler.transform(test_input),  columns = test_input.columns )
    output_scaler = RICH.getScaler( all_target )
    train_target  = pd.DataFrame( output_scaler.transform(train_target), columns = train_target.columns )
    test_target   = pd.DataFrame( output_scaler.transform(test_target),  columns = test_target.columns )

    # plots of the target normalised output
    RICH.plots( "output_norm", all_target, plots_dir )

    # make some plots of the normalised inputs
    RICH.plots( "inputs_norm", all_input, plots_dir )

    # Build our GAN netowrk
    adam          = get_optimizer()
    generator     = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan           = get_gan_network(discriminator, input_dim, generator, adam)

    test_loss   = [ ]
    test_acc    = [ ]
    train_loss  = [ ]
    train_acc   = [ ]

    run_epochs = [ ]

    for e in range(1, epochs+1):

        run_epochs += [ e ]
        print ( '-'*15, 'Epoch %d' % e, '-'*15 )

        for _ in tqdm(range(batches_per_epoch)):

            # get a random set of indices
            batch_i = np.random.randint( 0, train_input.shape[0], size=batch_size ) 

            # generate output
            generated_output = generator.predict( getTrainBatch(train_input.values,batch_i) )

            # target output values for this batch
            batch_target = train_target.values[batch_i]

            # concatenate real and generated output
            disc_train_d = np.concatenate( [ batch_target, generated_output ] )

            # Labels for generated (0) and real (0.9) target data
            disc_target = np.zeros(2*batch_size)
            # One-sided label smoothing
            disc_target[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch( disc_train_d, disc_target )

            # get a new random set of indices
            batch_i = np.random.randint( 0, train_input.values.shape[0], size=batch_size ) 

            # Train generator
            discriminator.trainable = False
            gan.train_on_batch( getTrainBatch(train_input.values,batch_i), np.ones(batch_size) )

        # evaluate the generator on training data
        train_noise = np.random.normal( 0, 1, size=[train_input.values.shape[0],n_noise_data] )
        all_train   = np.concatenate( (train_input,train_noise), axis=1 )
        loss_acc    = generator.evaluate( all_train, train_target, batch_size )
        train_loss += [ loss_acc[0] ]
        train_acc  += [ loss_acc[1] ]

        # evaluate the generator on test data
        test_noise  = np.random.normal( 0, 1, size=[test_input.shape[0],n_noise_data] )
        all_test    = np.concatenate( (test_input,test_noise), axis=1 )
        loss_acc    = generator.evaluate( all_test, test_target, batch_size )
        test_loss  += [ loss_acc[0] ]
        test_acc   += [ loss_acc[1] ]
  
        # make plots every now and then
        if e == 1 or e % 5 == 0:

            epoch_dir = plots_dir+"epochs/"+str( '%06d' % e )+"/"
            if not os.path.exists(epoch_dir) : os.makedirs(epoch_dir)

            outputs_test  = generator.predict( all_test )
            outputs_train = generator.predict( all_train )

            # make some plots of example generated output for this epoch
         
            RICH.plots( "gen_norm_outputs", pd.DataFrame(outputs_test,columns=target_names), epoch_dir )
            outputs_test_raw = output_scaler.inverse_transform( outputs_test )
            RICH.plots( "gen_raw_outputs", pd.DataFrame(outputs_test_raw,columns=target_names), epoch_dir )

            # output correlations
            RICH.outputCorrs( "correlations", outputs_test, test_target.values, target_names, epoch_dir )
        
            # loss plots
            plt.figure(figsize=(18,15))
            plt.plot( run_epochs, train_loss, 'bo', label='Training Loss')
            plt.plot( run_epochs, test_loss,   'b', label='Validation Loss')
            plt.title('Training and validation loss')
            plt.legend()
            plt.savefig(epoch_dir+'loss.png')
            plt.close()

            # accuracy plots
            plt.figure(figsize=(18,15))
            plt.plot( run_epochs, train_acc, 'bo', label='Training Accuracy')
            plt.plot( run_epochs, test_acc,   'b', label='Validation Accuracy')
            plt.title('Training and validation accuracy')
            plt.legend()
            plt.savefig(epoch_dir+'accuracy.png')
            plt.close()

            disc_bins = 100

            # Discriminator output (test data)
            disc_gen = discriminator.predict( outputs_test )
            disc_tar = discriminator.predict( test_target  )
            plt.figure(figsize=(18,15))
            data = np.concatenate( [disc_gen,disc_tar], axis=1 ) 
            plt.hist( data, disc_bins, density=True, histtype='step', label=['Generated','Target'] )
            plt.title('Discriminator : Test Data')
            plt.legend()
            plt.savefig(epoch_dir+'discriminator-test.png')
            plt.close()

            # Discriminator output (train data)
            disc_gen = discriminator.predict( outputs_train )
            disc_tar = discriminator.predict( train_target  )
            plt.figure(figsize=(18,15))
            data = np.concatenate( [disc_gen,disc_tar], axis=1 ) 
            plt.hist( data, disc_bins, density=True, histtype='step', label=['Generated','Target'] )
            plt.title('Discriminator : Train Data')
            plt.legend()
            plt.savefig(epoch_dir+'discriminator-train.png')
            plt.close()

if __name__ == '__main__' :

    # main full options
    #maxData    = -1
    #batch_size = 256
    #epochs     = 500
    #b_per_e    = 2000 

    # medium options
    maxData    = 100000
    batch_size = 256
    epochs     = 100
    b_per_e    = 200

    #train( epochs, b_per_e, batch_size, maxData )

    # testing
    train( 50, 100, 128, 10000 )
