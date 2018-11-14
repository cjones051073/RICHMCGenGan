#! /usr/bin/env python

import os, shutil
import numpy as np

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(1234)

# The dimension of the input and output data
input_dim  = 128
output_dim = 6

# names of variables to extract for training data
train_names = [ 'MCID',
                'NumPVs', 'NumRich1Hits', 'NumRich2Hits', "NumLongTracks",
                'TrackP', 'TrackPt', "TrackChi2PerDof", "TrackNumDof",
                'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
                'TrackRich1EntryX', 'TrackRich1EntryY', 
                'TrackRich1ExitX', 'TrackRich1ExitY', 
                'TrackRich2EntryX', 'TrackRich2EntryY',
                'TrackRich2ExitX', 'TrackRich2ExitY' ]
# Train on random input only
#train_names = [ ]

# names for target data
target_names = [ 'RichDLLe', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLd', 'RichDLLbt' ]

# amount of noise to add to training data
n_noise_data = input_dim - len(train_names) 

# load the LHCb data
def createLHCbData( maxData = -1 ):

    import gzip, random
    
    # open the compressed data file
    f = gzip.open('data/PID-train-data.txt.gz')

    # working arrays for the indices into the data file to get the required entries
    train_indices  = [ ]
    target_indices = [ ]

    # data arrays to return
    train_data   = [ ]
    test_data    = [ ]
    train_target = [ ]
    test_target  = [ ]

    # fraction of data to use for testing
    test_frac = 0.2

    print( "Loading the data from file" )

    # loop ove files lines
    iData   = 0
    for d in f.readlines() :
        
        values = d.split()
        #print ( values )

        if iData == 0 :
            # First line in the file are the names, not values

            # Extract the indices for training data
            for n in train_names :
                i = 0
                for v in values :
                    if v.decode('UTF-8') == n : 
                        print( n, "index", i )
                        train_indices += [i]
                    i += 1

            # extract the indices for the target data
            for n in target_names :
                i = 0
                for v in values :
                    if v.decode('UTF-8') == n : 
                        print( n, "index", i )
                        target_indices += [i]
                    i += 1

        else:

            # Read the training data
            d_train = [ ]
            for i in train_indices :
                d_train += [ float(values[i]) ]
                
            # read the target data
            d_target = [ ]
            for i in target_indices :
                d_target += [ float(values[i]) ]

            #print ( d_train )

            # save as either train or test data
            if random.uniform(0,1) > test_frac :
                train_data   += [ d_train ]
                train_target += [ d_target ]
            else:
                test_data    += [ d_train ]
                test_target  += [ d_target ]

        # check how much data has been loaded
        iData += 1
        if maxData > 0 and iData > maxData : break

    f.close()

    # finally return the data arrays
    return ( np.array(train_data),   np.array(test_data),
             np.array(train_target), np.array(test_target) )


def normalise( train_input, test_input, train_target, test_target ):

    print( "Normalising the data" )

    # Use a preprocessor from sklearn
    from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
    
    if len(train_names) > 0 :
        #scalar_input = RobustScaler(quantile_range=(25,75))
        scalar_input = PowerTransformer(method='yeo-johnson')
        train_input  = scalar_input.fit_transform(train_input)
        test_input   = scalar_input.fit_transform(test_input)

    scalar_output = MinMaxScaler()
    train_target = scalar_output.fit_transform(train_target)
    test_target  = scalar_output.fit_transform(test_target)
    
    # finally return the data arrays
    return train_input, test_input, train_target, test_target

# Adam optimizer
def get_optimizer():
    from keras.optimizers import Adam
    return Adam(lr=0.0002, beta_1=0.5)

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

def get_discriminator(optimizer):

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout
    from keras.layers.advanced_activations import LeakyReLU
    from keras import initializers
    
    discriminator = Sequential()

    discriminator.add(Dense( 2*output_dim, input_dim=output_dim,
                             activation='relu'))
    #kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    #discriminator.add(Dropout(0.3))
    #discriminator.add(Dropout(0.1))

    discriminator.add(Dense(3*output_dim))
    discriminator.add(LeakyReLU(0.2))
    #discriminator.add(Dropout(0.1))

    discriminator.add(Dense(2*output_dim))
    discriminator.add(LeakyReLU(0.2))
    #discriminator.add(Dropout(0.1))

    #discriminator.add(LeakyReLU(0.2))
    #discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
    gan.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'] )

    gan.summary()

    return gan

def plotDimensions( nvars ):
    # work out the dimensions to use when plotting
    import math
    iy = ix = int(math.sqrt(nvars))
    if ix*iy < nvars : ix = ix+1
    return (ix,iy)

def plots( title, names, values, dir = 'plots/' ):

    import matplotlib.pyplot as plt
    import os

    nvars = len(names)
    if nvars > 0 :

        print( "Producing", title, "plots in", dir )
    
        if not os.path.exists(dir) : os.makedirs(dir)

        ix, iy = plotDimensions(nvars)

        plt.figure(figsize=(18,15))
        for i in range(0,len(names)) :

            #print( "Plotting", i, names[i] )
            plt.subplot(ix, iy, i+1)
            plt.hist( values[:,i], 50, histtype='bar' )
            plt.grid(True)
            plt.title( names[i] )

        plt.tight_layout()
        plt.savefig(dir+title+'.png')
        plt.close()

def outputCorrs( title, generated_out, true_out, names, dir = 'plots/' ):

    import matplotlib.pyplot as plt

    nvars = len(names)
    if nvars > 0 :

        ix, iy = plotDimensions(nvars)

        plt.figure(figsize=(15,15))
        for i in range(0,len(names)) :
            
            plt.subplot(ix, iy, i+1)
            plt.hist2d( generated_out[:,i], true_out[:,i], 50 )
            plt.grid(True)
            plt.title( names[i] )
            plt.xlabel('Generated')
            plt.ylabel('Target')
            plt.colorbar()
    
        plt.tight_layout()
        plt.savefig(dir+title+'.png')
        plt.close()

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
    train_input, test_input, train_target, test_target = createLHCbData(maxData)

    # plots of the target output
    plots( "output_raw", target_names, train_target, plots_dir )

    # make some plots of the raw inputs
    plots( "inputs_raw", train_names, train_input, plots_dir )

    # normalise the data
    train_input, test_input, train_target, test_target = normalise( train_input, 
                                                                    test_input, 
                                                                    train_target,
                                                                    test_target )

    # plots of the target normalised output
    plots( "output_norm", target_names, train_target, plots_dir )

    # make some plots of the normalised inputs
    plots( "inputs_norm", train_names, train_input, plots_dir )

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
            generated_output = generator.predict( getTrainBatch(train_input,batch_i) )

            # target output values for this batch
            batch_target = train_target[batch_i]

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
            batch_i = np.random.randint( 0, train_input.shape[0], size=batch_size ) 

            # Train generator
            discriminator.trainable = False
            gan.train_on_batch( getTrainBatch(train_input,batch_i), np.ones(batch_size) )

        # evaluate the generator on training data
        train_noise = np.random.normal( 0, 1, size=[train_input.shape[0],n_noise_data] )
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
         
            plots( "gen_norm_outputs", target_names, outputs_test, epoch_dir )

            # output correlations
            outputCorrs( "correlations", outputs_test, test_target, target_names, epoch_dir )
        
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

    train( epochs, b_per_e, batch_size, maxData )

    # testing
    #train( 50, 100, 128, 1000 )
