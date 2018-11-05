#! /usr/bin/env python3

# Based on
# https://www.datacamp.com/community/tutorials/generative-adversarial-networks

import os
import numpy as np

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

# The dimension of our random noise vector.
random_dim = 100

def load_minst_data():
    from keras.datasets import mnist

    # load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    x_train = x_train.reshape( x_train.shape[0], 784)
    x_test  =  x_test.reshape( x_test.shape[0],  784)

    print( "Training : ", x_train.shape )
    print( "Testing  : ", x_test.shape )

    return (x_train, y_train, x_test, y_test)

# You will use the Adam optimizer
def get_optimizer():
    from keras.optimizers import Adam
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.layers.advanced_activations import LeakyReLU
    from keras import initializers
    
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return generator

def get_discriminator(optimizer):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout
    from keras.layers.advanced_activations import LeakyReLU
    from keras import initializers
    
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return discriminator

def get_gan_network(discriminator, random_dim, generator, optimizer):
    from keras.models import Model
    from keras.layers import Input

    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False

    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))

    # the output of the generator (an image)
    x = generator(gan_input)

    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)

    # create the gan
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return gan

# Create a wall of generated MNIST images
def plot_generated_images(epoch, generator, dir, examples=100, dim=(10, 10), figsize=(10, 10) ):
    import matplotlib.pyplot as plt
    
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(dir+'gan_generated_image.png')
    plt.close()

def train(epochs=1, batch_size=128):

    import os, shutil, platform
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    dir = "plots/"+platform.node()+"/MINSTGan/"
    if     os.path.exists(dir) : shutil.rmtree(dir) 
    if not os.path.exists(dir) : os.makedirs(dir)

    # Get the training and testing data
    x_train, y_train, x_test, y_test = load_minst_data()
    # Split the training data into batches of size 128
    batch_count = int( x_train.shape[0] / batch_size )

    # Build our GAN netowrk
    adam          = get_optimizer()
    generator     = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan           = get_gan_network(discriminator, random_dim, generator, adam)

    test_loss   = [ ]
    test_acc    = [ ]
    train_loss  = [ ]
    train_acc   = [ ]

    run_epochs = [ ]

    for e in range(1, epochs+1):

        run_epochs += [ e ]
        print ( '-'*15, 'Epoch %d' % e, '-'*15 )

        for _ in tqdm(range(batch_count)):

            # Get a random set of input noise and images
            noise       = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            discriminator.trainable = False
            gan.train_on_batch(noise, np.ones(batch_size))


        # evaluate the generator on training input
        noise = np.random.normal(0, 1, size=[x_train.shape[0], random_dim])
        loss_acc = generator.evaluate( noise, x_train, batch_size )
        train_loss += [ loss_acc[0] ]
        train_acc  += [ loss_acc[1] ]

        # evaluate the generator on test input
        noise = np.random.normal(0, 1, size=[x_test.shape[0], random_dim])
        loss_acc = generator.evaluate( noise, x_test, batch_size )
        test_loss += [ loss_acc[0] ]
        test_acc  += [ loss_acc[1] ]

        if e == 1 or e % 5 == 0:

            e_dir = dir+"epochs/"+str( '%06d' % e )+"/"
            if not os.path.exists(e_dir) : os.makedirs(e_dir)

            plot_generated_images(e, generator, e_dir)

            # loss plots
            plt.figure(figsize=(18,15))
            plt.plot( run_epochs, train_loss, 'bo', label='Training Loss')
            plt.plot( run_epochs, test_loss,   'b', label='Validation Loss')
            plt.title('Training and validation loss')
            plt.legend()
            plt.savefig(e_dir+'loss.png')
            plt.close()

            # accuracy plots
            plt.figure(figsize=(18,15))
            plt.plot( run_epochs, train_acc, 'bo', label='Training Accuracy')
            plt.plot( run_epochs, test_acc,   'b', label='Validation Accuracy')
            plt.title('Training and validation accuracy')
            plt.legend()
            plt.savefig(e_dir+'accuracy.png')
            plt.close()

            disc_bins = 100

            # generated output
            noise = np.random.normal(0, 1, size=[x_train.shape[0], random_dim])
            x_gen = generator.predict( noise )

            # Discriminator output
            disc_train = discriminator.predict( x_train )
            disc_gen   = discriminator.predict( x_gen )
            plt.figure(figsize=(18,15))
            data = np.concatenate( [disc_train,disc_gen], axis=1 ) 
            plt.hist( data, disc_bins, density=True, histtype='step', label=['Target','Generated'] )
            plt.title('Discriminator Output')
            plt.legend()
            plt.savefig(e_dir+'discriminator.png')
            plt.close()


if __name__ == '__main__':
    train(400, 128)

