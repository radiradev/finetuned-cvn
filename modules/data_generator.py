"""
DUNE CVN generator module.
"""
__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve, Leigh Howard Whitehead'
__email__ = "saul.alonso.monsalve@cern.ch, leigh.howard.whitehead@cern.ch"

import numpy as np
import zlib

class DataGenerator(object):
    'Generates data for tf.keras'

    '''
    Initialization function of the class
    '''
    def __init__(self, cells=500, planes=500, views=3, batch_size=32,
                 images_path = 'dataset', shuffle=True, test_values=[]):
        'Initialization'
        self.cells = cells
        self.planes = planes
        self.views = views
        self.batch_size = batch_size
        self.images_path = images_path
        self.shuffle = shuffle
        self.test_values = test_values
 
    '''
    Goes through the dataset and outputs one batch at a time.
    ''' 
    def generate(self, labels, list_IDs):
        'Generates batches of samples'

        # Infinite loop
        while 1:
            # Generate random order of exploration of dataset (to make each epoch different)
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes)/self.batch_size) # number of batches

            for i in range(imax):
                 # Find list of IDs for one batch
                 list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                 # Generate data
                 X = self.__data_generation(labels, list_IDs_temp)

                 yield X

    '''
    Generates a random order of exploration for a given set of list_IDs. 
    If activated, this feature will shuffle the order in which the examples 
    are fed to the classifier so that batches between epochs do not look alike. 
    Doing so will eventually make our model more robust.
    '''
    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'

        # Find exploration order
        indexes = np.arange(len(list_IDs))

        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    '''
    Outputs batches of data and only needs to know about the list of IDs included 
    in batches as well as their corresponding labels.
    '''
    def __data_generation(self, labels, list_IDs_temp):
        'Generates data of batch_size samples'

        X = [None]*self.views

        for view in range(self.views):
            X[view] = np.empty((self.batch_size, self.planes, self.cells, 1), dtype='float32')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Decompress images into pixel NumPy tensor
            with open('dataset/event' + ID + '.gz', 'rb') as image_file:
                pixels = np.fromstring(zlib.decompress(image_file.read()), dtype=np.uint8, sep='').reshape(self.views, self.planes, self.cells)

            # Store volume
            for view in range(self.views):
                X[view][i, :, :, :] = pixels[view, :, :].reshape(self.planes, self.cells, 1)

            # get y value
            y_value = labels[ID]

            # store actual label
            self.test_values.append({'y_value':y_value})

        return X
