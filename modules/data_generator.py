"""
DUNE CVN generator module.
"""
__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve, Leigh Howard Whitehead'
__email__ = "saul.alonso.monsalve@cern.ch, leigh.howard.whitehead@cern.ch"

import numpy as np
import zlib

class DataGenerator(object):
    ''' Generate data for tf.keras.
    '''

    def __init__(self, cells=500, planes=500, views=3, batch_size=32,
                 images_path = 'dataset', shuffle=True, test_values=[]):
        ''' Constructor.

        Args:
            cells: image cells.
            planes: image planes.
            views: number of views.
            batch_size: batch size.
            images_path: path of input events.
            shuffle: shuffle the events.
            test_values: array to be filled with test values.
        '''
        self.cells = cells
        self.planes = planes
        self.views = views
        self.batch_size = batch_size
        self.images_path = images_path
        self.shuffle = shuffle
        self.test_values = test_values
 
    def generate(self, labels, list_IDs):
        ''' Generates batches of samples.
        
        Args:
            labels: event labels.
            list_IDs: event IDs within partition.

        Yields: a batch of events.
        '''
        # infinite loop
        while 1:
            # generate random order of exploration of dataset (to make each epoch different)
            indexes = self.get_exploration_order(list_IDs)
            # generate batches
            imax = int(len(indexes)/self.batch_size) # number of batches
            for i in range(imax):
                 # find list of IDs for one batch
                 list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                 # generate data
                 X = self.data_generation(labels, list_IDs_temp)

                 yield X

    def get_exploration_order(self, list_IDs):
        ''' Generates order of exploration.

        Args:
            list_IDs: event IDs within partition.

        Returns: random order of exploration.
        '''
        # find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def data_generation(self, labels, list_IDs_temp):
        ''' Generates data of batch_size sample.

        Args:
            labels: event labels.
            list_IDs: event IDs within partition.

        Returns: a batch of events.
        '''
        X = [None]*self.views
        for view in range(self.views):
            X[view] = np.empty((self.batch_size, self.planes, self.cells, 1), dtype='float32')

        # generate data
        for i, ID in enumerate(list_IDs_temp):
            # decompress images into pixel numpy array
            with open('dataset/event' + ID + '.gz', 'rb') as image_file:
                pixels = np.fromstring(zlib.decompress(image_file.read()), dtype=np.uint8, sep='')
                pixels = pixels.reshape(self.views, self.planes, self.cells)
            # store volume
            for view in range(self.views):
                X[view][i, :, :, :] = pixels[view, :, :].reshape(self.planes, self.cells, 1)
            # get y label
            y_value = labels[ID]
            # store actual y label
            self.test_values.append(y_value)

        return X
