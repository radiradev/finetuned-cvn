"""
DUNE CVN test module.
"""
__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve, Leigh Howard Whitehead'
__email__ = "saul.alonso.monsalve@cern.ch, leigh.howard.whitehead@cern.ch"

import shutil
import numpy as np
import pickle as pk
import sys
import os

sys.path.append(os.path.join(sys.path[0], 'modules'))

from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix
from data_generator import DataGenerator
from opts import get_args

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

args = get_args()

def test():
    ''' Test the DUNE CVN on input dataset.
    '''
    # parameters
    test_values = []
    TEST_PARAMS = {'batch_size':args.batch_size,
                   'images_path':args.dataset,
                   'shuffle':args.shuffle,
                   'test_values':test_values}


    # Load dataset
    print('Reading dataset from serialized file...')
    with open('dataset/partition.p', 'rb') as partition_file:
        IDs, labels = pk.load(partition_file)

    # Print some dataset statistics
    print('Number of test examples: %d', len(IDs))

    prediction_generator = DataGenerator(**TEST_PARAMS).generate(labels, IDs)

    # Load model
    print('Loading model from disk...')
    with open('saved_model/model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('saved_model/weights.h5')
    if(args.print_model):
        model.summary()

    print('Performing test...')
    is_antinu_labels = ['nu', 'antinu']
    flav_labels = ['CC Numu', 'CC Nue', 'CC Nutau', 'NC']
    inte_labels = ['CC QE', 'CC Res', 'CC DIS', 'CC other']
    cate_labels = ['CC QE Numu', 'CC Res Numu', 'CC DIS Numu', 'CC other Numu',\
                   'CC QE Nue', 'CC Res Nue', 'CC DIS Nue', 'CC other Nue',\
                   'CC QE Nutau', 'CC Res Nutau', 'CC DIS Nutau', 'CC other Nutau',\
                   'NC']
    prot_labels = ['0 protons', '1 protons', '2 protons', '>2 protons']
    chpi_labels = ['0 ch. pions', '1 ch. pions', '2 ch. pions', '>2 ch. pions']
    nepi_labels = ['0 neu. pions', '1 neu. pions', '2 neu. pions', '>2 neu. pions']
    neut_labels = ['0 neutrons', '1 neutrons', '2 neutrons', '>2 neutrons']

    # Predict results
    Y_pred = model.predict(x = prediction_generator,
                           steps = len(IDs)//args.batch_size,
                           verbose = 1)

    test_values = np.array(test_values[0:Y_pred[0].shape[0]]) # array with y true values

    y_pred_is_antinu = np.around(Y_pred[0]).reshape((Y_pred[0].shape[0], 1)).astype(int) # 1-dim arr. of pred. values (is_antinu)
    y_pred_flav = np.argmax(Y_pred[1], axis=1).reshape((Y_pred[1].shape[0], 1))          # 1-dim arr. of pred. values (flavour)
    y_pred_inte = np.argmax(Y_pred[2], axis=1).reshape((Y_pred[2].shape[0], 1))          # 1-dim arr. of pred. values (interaction)
    y_pred_cate = np.zeros(y_pred_flav.shape, dtype=int)                                 # 1-dim arr. of pred. values (categories)
    y_pred_prot = np.argmax(Y_pred[3], axis=1).reshape((Y_pred[3].shape[0], 1))          # 1-dim arr. of pred. values (protons)
    y_pred_chpi = np.argmax(Y_pred[4], axis=1).reshape((Y_pred[4].shape[0], 1))          # 1-dim arr. of pred. values (charged pions)
    y_pred_nepi = np.argmax(Y_pred[5], axis=1).reshape((Y_pred[5].shape[0], 1))          # 1-dim arr. of pred. values (neutral pions)
    y_pred_neut = np.argmax(Y_pred[6], axis=1).reshape((Y_pred[6].shape[0], 1))          # 1-dim arr. of pred. values (neutrons)

    y_test_is_antinu = np.array([aux['y_value'][0] for aux in test_values]).reshape(y_pred_is_antinu.shape)
    y_test_flav = np.array([aux['y_value'][1] for aux in test_values]).reshape(y_pred_flav.shape)
    y_test_inte = np.array([aux['y_value'][2] for aux in test_values]).reshape(y_pred_inte.shape)
    y_test_cate = np.zeros(y_test_flav.shape, dtype=int)
    y_test_prot = np.array([aux['y_value'][3] for aux in test_values]).reshape(y_pred_prot.shape)
    y_test_chpi = np.array([aux['y_value'][4] for aux in test_values]).reshape(y_pred_chpi.shape)
    y_test_nepi = np.array([aux['y_value'][5] for aux in test_values]).reshape(y_pred_nepi.shape)
    y_test_neut = np.array([aux['y_value'][6] for aux in test_values]).reshape(y_pred_neut.shape)

    # manually set y_pred_cate and y_test_cate
    for i in range(y_pred_cate.shape[0]):
        # inter
        y_pred_cate[i] = y_pred_inte[i]
        y_test_cate[i] = y_test_inte[i]    

        # flavour
        y_pred_cate[i] += (y_pred_flav[i]*4)
        y_test_cate[i] += (y_test_flav[i]*4)

        if y_pred_flav[i] == 3:
            y_pred_is_antinu[i] = 2
            y_pred_inte[i] = 4
            y_pred_cate[i] = 12

        if y_test_flav[i] == 3:
            y_test_is_antinu[i] = 2
            y_test_inte[i] = 4
            y_test_cate[i] = 12

    with open(args.output_file, 'w') as fd:
        # is_antinu
        print('is_antinu report:\n', file=fd)
        print(classification_report(y_test_is_antinu, y_pred_is_antinu, labels=list(range(len(is_antinu_labels))),\
              target_names=is_antinu_labels), file=fd)
        print('is_antinu confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        is_antinu_conf_mat = confusion_matrix(y_pred_is_antinu, y_test_is_antinu, labels=list(range(len(is_antinu_labels))))
        print(is_antinu_conf_mat, '\n', file=fd)

        # flavour 
        print('flavour report:\n', file=fd)
        print(classification_report(y_test_flav, y_pred_flav, labels=list(range(len(flav_labels))),\
              target_names=flav_labels), file=fd)
        print('flavour confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        flav_conf_mat = confusion_matrix(y_pred_flav, y_test_flav, labels=list(range(len(flav_labels))))
        print(flav_conf_mat, '\n', file=fd)

        # interaction
        print('interaction report:\n', file=fd)
        print(classification_report(y_test_inte, y_pred_inte, labels=list(range(len(inte_labels))),\
              target_names=inte_labels), file=fd)
        print('interaction confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        inte_conf_mat = confusion_matrix(y_pred_inte, y_test_inte, labels=list(range(len(inte_labels))))
        print(inte_conf_mat, '\n', file=fd)

        # categories
        print('categories report:\n', file=fd)
        print(classification_report(y_test_cate, y_pred_cate, labels=list(range(len(cate_labels))),\
              target_names=cate_labels), file=fd)
        print('categories confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        cate_conf_mat = confusion_matrix(y_pred_cate, y_test_cate, labels=list(range(len(cate_labels))))
        print(cate_conf_mat, '\n', file=fd)

        # protons
        print('protons report:\n', file=fd)
        print(classification_report(y_test_prot, y_pred_prot, labels=list(range(len(prot_labels))),\
              target_names=prot_labels), file=fd)
        print('protons confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        prot_conf_mat = confusion_matrix(y_pred_prot, y_test_prot, labels=list(range(len(prot_labels))))
        print(prot_conf_mat, '\n', file=fd)

        # charged pions
        print('charged pions report:\n', file=fd)
        print(classification_report(y_test_chpi, y_pred_chpi, labels=list(range(len(chpi_labels))),\
              target_names=chpi_labels), file=fd)
        print('charged pions confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        chpi_conf_mat = confusion_matrix(y_pred_chpi, y_test_chpi, labels=list(range(len(chpi_labels))))
        print(chpi_conf_mat, '\n', file=fd)

        # neutral pions
        print('neutral pions report:\n', file=fd)
        print(classification_report(y_test_nepi, y_pred_nepi, labels=list(range(len(nepi_labels))),\
              target_names=nepi_labels), file=fd)
        print('neutral pions confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        nepi_conf_mat = confusion_matrix(y_pred_nepi, y_test_nepi, labels=list(range(len(nepi_labels))))
        print(nepi_conf_mat, '\n', file=fd)

        # neutrons
        print('neutrons report:\n', file=fd)
        print(classification_report(y_test_neut, y_pred_neut, labels=list(range(len(neut_labels))),\
              target_names=neut_labels), file=fd)
        print('neutrons confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        neut_conf_mat = confusion_matrix(y_pred_neut, y_test_neut, labels=list(range(len(neut_labels))))
        print(neut_conf_mat, file=fd)

if __name__ == '__main__':
    test()
