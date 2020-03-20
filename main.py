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
os.environ["CUDA_VISIBLE_DEVICES"]="5"

args = get_args()

def test():
    '''
    ****************************************
    *************** DATASET ****************
    ****************************************
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


    '''
    ****************************************
    ************** GENERATOR ***************
    ****************************************
    '''
    prediction_generator = DataGenerator(**TEST_PARAMS).generate(labels, IDs)


    '''
    ****************************************
    ************** LOAD MODEL **************
    ****************************************
    '''
    # Load model
    print('Loading model from disk...')
    with open('saved_model/model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('saved_model/weights.h5')
    if(args.print_model):
        model.summary()

    '''
    ****************************************
    ***************** TEST *****************
    ****************************************
    '''
    print('Performing test...')
    is_antineutrino_target_names = ['neutrino', 'antineutrino']
    flavour_target_names = ['CC Numu', 'CC Nue', 'CC Nutau', 'NC']
    interaction_target_names = ['CC QE', 'CC Res', 'CC DIS', 'CC Other']
    categories_target_names = ['category 0', 'category 1', 'category 2', 'category 3', 'category 4', 'category 5', 'category 6', 
                               'category 7', 'category 8', 'category 9', 'category 10', 'category 11', 'category 13']
    protons_target_names = ['0', '1', '2', '>2']
    pions_target_names = ['0', '1', '2', '>2']
    pizeros_target_names = ['0', '1', '2', '>2']
    neutrons_target_names = ['0', '1', '2', '>2']

    # Predict results
    Y_pred = model.predict(x = prediction_generator,
                           steps = len(IDs)//args.batch_size,
                           verbose = 1)

    test_values = np.array(test_values[0:Y_pred[0].shape[0]]) # array with y true values

    y_pred_is_antineutrino = np.around(Y_pred[0]).reshape((Y_pred[0].shape[0], 1)).astype(int) # 1-DIM array of predicted values (is_antineutrino)
    y_pred_flavour = np.argmax(Y_pred[1], axis=1).reshape((Y_pred[1].shape[0], 1))             # 1-DIM array of predicted values (flavour)
    y_pred_interaction = np.argmax(Y_pred[2], axis=1).reshape((Y_pred[2].shape[0], 1))         # 1-DIM array of predicted values (interaction)
    y_pred_categories = np.zeros(y_pred_flavour.shape, dtype=int)                              # 1-DIM array of predicted values (categories)
    y_pred_protons = np.argmax(Y_pred[3], axis=1).reshape((Y_pred[3].shape[0], 1))             # 1-DIM array of predicted values (protons)
    y_pred_pions = np.argmax(Y_pred[4], axis=1).reshape((Y_pred[4].shape[0], 1))               # 1-DIM array of predicted values (pions)
    y_pred_pizeros = np.argmax(Y_pred[5], axis=1).reshape((Y_pred[5].shape[0], 1))             # 1-DIM array of predicted values (pizeros)
    y_pred_neutrons = np.argmax(Y_pred[6], axis=1).reshape((Y_pred[6].shape[0], 1))            # 1-DIM array of predicted values (neutrons)

    y_test_is_antineutrino = np.array([aux['y_value'][0] for aux in test_values]).reshape(y_pred_is_antineutrino.shape)
    y_test_flavour = np.array([aux['y_value'][1] for aux in test_values]).reshape(y_pred_flavour.shape)
    y_test_interaction = np.array([aux['y_value'][2] for aux in test_values]).reshape(y_pred_interaction.shape)
    y_test_categories = np.zeros(y_test_flavour.shape, dtype=int)
    y_test_protons = np.array([aux['y_value'][3] for aux in test_values]).reshape(y_pred_protons.shape)
    y_test_pions = np.array([aux['y_value'][4] for aux in test_values]).reshape(y_pred_pions.shape)
    y_test_pizeros = np.array([aux['y_value'][5] for aux in test_values]).reshape(y_pred_pizeros.shape)
    y_test_neutrons = np.array([aux['y_value'][6] for aux in test_values]).reshape(y_pred_neutrons.shape)

    # manually set y_pred_categories and y_test_categories
    for i in range(y_pred_categories.shape[0]):
        # inter
        y_pred_categories[i] = y_pred_interaction[i]
        y_test_categories[i] = y_test_interaction[i]    

        # flavour
        y_pred_categories[i] += (y_pred_flavour[i]*4)
        y_test_categories[i] += (y_test_flavour[i]*4)

        if y_pred_flavour[i] == 3:
            y_pred_is_antineutrino[i] = 2
            y_pred_interaction[i] = 4
            y_pred_categories[i] = 12

        if y_test_flavour[i] == 3:
            y_test_is_antineutrino[i] = 2
            y_test_interaction[i] = 4
            y_test_categories[i] = 12

    with open(args.output_file, 'w') as fd:
        # is_antineutrino
        print('is_antineutrino report:\n', file=fd)
        print(classification_report(y_test_is_antineutrino, y_pred_is_antineutrino, labels=list(range(len(is_antineutrino_target_names))),\
              target_names=is_antineutrino_target_names), file=fd)
        print('is_antineutrino confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        is_antineutrino_conf_matrix = confusion_matrix(y_pred_is_antineutrino, y_test_is_antineutrino, labels=list(range(len(is_antineutrino_target_names))))
        print(is_antineutrino_conf_matrix, '\n', file=fd)

        # flavour 
        print('flavour report:\n', file=fd)
        print(classification_report(y_test_flavour, y_pred_flavour, labels=list(range(len(flavour_target_names))),\
              target_names=flavour_target_names), file=fd)
        print('flavour confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        flavour_conf_matrix = confusion_matrix(y_pred_flavour, y_test_flavour, labels=list(range(len(flavour_target_names))))
        print(flavour_conf_matrix, '\n', file=fd)

        # interaction
        print('interaction report:\n', file=fd)
        print(classification_report(y_test_interaction, y_pred_interaction, labels=list(range(len(interaction_target_names))),\
              target_names=interaction_target_names), file=fd)
        print('interaction confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        interaction_conf_matrix = confusion_matrix(y_pred_interaction, y_test_interaction, labels=list(range(len(interaction_target_names))))
        print(interaction_conf_matrix, '\n', file=fd)

        # categories
        print('categories report:\n', file=fd)
        print(classification_report(y_test_categories, y_pred_categories, labels=list(range(len(categories_target_names))),\
              target_names=categories_target_names), file=fd)
        print('categories confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        categories_conf_matrix = confusion_matrix(y_pred_categories, y_test_categories, labels=list(range(len(categories_target_names))))
        print(categories_conf_matrix, '\n', file=fd)

        # protons
        print('protons report:\n', file=fd)
        print(classification_report(y_test_protons, y_pred_protons, labels=list(range(len(protons_target_names))),\
              target_names=protons_target_names), file=fd)
        print('protons confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        protons_conf_matrix = confusion_matrix(y_pred_protons, y_test_protons, labels=list(range(len(protons_target_names))))
        print(protons_conf_matrix, '\n', file=fd)

        # pions
        print('charged pions report:\n', file=fd)
        print(classification_report(y_test_pions, y_pred_pions, labels=list(range(len(pions_target_names))),\
              target_names=pions_target_names), file=fd)
        print('charged pions confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        pions_conf_matrix = confusion_matrix(y_pred_pions, y_test_pions, labels=list(range(len(pions_target_names))))
        print(pions_conf_matrix, '\n', file=fd)

        # pizeros
        print('neutral pions report:\n', file=fd)
        print(classification_report(y_test_pizeros, y_pred_pizeros, labels=list(range(len(pizeros_target_names))),\
              target_names=pizeros_target_names), file=fd)
        print('neutral pions confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        pizeros_conf_matrix = confusion_matrix(y_pred_pizeros, y_test_pizeros, labels=list(range(len(pizeros_target_names))))
        print(pizeros_conf_matrix, '\n', file=fd)

        # neutrons
        print('neutrons report:\n', file=fd)
        print(classification_report(y_test_neutrons, y_pred_neutrons, labels=list(range(len(neutrons_target_names))),\
              target_names=neutrons_target_names), file=fd)
        print('neutrons confusion matrix (rows = predicted classes, cols = actual classes):\n', file=fd)
        neutrons_conf_matrix = confusion_matrix(y_pred_neutrons, y_test_neutrons, labels=list(range(len(neutrons_target_names))))
        print(neutrons_conf_matrix, file=fd)

if __name__ == '__main__':
    test()
