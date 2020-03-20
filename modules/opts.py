'''
DUNE CVN args module
'''
__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve, Leigh Howard Whitehead'
__email__ = "saul.alonso.monsalve@cern.ch, leigh.howard.whitehead@cern.ch"

from argparse import ArgumentParser

def get_args():
    # training related
    parser = ArgumentParser(description='DUNE CVN')
    arg = parser.add_argument
    arg('--batch_size', type=int, default=10, help='batch size')
    arg('--model',  type=str, default='saved_model/model.json', help='JSON model path')
    arg('--weights',  type=str, default='saved_model/weights.h5', help='HDF5 pretrained model weights')
    arg('--dataset',type=str, default='dataset', help='Dataset path')
    arg('--partition',type=str, default='dataset/partition.p', help='Pickle partition path')
    arg('--shuffle',type=bool, default=False, help='Shuffle partition')
    arg('--print_model',type=bool, default=False, help='Print model summary')
    arg('--output_file',type=str, default='output/results.txt', help='Output file path')

    args = parser.parse_args()
    return args
