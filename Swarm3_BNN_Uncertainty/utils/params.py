# Copyright 2020, Prof. Marko Orescanin, NPS
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Created by marko.orescanin@nps.edu on 7/21/20

"""params.py

This module contains all parameter handling for the project, including
all command line tunable parameter definitions, any preprocessing
of those parameters, or generation/specification of other parameters.

"""
import argparse
import os


def make_argparser():
    parser = argparse.ArgumentParser(description="Arguments to run training")
    
    parser.add_argument("--mode", type=str, default="predict", help="train or predict")

    parser.add_argument("--trained_model", type=str, help="full path to previously trained 'model.keras' to use for predictions")
    parser.add_argument("--data_folder", type=str, default="/home/donald.peltier/swarm/data", help="folder for npz files")
    parser.add_argument("--mean_var_NN_DS", type=str, default=None, help="Path to mean/var file for dataset used to train NN in order to rescale current dataset")
    parser.add_argument("--mean_var_current_DS", type=str, default=None, help="Path to mean/var file for current dataset")
    
    parser.add_argument("--class_names", type=str, nargs='+',
                        default=['Greedy', 'Greedy+', 'Auction', 'Auction+'],
                        help="List of class names. Provide multiple names separated by spaces. Default: ['Greedy', 'Greedy+', 'Auction', 'Auction+']")
    parser.add_argument("--num_att", type=int, default=10, help="number of attackers (killers)")
    parser.add_argument("--num_def", type=int, default=10, help="number of defenders (decoys)")
    parser.add_argument("--motion", type=str, default="star", help="decoy motions that will map to datasets to append to data_path")
    # parser.add_argument("--real_data_path", type=str, default=None, help="location of real_data.npz file")
    parser.add_argument("--window", type=int, default=20, help="observation window; -1 uses full window")
    parser.add_argument("--features", type=str, default="pv", help="pv=position & velocity, p=position only, v=velocity only")

    parser.add_argument("--model_type", type=str, default="cn", help="fc=fully connect, cn=CNN, fcn=fully CNN, lstm=LSTM, res=ResNet, tr=transformer")
    parser.add_argument("--bnn", type=str2bool, default=False, help="Default (False) trains deterministic NN; True trains Bayesian NN")
    parser.add_argument("--num_monte_carlo", type=int, default=20, help="number of Monte Carlo predictions")
    parser.add_argument("--num_instances_visualize", type=int, default=5, help="number of each class to visualize results for")
    parser.add_argument('--tuned', type=str2bool, default=True, help="if training, use tuned parameters for selected model")
    parser.add_argument("--output_type", type=str, default="mc", help="mc=multiclass, ml=multilabel, mh=multihead")
    parser.add_argument("--output_length", type=str, default="vec", help="vec=vector (final prediction only), seq=sequence (prediction at every time step)")
    
    parser.add_argument("--dropout", type=float, default=0, help="dropout percentage 0 to 1 (ie. 0.2=20%)")
    # parser.add_argument("--kernel_initializer", type=str, default="he_normal", help="glorot_uniform/normal (default, sigmoid), he_uniform/normal (relu)")
    # parser.add_argument("--kernel_regularizer", type=str, default="none", help="l1, l2, l1_l2")

    # parser.add_argument("--optimizer", type=str, default="adam", help="SGD optimizer")
    parser.add_argument("--initial_learning_rate", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--callback_list", type=str, default="checkpoint, early_stopping, csv_log", help="comma separated callbacks to be added")
    parser.add_argument("--patience", type=int, default=50, help="training epochs with negligible improvement before training stops")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=50, help="bacth size")
    parser.add_argument("--val_split", type=float, default=0.2, help="validation holdout percentage 0 to 1 (ie. 0.2=20%)")
    parser.add_argument("--model_dir", type=str, required=True, help="folder to save all outputs and trained model")
    
    # parser.add_argument("--tune_type", type=str, help="tuner type: r=random, b=bayesian, h=hyperband")
    # parser.add_argument("--tune_epochs", type=int, help="number of epochs used in Keras Tuner")

    return parser.parse_args()


# you have to use str2bool because of an issue with argparser and bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_hparams():
    """any preprocessing, special handling of the hparams object"""
    parser = make_argparser()
    print('\n*** PARAMETERS ***')
    print(parser)
    return parser

def make_model_folder(hparams):
    if not os.path.exists(hparams.model_dir):
        os.mkdir(hparams.model_dir)

def save_hparams(hparams):
    make_model_folder(hparams)
    path_ = os.path.join(hparams.model_dir, "params.txt")
    hparams_ = vars(hparams)
    with open(path_, "w") as f:
        for arg in hparams_:
            print(arg, ":", hparams_[arg])
            f.write(arg + ":" + str(hparams_[arg]) + "\n")

    ## Commented out as don't require "params.yml" 
    # path_ = os.path.join(hparams.model_dir, "params.yml")
    # with open(path_, "w") as f:
    #     yaml.dump(
    #         hparams_, f, default_flow_style=False
    #     )  # save hparams as a yaml, since that's easier to read and use
