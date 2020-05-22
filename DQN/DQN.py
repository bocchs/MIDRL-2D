#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Amir Alansary <amiralansary@gmail.com>
# Modified: Alex Bocchieri <abocchi2@jhu.edu>

# for importing from tensorpack_medical
import sys
sys.path.append('../../../../rl-medical')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)

import numpy as np

import os
import sys
import time
import argparse
from collections import deque

import tensorflow as tf
from medical import MedicalPlayer, FrameStack
from tensorpack.input_source import QueueInput
from tensorpack.models.conv2d import Conv2D
from tensorpack_medical.models.pool2d import MaxPooling2D
from common import Evaluator, eval_model_multithread, play_n_episodes
from DQNModel import Model2D as DQNModel
from expreplay import ExpReplay

from tensorpack import (PredictConfig, OfflinePredictor, get_model_loader,
                        logger, TrainConfig, ModelSaver, PeriodicTrigger,
                        ScheduledHyperParamSetter, ObjAttrParam,
                        HumanHyperParamSetter, argscope, RunOp, LinearWrap,
                        FullyConnected, PReLU, SimpleTrainer,
                        launch_train_with_config)


###############################################################################
# BATCH SIZE USED IN NATURE PAPER IS 32 - MEDICAL IS 256
BATCH_SIZE = 48
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
IMAGE_SIZE = (45, 45)
# how many frames to keep
# in other words, how many observations the network can see
FRAME_HISTORY = 4
# the frequency of updating the target network
UPDATE_FREQ = 4
# DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
GAMMA = 0.9 #0.99
# REPLAY MEMORY SIZE - NATURE (1e6) - MEDICAL (1e5 view-patches)
MEMORY_SIZE = 1e5#6
# consume at least 1e6 * 27 * 27 * 27 bytes
INIT_MEMORY_SIZE = MEMORY_SIZE // 20 #5e4
# each epoch is 100k played frames
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10
# num training epochs in between model evaluations
EPOCHS_PER_EVAL = 2
# the number of episodes to run during evaluation
EVAL_EPISODE = 50

###############################################################################

def get_player(directory=None, files_list= None, viz=False,
               task='play', saveGif=False, saveVideo=False):
    # in atari paper, max_num_frames = 30000
    ####################################################
    if task == 'train':
        max_frames = 1500
    else:
        max_frames = 100 # max num frames in case model does not converge
    ####################################################
    env = MedicalPlayer(directory=directory, screen_dims=IMAGE_SIZE,
                        viz=viz, saveGif=saveGif, saveVideo=saveVideo,
                        task=task, files_list=files_list, max_num_frames=max_frames)
    if (task != 'train'):
        # in training, env will be decorated by ExpReplay, and history
        # is taken care of in expreplay buffer
        # otherwise, FrameStack modifies self.step to save observations into a queue
        env = FrameStack(env, FRAME_HISTORY)
    return env

###############################################################################

class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]

        :returns predicted Q values"""
        # normalize image values to [0, 1]
        image = image / 255.0

        with argscope(Conv2D, nl=PReLU.symbolic_function, use_bias=True):
            # core layers of the network
            conv = (LinearWrap(image)
                 .Conv2D('conv0', out_channel=32,
                         kernel_shape=[5,5], stride=[1,1])
                 .MaxPooling2D('pool0',2)
                 .Conv2D('conv1', out_channel=32,
                         kernel_shape=[5,5], stride=[1,1])
                 .MaxPooling2D('pool1',2)
                 .Conv2D('conv2', out_channel=64,
                         kernel_shape=[4,4], stride=[1,1])
                 .MaxPooling2D('pool2',2)
                 .Conv2D('conv3', out_channel=64,
                         kernel_shape=[3,3], stride=[1,1])
                 # .MaxPooling2D('pool3',2)
                 )

        if 'Dueling' not in self.method:
            lq = (conv
                 .FullyConnected('fc0', 512).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc1', 256).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc2', 128).tf.nn.leaky_relu(alpha=0.01)())
            Q = FullyConnected('fct', lq, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN or Double Dueling
            # state value function
            lv = (conv
                 .FullyConnected('fc0V', 512).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc1V', 256).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc2V', 128).tf.nn.leaky_relu(alpha=0.01)())
            V = FullyConnected('fctV', lv, 1, nl=tf.identity)
            # advantage value function
            la = (conv
                 .FullyConnected('fc0A', 512).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc1A', 256).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc2A', 128).tf.nn.leaky_relu(alpha=0.01)())
            As = FullyConnected('fctA', la, self.num_actions, nl=tf.identity)

            Q = tf.add(As, V - tf.reduce_mean(As, 1, keepdims=True))

        return tf.identity(Q, name='Qvalue')


###############################################################################

def get_config(files_list):
    """This is only used during training."""
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(task='train', files_list=files_list),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY
    )

    return TrainConfig(
        # dataflow=expreplay,
        data=QueueInput(expreplay),
        model=Model(),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                # update target network every 10k steps
                every_k_steps=10000 // UPDATE_FREQ),
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(60, 4e-4), (100, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                # 1->0.1 in the first million steps
                [(0, 1), (10, 0.1), (320, 0.01)],
                interp='linear'),
            PeriodicTrigger(
                Evaluator(nr_eval=EVAL_EPISODE, input_names=['state'],
                          output_names=['Qvalue'], files_list=files_list,
                          get_player_fn=get_player),
                every_k_epochs=EPOCHS_PER_EVAL),
            HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=20, # originally 1000
    )



###############################################################################
###############################################################################


def main(cmdline=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform. Must load a pretrained model if task is "play" or "eval"',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling','DuelingDouble'],
                        default='DQN')
    if cmdline:
        parser.add_argument('--files', help='single cmdline', default='')
    else:
        parser.add_argument('--files', type=argparse.FileType('r'), nargs='+',
                        help="""Filepath to the text file that comtains list of images.
                                Each line of this file is a full path to an image scan.
                                For (task == train or eval) there should be two input files ['images', 'landmarks']""")
    parser.add_argument('--saveGif', help='save gif image of the game',
                        action='store_true', default=False)
    parser.add_argument('--saveVideo', help='save video of the game',
                        action='store_true', default=False)
    parser.add_argument('--logDir', help='store logs in this directory during training',
                        default='train_log')
    parser.add_argument('--name', help='name of current experiment for logs',
                        default='experiment_1')


    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not cmdline:
        # check input files
        if args.task == 'play':
            error_message = """Wrong input files {} for {} task - should be 1 \'images.txt\' """.format(len(args.files), args.task)
            assert len(args.files) == 1
        else:
            error_message = """Wrong input files {} for {} task - should be 2 [\'images.txt\', \'landmarks.txt\'] """.format(len(args.files), args.task)
            assert len(args.files) == 2, (error_message)

    global METHOD
    METHOD = args.algo
    if cmdline:
        init_player = MedicalPlayer(files_list=args.files,
                                screen_dims=IMAGE_SIZE,
                                task='play')
    else:
        # load files into env to set num_actions, num_validation_files
        init_player = MedicalPlayer(files_list=args.files,
                                    screen_dims=IMAGE_SIZE,
                                    task='play')
    global NUM_ACTIONS
    NUM_ACTIONS = init_player.action_space.n
    num_files = init_player.files.num_files

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue']))

        if cmdline:
            if args.task == 'play':
                locs = play_n_episodes(get_player(files_list=args.files, viz=0.01,
                                           saveGif=args.saveGif,
                                           saveVideo=args.saveVideo,
                                           task='play'),
                                pred, 1) # 1 file from cmd line
                return locs[0][2] # return slice number

        # demo pretrained model one episode at a time
        if args.task == 'play':
            locs = play_n_episodes(get_player(files_list=args.files, viz=0.01,
                                       saveGif=args.saveGif,
                                       saveVideo=args.saveVideo,
                                       task='play'),
                            pred, num_files)
            if num_files == 1:
                return locs[0][2] # return slice number
            # for loc in locs:
            #     print("final location slice number = " + str(loc[2]))
        # run episodes in parallel and evaluate pretrained model
        elif args.task == 'eval':
            locs = play_n_episodes(get_player(files_list=args.files, viz=0.01,
                                       saveGif=args.saveGif,
                                       saveVideo=args.saveVideo,
                                       task='eval'),
                            pred, num_files)
            # for loc in locs:
            #     print("final location slice number = " + str(loc[2]))
    else:  # train model
        logger_dir = os.path.join(args.logDir, args.name)
        logger.set_logger_dir(logger_dir)
        config = get_config(args.files)
        if args.load:  # resume training from a saved checkpoint
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
    
    return -1


if __name__ == '__main__':
    cmdline = False # false if training, true if actually using for a volume
    slice_num = main(cmdline)
    print(slice_num)
