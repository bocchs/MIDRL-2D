#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Modified: Amir Alansary <amiralansary@gmail.com>
# Modified: Alex Bocchieri <abocchi2@jhu.edu>

import random
import time
import threading
import numpy as np
from tqdm import tqdm
import multiprocessing
from six.moves import queue

# from tensorpack import *
# from tensorpack.utils.stats import *
from tensorpack.utils import logger
# from tensorpack.callbacks import Triggerable
from tensorpack.callbacks.base import Callback
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack.utils.concurrency import (StoppableThread, ShareSessionThread)

import traceback
import sys
import statistics
import xlwt
from xlwt import Workbook

###############################################################################

def play_one_episode(env, func, render=False):
    def predict(s):
        """
        Run a full episode, mapping observation to action, WITHOUT 0.001 greedy.
    :returns sum of rewards
        """
        # pick action with best predicted Q-value
        q_values = func(s[None, :, :, :])[0][0]
        act = q_values.argmax()

        # eps greedy disabled
        # if random.random() < 0.001:
        #     spc = env.action_space
        #     act = spc.sample()
        return act, q_values

    ob = env.reset()
    sum_r = 0
    while True:
        act, q_values = predict(ob)
        ob, r, isOver, info = env.step(act, q_values)
        # print(info)
        # print(env._location)
        # print(dir(env.env))
        # print(type(env.env))
        # print(env.env._location)
        # sys.exit()
        if render:
            env.render()
        sum_r += r
        if isOver:
            return sum_r, info['filename'], info['distErrorMM'], q_values, env.env._location, env.dice, env.overlap, env.landmark


###############################################################################

def play_n_episodes(player, predfunc, nr, render=False):
    """wraps play_one_episode, playing a single episode at a time and logs results
    used when playing demos."""
    locs = [] # list of the last locations (x,y,z)
    logger.info("Start Playing ... ")
    wb = Workbook()
    heart_sheet = wb.add_sheet('heart')
    kidney_sheet = wb.add_sheet('kidney')
    troch_sheet = wb.add_sheet('trochanter')
    knee_sheet = wb.add_sheet('knee')
    pros_sheet = wb.add_sheet('prostate')
    breast_sheet = wb.add_sheet('breast')
    all_sheet = wb.add_sheet('all')

    all_sheet.write(0,0,'Image')
    all_sheet.write(0,1,'Distance Error (mm)')
    all_sheet.write(0,2,'Total Reward')
    all_sheet.write(0,3,'Dice')
    all_sheet.write(0,4,'Overlap (IoU)')
    all_row = 1
    all_dist_errors = []
    all_total_rewards = []
    all_dices = []
    all_overlaps = []


    heart_sheet.write(0,0,'Image')
    heart_sheet.write(0,1,'Distance Error (mm)')
    heart_sheet.write(0,2,'Total Reward')
    heart_sheet.write(0,3,'Dice')
    heart_sheet.write(0,4,'Overlap (IoU)')
    heart_row = 1
    heart_dist_errors = []
    heart_total_rewards = []
    heart_dices = []
    heart_overlaps = []
    heart_dist_errors_in = []
    heart_total_rewards_in = []
    heart_dices_in = []
    heart_overlaps_in = []
    heart_dist_errors_opp = []
    heart_total_rewards_opp = []
    heart_dices_opp = []
    heart_overlaps_opp = []
    heart_dist_errors_F = []
    heart_total_rewards_F = []
    heart_dices_F = []
    heart_overlaps_F = []
    heart_dist_errors_W = []
    heart_total_rewards_W = []
    heart_dices_W = []
    heart_overlaps_W = []
    heart_dist_errors_T1 = []
    heart_total_rewards_T1 = []
    heart_dices_T1 = []
    heart_overlaps_T1 = []
    heart_dist_errors_T2 = []
    heart_total_rewards_T2 = []
    heart_dices_T2 = []
    heart_overlaps_T2 = []
    

    kidney_sheet.write(0,0,'Image')
    kidney_sheet.write(0,1,'Distance Error (mm)')
    kidney_sheet.write(0,2,'Total Reward')
    kidney_sheet.write(0,3,'Dice')
    kidney_sheet.write(0,4,'Overlap (IoU)')
    kidney_row = 1
    kidney_dist_errors = []
    kidney_total_rewards = []
    kidney_dices = []
    kidney_overlaps = []

    kidney_dist_errors_in = []
    kidney_total_rewards_in = []
    kidney_dices_in = []
    kidney_overlaps_in = []

    kidney_dist_errors_opp = []
    kidney_total_rewards_opp = []
    kidney_dices_opp = []
    kidney_overlaps_opp = []

    kidney_dist_errors_F = []
    kidney_total_rewards_F = []
    kidney_dices_F = []
    kidney_overlaps_F = []

    kidney_dist_errors_W = []
    kidney_total_rewards_W = []
    kidney_dices_W = []
    kidney_overlaps_W = []

    kidney_dist_errors_T1 = []
    kidney_total_rewards_T1 = []
    kidney_dices_T1 = []
    kidney_overlaps_T1 = []

    kidney_dist_errors_T2 = []
    kidney_total_rewards_T2 = []
    kidney_dices_T2 = []
    kidney_overlaps_T2 = []



    troch_sheet.write(0,0,'Image')
    troch_sheet.write(0,1,'Distance Error (mm)')
    troch_sheet.write(0,2,'Total Reward')
    troch_sheet.write(0,3,'Dice')
    troch_sheet.write(0,4,'Overlap (IoU)')
    troch_row = 1
    troch_dist_errors = []
    troch_total_rewards = []
    troch_dices = []
    troch_overlaps = []

    troch_dist_errors_in = []
    troch_total_rewards_in = []
    troch_dices_in = []
    troch_overlaps_in = []

    troch_dist_errors_opp = []
    troch_total_rewards_opp = []
    troch_dices_opp = []
    troch_overlaps_opp = []

    troch_dist_errors_F = []
    troch_total_rewards_F = []
    troch_dices_F = []
    troch_overlaps_F = []

    troch_dist_errors_W = []
    troch_total_rewards_W = []
    troch_dices_W = []
    troch_overlaps_W = []

    troch_dist_errors_T1 = []
    troch_total_rewards_T1 = []
    troch_dices_T1 = []
    troch_overlaps_T1 = []

    troch_dist_errors_T2 = []
    troch_total_rewards_T2 = []
    troch_dices_T2 = []
    troch_overlaps_T2 = []



    knee_sheet.write(0,0,'Image')
    knee_sheet.write(0,1,'Distance Error (mm)')
    knee_sheet.write(0,2,'Total Reward')
    knee_sheet.write(0,3,'Dice')
    knee_sheet.write(0,4,'Overlap (IoU)')
    knee_row = 1
    knee_dist_errors = []
    knee_total_rewards = []
    knee_dices = []
    knee_overlaps = []

    knee_dist_errors_in = []
    knee_total_rewards_in = []
    knee_dices_in = []
    knee_overlaps_in = []

    knee_dist_errors_opp = []
    knee_total_rewards_opp = []
    knee_dices_opp = []
    knee_overlaps_opp = []

    knee_dist_errors_F = []
    knee_total_rewards_F = []
    knee_dices_F = []
    knee_overlaps_F = []

    knee_dist_errors_W = []
    knee_total_rewards_W = []
    knee_dices_W = []
    knee_overlaps_W = []

    knee_dist_errors_T1 = []
    knee_total_rewards_T1 = []
    knee_dices_T1 = []
    knee_overlaps_T1 = []

    knee_dist_errors_T2 = []
    knee_total_rewards_T2 = []
    knee_dices_T2 = []
    knee_overlaps_T2 = []

    pros_sheet.write(0,0,'Image')
    pros_sheet.write(0,1,'Distance Error (mm)')
    pros_sheet.write(0,2,'Total Reward')
    pros_sheet.write(0,3,'Dice')
    pros_sheet.write(0,4,'Overlap (IoU)')
    pros_row = 1
    pros_dist_errors = []
    pros_total_rewards = []
    pros_dices = []
    pros_overlaps = []

    pros_dist_errors_adc = []
    pros_total_rewards_adc = []
    pros_dices_adc = []
    pros_overlaps_adc = []

    pros_dist_errors_T2 = []
    pros_total_rewards_T2 = []
    pros_dices_T2 = []
    pros_overlaps_T2 = []

    breast_sheet.write(0,0,'Image')
    breast_sheet.write(0,1,'Distance Error (mm)')
    breast_sheet.write(0,2,'Total Reward')
    breast_sheet.write(0,3,'Dice')
    breast_sheet.write(0,4,'Overlap (IoU)')
    breast_row = 1
    breast_dist_errors = []
    breast_total_rewards = []
    breast_dices = []
    breast_overlaps = []

    breast_dist_errors_Post = []
    breast_total_rewards_Post = []
    breast_dices_Post = []
    breast_overlaps_Post = []

    breast_dist_errors_Pre = []
    breast_total_rewards_Pre = []
    breast_dices_Pre = []
    breast_overlaps_Pre = []

    breast_dist_errors_SUB = []
    breast_total_rewards_SUB = []
    breast_dices_SUB = []
    breast_overlaps_SUB = []

    breast_dist_errors_T1 = []
    breast_total_rewards_T1 = []
    breast_dices_T1 = []
    breast_overlaps_T1 = []

    breast_dist_errors_T2 = []
    breast_total_rewards_T2 = []
    breast_dices_T2 = []
    breast_overlaps_T2 = []


    for k in range(nr):
        # if k != 0:
        #     player.restart_episode()
        score, filename, distance_error, q_values, location, dice, overlap, landmark = play_one_episode(player,
                                                                    predfunc,
                                                                    render=render)
        logger.info(
            "{}/{} - {} - score {} - distError {} - q_values {}".format(k + 1, nr, filename, score, distance_error,
                                                                        q_values))
        locs.append(location) # return last location of each episode
        #print(landmark)

        all_sheet.write(all_row,0,filename)
        all_sheet.write(all_row,1,distance_error)
        all_sheet.write(all_row,2,score)
        all_sheet.write(all_row,3,dice)
        all_sheet.write(all_row,4,overlap)
        all_dist_errors.append(distance_error)
        all_total_rewards.append(score)
        all_dices.append(dice)
        all_overlaps.append(overlap)
        all_row += 1

        if landmark == "heart":
            heart_sheet.write(heart_row,0,filename)
            heart_sheet.write(heart_row,1,distance_error)
            heart_sheet.write(heart_row,2,score)
            heart_sheet.write(heart_row,3,dice)
            heart_sheet.write(heart_row,4,overlap)
            heart_dist_errors.append(distance_error)
            heart_total_rewards.append(score)
            heart_dices.append(dice)
            heart_overlaps.append(overlap)
            if "_in_" in filename:
                heart_dist_errors_in.append(distance_error)
                heart_total_rewards_in.append(score)
                heart_dices_in.append(dice)
                heart_overlaps_in.append(overlap)
            elif "_opp_" in filename:
                heart_dist_errors_opp.append(distance_error)
                heart_total_rewards_opp.append(score)
                heart_dices_opp.append(dice)
                heart_overlaps_opp.append(overlap)
            elif "_F_" in filename:
                heart_dist_errors_F.append(distance_error)
                heart_total_rewards_F.append(score)
                heart_dices_F.append(dice)
                heart_overlaps_F.append(overlap)
            elif "_W_" in filename:
                heart_dist_errors_W.append(distance_error)
                heart_total_rewards_W.append(score)
                heart_dices_W.append(dice)
                heart_overlaps_W.append(overlap)
            elif "_t1_" in filename:
                heart_dist_errors_T1.append(distance_error)
                heart_total_rewards_T1.append(score)
                heart_dices_T1.append(dice)
                heart_overlaps_T1.append(overlap)
            elif "_t2_" in filename:
                heart_dist_errors_T2.append(distance_error)
                heart_total_rewards_T2.append(score)
                heart_dices_T2.append(dice)
                heart_overlaps_T2.append(overlap)
            else:
                print("unknown image type, exiting...")
                sys.exit()
            heart_row += 1
        elif landmark == "kidney":
            kidney_sheet.write(kidney_row,0,filename)
            kidney_sheet.write(kidney_row,1,distance_error)
            kidney_sheet.write(kidney_row,2,score)
            kidney_sheet.write(kidney_row,3,dice)
            kidney_sheet.write(kidney_row,4,overlap)
            kidney_dist_errors.append(distance_error)
            kidney_total_rewards.append(score)
            kidney_dices.append(dice)
            kidney_overlaps.append(overlap)
            if "_in_" in filename:
                kidney_dist_errors_in.append(distance_error)
                kidney_total_rewards_in.append(score)
                kidney_dices_in.append(dice)
                kidney_overlaps_in.append(overlap)
            elif "_opp_" in filename:
                kidney_dist_errors_opp.append(distance_error)
                kidney_total_rewards_opp.append(score)
                kidney_dices_opp.append(dice)
                kidney_overlaps_opp.append(overlap)
            elif "_F_" in filename:
                kidney_dist_errors_F.append(distance_error)
                kidney_total_rewards_F.append(score)
                kidney_dices_F.append(dice)
                kidney_overlaps_F.append(overlap)
            elif "_W_" in filename:
                kidney_dist_errors_W.append(distance_error)
                kidney_total_rewards_W.append(score)
                kidney_dices_W.append(dice)
                kidney_overlaps_W.append(overlap)
            elif "_t1_" in filename:
                kidney_dist_errors_T1.append(distance_error)
                kidney_total_rewards_T1.append(score)
                kidney_dices_T1.append(dice)
                kidney_overlaps_T1.append(overlap)
            elif "_t2_" in filename:
                kidney_dist_errors_T2.append(distance_error)
                kidney_total_rewards_T2.append(score)
                kidney_dices_T2.append(dice)
                kidney_overlaps_T2.append(overlap)
            else:
                print("unknown image type, exiting...")
                sys.exit()
            kidney_row += 1
        elif landmark == "trochanter":
            troch_sheet.write(troch_row,0,filename)
            troch_sheet.write(troch_row,1,distance_error)
            troch_sheet.write(troch_row,2,score)
            troch_sheet.write(troch_row,3,dice)
            troch_sheet.write(troch_row,4,overlap)
            troch_dist_errors.append(distance_error)
            troch_total_rewards.append(score)
            troch_dices.append(dice)
            troch_overlaps.append(overlap)
            if "_in_" in filename:
                troch_dist_errors_in.append(distance_error)
                troch_total_rewards_in.append(score)
                troch_dices_in.append(dice)
                troch_overlaps_in.append(overlap)
            elif "_opp_" in filename:
                troch_dist_errors_opp.append(distance_error)
                troch_total_rewards_opp.append(score)
                troch_dices_opp.append(dice)
                troch_overlaps_opp.append(overlap)
            elif "_F_" in filename:
                troch_dist_errors_F.append(distance_error)
                troch_total_rewards_F.append(score)
                troch_dices_F.append(dice)
                troch_overlaps_F.append(overlap)
            elif "_W_" in filename:
                troch_dist_errors_W.append(distance_error)
                troch_total_rewards_W.append(score)
                troch_dices_W.append(dice)
                troch_overlaps_W.append(overlap)
            elif "_t1_" in filename:
                troch_dist_errors_T1.append(distance_error)
                troch_total_rewards_T1.append(score)
                troch_dices_T1.append(dice)
                troch_overlaps_T1.append(overlap)
            elif "_t2_" in filename:
                troch_dist_errors_T2.append(distance_error)
                troch_total_rewards_T2.append(score)
                troch_dices_T2.append(dice)
                troch_overlaps_T2.append(overlap)
            else:
                print("unknown image type, exiting...")
                sys.exit()
            troch_row += 1
        elif landmark == "knee":
            knee_sheet.write(knee_row,0,filename)
            knee_sheet.write(knee_row,1,distance_error)
            knee_sheet.write(knee_row,2,score)
            knee_sheet.write(knee_row,3,dice)
            knee_sheet.write(knee_row,4,overlap)
            knee_dist_errors.append(distance_error)
            knee_total_rewards.append(score)
            knee_dices.append(dice)
            knee_overlaps.append(overlap)
            if "_in_" in filename:
                knee_dist_errors_in.append(distance_error)
                knee_total_rewards_in.append(score)
                knee_dices_in.append(dice)
                knee_overlaps_in.append(overlap)
            elif "_opp_" in filename:
                knee_dist_errors_opp.append(distance_error)
                knee_total_rewards_opp.append(score)
                knee_dices_opp.append(dice)
                knee_overlaps_opp.append(overlap)
            elif "_F_" in filename:
                knee_dist_errors_F.append(distance_error)
                knee_total_rewards_F.append(score)
                knee_dices_F.append(dice)
                knee_overlaps_F.append(overlap)
            elif "_W_" in filename:
                knee_dist_errors_W.append(distance_error)
                knee_total_rewards_W.append(score)
                knee_dices_W.append(dice)
                knee_overlaps_W.append(overlap)
            elif "_t1_" in filename:
                knee_dist_errors_T1.append(distance_error)
                knee_total_rewards_T1.append(score)
                knee_dices_T1.append(dice)
                knee_overlaps_T1.append(overlap)
            elif "_t2_" in filename:
                knee_dist_errors_T2.append(distance_error)
                knee_total_rewards_T2.append(score)
                knee_dices_T2.append(dice)
                knee_overlaps_T2.append(overlap)
            else:
                print("unknown image type, exiting...")
                sys.exit()
            knee_row += 1
        elif landmark == "prostate":
            pros_sheet.write(pros_row,0,filename)
            pros_sheet.write(pros_row,1,distance_error)
            pros_sheet.write(pros_row,2,score)
            pros_sheet.write(pros_row,3,dice)
            pros_sheet.write(pros_row,4,overlap)
            pros_dist_errors.append(distance_error)
            pros_total_rewards.append(score)
            pros_dices.append(dice)
            pros_overlaps.append(overlap)
            if "ADC" in filename or "adc" in filename:
                pros_dist_errors_adc.append(distance_error)
                pros_total_rewards_adc.append(score)
                pros_dices_adc.append(dice)
                pros_overlaps_adc.append(overlap)
            elif "_T2_" in filename:
                pros_dist_errors_T2.append(distance_error)
                pros_total_rewards_T2.append(score)
                pros_dices_T2.append(dice)
                pros_overlaps_T2.append(overlap)
            else:
                print("unknown image type, exiting...")
                sys.exit()
            pros_row += 1
        elif landmark == "breast":
            breast_sheet.write(breast_row,0,filename)
            breast_sheet.write(breast_row,1,distance_error)
            breast_sheet.write(breast_row,2,score)
            breast_sheet.write(breast_row,3,dice)
            breast_sheet.write(breast_row,4,overlap)
            breast_dist_errors.append(distance_error)
            breast_total_rewards.append(score)
            breast_dices.append(dice)
            breast_overlaps.append(overlap)
            breast_row += 1
            if "_Post" in filename:
                breast_dist_errors_Post.append(distance_error)
                breast_total_rewards_Post.append(score)
                breast_dices_Post.append(dice)
                breast_overlaps_Post.append(overlap)
            elif "_Pre" in filename:
                breast_dist_errors_Pre.append(distance_error)
                breast_total_rewards_Pre.append(score)
                breast_dices_Pre.append(dice)
                breast_overlaps_Pre.append(overlap)
            elif "_SUB" in filename:
                breast_dist_errors_SUB.append(distance_error)
                breast_total_rewards_SUB.append(score)
                breast_dices_SUB.append(dice)
                breast_overlaps_SUB.append(overlap)
            elif "_T1" in filename:
                breast_dist_errors_T1.append(distance_error)
                breast_total_rewards_T1.append(score)
                breast_dices_T1.append(dice)
                breast_overlaps_T1.append(overlap)
            elif "_T2" in filename:
                breast_dist_errors_T2.append(distance_error)
                breast_total_rewards_T2.append(score)
                breast_dices_T2.append(dice)
                breast_overlaps_T2.append(overlap)
            else:
                print("unknonwn image type, exiting...")
                sys.exit()
        else:
            print("UNKNOWN LANDMARK, exiting...")
            sys.exit()


    ###################### all ######################
    all_dist_errors_min = min(all_dist_errors)
    all_total_rewards_min = min(all_total_rewards)
    all_dices_min = min(all_dices)
    all_overlaps_min = min(all_overlaps)
    all_dist_errors_max = max(all_dist_errors)
    all_total_rewards_max = max(all_total_rewards)
    all_dices_max = max(all_dices)
    all_overlaps_max = max(all_overlaps)
    all_dist_errors_med = statistics.median(all_dist_errors)
    all_total_rewards_med = statistics.median(all_total_rewards)
    all_dices_med = statistics.median(all_dices)
    all_overlaps_med = statistics.median(all_overlaps)
    all_dist_errors_mean = statistics.mean(all_dist_errors)
    all_total_rewards_mean = statistics.mean(all_total_rewards)
    all_dices_mean = statistics.mean(all_dices)
    all_overlaps_mean = statistics.mean(all_overlaps)
    all_dist_errors_stdev = statistics.stdev(all_dist_errors)
    all_total_rewards_stdev = statistics.stdev(all_total_rewards)
    all_dices_stdev = statistics.stdev(all_dices)
    all_overlaps_stdev = statistics.stdev(all_overlaps)

    all_sheet.write(all_row,0,"min all img params")
    all_sheet.write(all_row,1,all_dist_errors_min)
    all_sheet.write(all_row,2,all_total_rewards_min)
    all_sheet.write(all_row,3,all_dices_min)
    all_sheet.write(all_row,4,all_overlaps_min)
    all_row += 1
    all_sheet.write(all_row,0,"max all img params")
    all_sheet.write(all_row,1,all_dist_errors_max)
    all_sheet.write(all_row,2,all_total_rewards_max)
    all_sheet.write(all_row,3,all_dices_max)
    all_sheet.write(all_row,4,all_overlaps_max)
    all_row += 1
    all_sheet.write(all_row,0,"median all img params")
    all_sheet.write(all_row,1,all_dist_errors_med)
    all_sheet.write(all_row,2,all_total_rewards_med)
    all_sheet.write(all_row,3,all_dices_med)
    all_sheet.write(all_row,4,all_overlaps_med)
    all_row += 1
    all_sheet.write(all_row,0,"mean all img params")
    all_sheet.write(all_row,1,all_dist_errors_mean)
    all_sheet.write(all_row,2,all_total_rewards_mean)
    all_sheet.write(all_row,3,all_dices_mean)
    all_sheet.write(all_row,4,all_overlaps_mean)
    all_row += 1
    all_sheet.write(all_row,0,"stdev all img params")
    all_sheet.write(all_row,1,all_dist_errors_stdev)
    all_sheet.write(all_row,2,all_total_rewards_stdev)
    all_sheet.write(all_row,3,all_dices_stdev)
    all_sheet.write(all_row,4,all_overlaps_stdev)
    all_row += 1


    ##################### heart #####################
    heart_dist_errors_min = min(heart_dist_errors)
    heart_total_rewards_min = min(heart_total_rewards)
    heart_dices_min = min(heart_dices)
    heart_overlaps_min = min(heart_overlaps)
    heart_dist_errors_max = max(heart_dist_errors)
    heart_total_rewards_max = max(heart_total_rewards)
    heart_dices_max = max(heart_dices)
    heart_overlaps_max = max(heart_overlaps)
    heart_dist_errors_med = statistics.median(heart_dist_errors)
    heart_total_rewards_med = statistics.median(heart_total_rewards)
    heart_dices_med = statistics.median(heart_dices)
    heart_overlaps_med = statistics.median(heart_overlaps)
    heart_dist_errors_mean = statistics.mean(heart_dist_errors)
    heart_total_rewards_mean = statistics.mean(heart_total_rewards)
    heart_dices_mean = statistics.mean(heart_dices)
    heart_overlaps_mean = statistics.mean(heart_overlaps)
    heart_dist_errors_stdev = statistics.stdev(heart_dist_errors)
    heart_total_rewards_stdev = statistics.stdev(heart_total_rewards)
    heart_dices_stdev = statistics.stdev(heart_dices)
    heart_overlaps_stdev = statistics.stdev(heart_overlaps)

    heart_dist_errors_in_min = min(heart_dist_errors_in)
    heart_total_rewards_in_min = min(heart_total_rewards_in)
    heart_dices_in_min = min(heart_dices_in)
    heart_overlaps_in_min = min(heart_overlaps_in)
    heart_dist_errors_in_max = max(heart_dist_errors_in)
    heart_total_rewards_in_max = max(heart_total_rewards_in)
    heart_dices_in_max = max(heart_dices_in)
    heart_overlaps_in_max = max(heart_overlaps_in)
    heart_dist_errors_in_med = statistics.median(heart_dist_errors_in)
    heart_total_rewards_in_med = statistics.median(heart_total_rewards_in)
    heart_dices_in_med = statistics.median(heart_dices_in)
    heart_overlaps_in_med = statistics.median(heart_overlaps_in)
    heart_dist_errors_in_mean = statistics.mean(heart_dist_errors_in)
    heart_total_rewards_in_mean = statistics.mean(heart_total_rewards_in)
    heart_dices_in_mean = statistics.mean(heart_dices_in)
    heart_overlaps_in_mean = statistics.mean(heart_overlaps_in)
    heart_dist_errors_in_stdev = statistics.stdev(heart_dist_errors_in)
    heart_total_rewards_in_stdev = statistics.stdev(heart_total_rewards_in)
    heart_dices_in_stdev = statistics.stdev(heart_dices_in)
    heart_overlaps_in_stdev = statistics.stdev(heart_overlaps_in)

    heart_dist_errors_opp_min = min(heart_dist_errors_opp)
    heart_total_rewards_opp_min = min(heart_total_rewards_opp)
    heart_dices_opp_min = min(heart_dices_opp)
    heart_overlaps_opp_min = min(heart_overlaps_opp)
    heart_dist_errors_opp_max = max(heart_dist_errors_opp)
    heart_total_rewards_opp_max = max(heart_total_rewards_opp)
    heart_dices_opp_max = max(heart_dices_opp)
    heart_overlaps_opp_max = max(heart_overlaps_opp)
    heart_dist_errors_opp_med = statistics.median(heart_dist_errors_opp)
    heart_total_rewards_opp_med = statistics.median(heart_total_rewards_opp)
    heart_dices_opp_med = statistics.median(heart_dices_opp)
    heart_overlaps_opp_med = statistics.median(heart_overlaps_opp)
    heart_dist_errors_opp_mean = statistics.mean(heart_dist_errors_opp)
    heart_total_rewards_opp_mean = statistics.mean(heart_total_rewards_opp)
    heart_dices_opp_mean = statistics.mean(heart_dices_opp)
    heart_overlaps_opp_mean = statistics.mean(heart_overlaps_opp)
    heart_dist_errors_opp_stdev = statistics.stdev(heart_dist_errors_opp)
    heart_total_rewards_opp_stdev = statistics.stdev(heart_total_rewards_opp)
    heart_dices_opp_stdev = statistics.stdev(heart_dices_opp)
    heart_overlaps_opp_stdev = statistics.stdev(heart_overlaps_opp)

    heart_dist_errors_F_min = min(heart_dist_errors_F)
    heart_total_rewards_F_min = min(heart_total_rewards_F)
    heart_dices_F_min = min(heart_dices_F)
    heart_overlaps_F_min = min(heart_overlaps_F)
    heart_dist_errors_F_max = max(heart_dist_errors_F)
    heart_total_rewards_F_max = max(heart_total_rewards_F)
    heart_dices_F_max = max(heart_dices_F)
    heart_overlaps_F_max = max(heart_overlaps_F)
    heart_dist_errors_F_med = statistics.median(heart_dist_errors_F)
    heart_total_rewards_F_med = statistics.median(heart_total_rewards_F)
    heart_dices_F_med = statistics.median(heart_dices_F)
    heart_overlaps_F_med = statistics.median(heart_overlaps_F)
    heart_dist_errors_F_mean = statistics.mean(heart_dist_errors_F)
    heart_total_rewards_F_mean = statistics.mean(heart_total_rewards_F)
    heart_dices_F_mean = statistics.mean(heart_dices_F)
    heart_overlaps_F_mean = statistics.mean(heart_overlaps_F)
    heart_dist_errors_F_stdev = statistics.stdev(heart_dist_errors_F)
    heart_total_rewards_F_stdev = statistics.stdev(heart_total_rewards_F)
    heart_dices_F_stdev = statistics.stdev(heart_dices_F)
    heart_overlaps_F_stdev = statistics.stdev(heart_overlaps_F)

    heart_dist_errors_W_min = min(heart_dist_errors_W)
    heart_total_rewards_W_min = min(heart_total_rewards_W)
    heart_dices_W_min = min(heart_dices_W)
    heart_overlaps_W_min = min(heart_overlaps_W)
    heart_dist_errors_W_max = max(heart_dist_errors_W)
    heart_total_rewards_W_max = max(heart_total_rewards_W)
    heart_dices_W_max = max(heart_dices_W)
    heart_overlaps_W_max = max(heart_overlaps_W)
    heart_dist_errors_W_med = statistics.median(heart_dist_errors_W)
    heart_total_rewards_W_med = statistics.median(heart_total_rewards_W)
    heart_dices_W_med = statistics.median(heart_dices_W)
    heart_overlaps_W_med = statistics.median(heart_overlaps_W)
    heart_dist_errors_W_mean = statistics.mean(heart_dist_errors_W)
    heart_total_rewards_W_mean = statistics.mean(heart_total_rewards_W)
    heart_dices_W_mean = statistics.mean(heart_dices_W)
    heart_overlaps_W_mean = statistics.mean(heart_overlaps_W)
    heart_dist_errors_W_stdev = statistics.stdev(heart_dist_errors_W)
    heart_total_rewards_W_stdev = statistics.stdev(heart_total_rewards_W)
    heart_dices_W_stdev = statistics.stdev(heart_dices_W)
    heart_overlaps_W_stdev = statistics.stdev(heart_overlaps_W)

    heart_dist_errors_T1_min = min(heart_dist_errors_T1)
    heart_total_rewards_T1_min = min(heart_total_rewards_T1)
    heart_dices_T1_min = min(heart_dices_T1)
    heart_overlaps_T1_min = min(heart_overlaps_T1)
    heart_dist_errors_T1_max = max(heart_dist_errors_T1)
    heart_total_rewards_T1_max = max(heart_total_rewards_T1)
    heart_dices_T1_max = max(heart_dices_T1)
    heart_overlaps_T1_max = max(heart_overlaps_T1)
    heart_dist_errors_T1_med = statistics.median(heart_dist_errors_T1)
    heart_total_rewards_T1_med = statistics.median(heart_total_rewards_T1)
    heart_dices_T1_med = statistics.median(heart_dices_T1)
    heart_overlaps_T1_med = statistics.median(heart_overlaps_T1)
    heart_dist_errors_T1_mean = statistics.mean(heart_dist_errors_T1)
    heart_total_rewards_T1_mean = statistics.mean(heart_total_rewards_T1)
    heart_dices_T1_mean = statistics.mean(heart_dices_T1)
    heart_overlaps_T1_mean = statistics.mean(heart_overlaps_T1)
    heart_dist_errors_T1_stdev = statistics.stdev(heart_dist_errors_T1)
    heart_total_rewards_T1_stdev = statistics.stdev(heart_total_rewards_T1)
    heart_dices_T1_stdev = statistics.stdev(heart_dices_T1)
    heart_overlaps_T1_stdev = statistics.stdev(heart_overlaps_T1)

    heart_dist_errors_T2_min = min(heart_dist_errors_T2)
    heart_total_rewards_T2_min = min(heart_total_rewards_T2)
    heart_dices_T2_min = min(heart_dices_T2)
    heart_overlaps_T2_min = min(heart_overlaps_T2)
    heart_dist_errors_T2_max = max(heart_dist_errors_T2)
    heart_total_rewards_T2_max = max(heart_total_rewards_T2)
    heart_dices_T2_max = max(heart_dices_T2)
    heart_overlaps_T2_max = max(heart_overlaps_T2)
    heart_dist_errors_T2_med = statistics.median(heart_dist_errors_T2)
    heart_total_rewards_T2_med = statistics.median(heart_total_rewards_T2)
    heart_dices_T2_med = statistics.median(heart_dices_T2)
    heart_overlaps_T2_med = statistics.median(heart_overlaps_T2)
    heart_dist_errors_T2_mean = statistics.mean(heart_dist_errors_T2)
    heart_total_rewards_T2_mean = statistics.mean(heart_total_rewards_T2)
    heart_dices_T2_mean = statistics.mean(heart_dices_T2)
    heart_overlaps_T2_mean = statistics.mean(heart_overlaps_T2)
    heart_dist_errors_T2_stdev = statistics.stdev(heart_dist_errors_T2)
    heart_total_rewards_T2_stdev = statistics.stdev(heart_total_rewards_T2)
    heart_dices_T2_stdev = statistics.stdev(heart_dices_T2)
    heart_overlaps_T2_stdev = statistics.stdev(heart_overlaps_T2)

    heart_sheet.write(heart_row,0,"min all heart img params")
    heart_sheet.write(heart_row,1,heart_dist_errors_min)
    heart_sheet.write(heart_row,2,heart_total_rewards_min)
    heart_sheet.write(heart_row,3,heart_dices_min)
    heart_sheet.write(heart_row,4,heart_overlaps_min)
    heart_row += 1
    heart_sheet.write(heart_row,0,"max all heart img params")
    heart_sheet.write(heart_row,1,heart_dist_errors_max)
    heart_sheet.write(heart_row,2,heart_total_rewards_max)
    heart_sheet.write(heart_row,3,heart_dices_max)
    heart_sheet.write(heart_row,4,heart_overlaps_max)
    heart_row += 1
    heart_sheet.write(heart_row,0,"median all heart img params")
    heart_sheet.write(heart_row,1,heart_dist_errors_med)
    heart_sheet.write(heart_row,2,heart_total_rewards_med)
    heart_sheet.write(heart_row,3,heart_dices_med)
    heart_sheet.write(heart_row,4,heart_overlaps_med)
    heart_row += 1
    heart_sheet.write(heart_row,0,"mean all heart img params")
    heart_sheet.write(heart_row,1,heart_dist_errors_mean)
    heart_sheet.write(heart_row,2,heart_total_rewards_mean)
    heart_sheet.write(heart_row,3,heart_dices_mean)
    heart_sheet.write(heart_row,4,heart_overlaps_mean)
    heart_row += 1
    heart_sheet.write(heart_row,0,"stdev all heart img params")
    heart_sheet.write(heart_row,1,heart_dist_errors_stdev)
    heart_sheet.write(heart_row,2,heart_total_rewards_stdev)
    heart_sheet.write(heart_row,3,heart_dices_stdev)
    heart_sheet.write(heart_row,4,heart_overlaps_stdev)
    heart_row += 1

    heart_sheet.write(heart_row,0,"min in")
    heart_sheet.write(heart_row,1,heart_dist_errors_in_min)
    heart_sheet.write(heart_row,2,heart_total_rewards_in_min)
    heart_sheet.write(heart_row,3,heart_dices_in_min)
    heart_sheet.write(heart_row,4,heart_overlaps_in_min)
    heart_row += 1
    heart_sheet.write(heart_row,0,"max in")
    heart_sheet.write(heart_row,1,heart_dist_errors_in_max)
    heart_sheet.write(heart_row,2,heart_total_rewards_in_max)
    heart_sheet.write(heart_row,3,heart_dices_in_max)
    heart_sheet.write(heart_row,4,heart_overlaps_in_max)
    heart_row += 1
    heart_sheet.write(heart_row,0,"median in")
    heart_sheet.write(heart_row,1,heart_dist_errors_in_med)
    heart_sheet.write(heart_row,2,heart_total_rewards_in_med)
    heart_sheet.write(heart_row,3,heart_dices_in_med)
    heart_sheet.write(heart_row,4,heart_overlaps_in_med)
    heart_row += 1
    heart_sheet.write(heart_row,0,"mean in")
    heart_sheet.write(heart_row,1,heart_dist_errors_in_mean)
    heart_sheet.write(heart_row,2,heart_total_rewards_in_mean)
    heart_sheet.write(heart_row,3,heart_dices_in_mean)
    heart_sheet.write(heart_row,4,heart_overlaps_in_mean)
    heart_row += 1
    heart_sheet.write(heart_row,0,"stdev in")
    heart_sheet.write(heart_row,1,heart_dist_errors_in_stdev)
    heart_sheet.write(heart_row,2,heart_total_rewards_in_stdev)
    heart_sheet.write(heart_row,3,heart_dices_in_stdev)
    heart_sheet.write(heart_row,4,heart_overlaps_in_stdev)
    heart_row += 1

    heart_sheet.write(heart_row,0,"min opp")
    heart_sheet.write(heart_row,1,heart_dist_errors_opp_min)
    heart_sheet.write(heart_row,2,heart_total_rewards_opp_min)
    heart_sheet.write(heart_row,3,heart_dices_opp_min)
    heart_sheet.write(heart_row,4,heart_overlaps_opp_min)
    heart_row += 1
    heart_sheet.write(heart_row,0,"max opp")
    heart_sheet.write(heart_row,1,heart_dist_errors_opp_max)
    heart_sheet.write(heart_row,2,heart_total_rewards_opp_max)
    heart_sheet.write(heart_row,3,heart_dices_opp_max)
    heart_sheet.write(heart_row,4,heart_overlaps_opp_max)
    heart_row += 1
    heart_sheet.write(heart_row,0,"median opp")
    heart_sheet.write(heart_row,1,heart_dist_errors_opp_med)
    heart_sheet.write(heart_row,2,heart_total_rewards_opp_med)
    heart_sheet.write(heart_row,3,heart_dices_opp_med)
    heart_sheet.write(heart_row,4,heart_overlaps_opp_med)
    heart_row += 1
    heart_sheet.write(heart_row,0,"mean opp")
    heart_sheet.write(heart_row,1,heart_dist_errors_opp_mean)
    heart_sheet.write(heart_row,2,heart_total_rewards_opp_mean)
    heart_sheet.write(heart_row,3,heart_dices_opp_mean)
    heart_sheet.write(heart_row,4,heart_overlaps_opp_mean)
    heart_row += 1
    heart_sheet.write(heart_row,0,"stdev opp")
    heart_sheet.write(heart_row,1,heart_dist_errors_opp_stdev)
    heart_sheet.write(heart_row,2,heart_total_rewards_opp_stdev)
    heart_sheet.write(heart_row,3,heart_dices_opp_stdev)
    heart_sheet.write(heart_row,4,heart_overlaps_opp_stdev)
    heart_row += 1

    heart_sheet.write(heart_row,0,"min F")
    heart_sheet.write(heart_row,1,heart_dist_errors_F_min)
    heart_sheet.write(heart_row,2,heart_total_rewards_F_min)
    heart_sheet.write(heart_row,3,heart_dices_F_min)
    heart_sheet.write(heart_row,4,heart_overlaps_F_min)
    heart_row += 1
    heart_sheet.write(heart_row,0,"max F")
    heart_sheet.write(heart_row,1,heart_dist_errors_F_max)
    heart_sheet.write(heart_row,2,heart_total_rewards_F_max)
    heart_sheet.write(heart_row,3,heart_dices_F_max)
    heart_sheet.write(heart_row,4,heart_overlaps_F_max)
    heart_row += 1
    heart_sheet.write(heart_row,0,"median F")
    heart_sheet.write(heart_row,1,heart_dist_errors_F_med)
    heart_sheet.write(heart_row,2,heart_total_rewards_F_med)
    heart_sheet.write(heart_row,3,heart_dices_F_med)
    heart_sheet.write(heart_row,4,heart_overlaps_F_med)
    heart_row += 1
    heart_sheet.write(heart_row,0,"mean F")
    heart_sheet.write(heart_row,1,heart_dist_errors_F_mean)
    heart_sheet.write(heart_row,2,heart_total_rewards_F_mean)
    heart_sheet.write(heart_row,3,heart_dices_F_mean)
    heart_sheet.write(heart_row,4,heart_overlaps_F_mean)
    heart_row += 1
    heart_sheet.write(heart_row,0,"stdev F")
    heart_sheet.write(heart_row,1,heart_dist_errors_F_stdev)
    heart_sheet.write(heart_row,2,heart_total_rewards_F_stdev)
    heart_sheet.write(heart_row,3,heart_dices_F_stdev)
    heart_sheet.write(heart_row,4,heart_overlaps_F_stdev)
    heart_row += 1

    heart_sheet.write(heart_row,0,"min W")
    heart_sheet.write(heart_row,1,heart_dist_errors_W_min)
    heart_sheet.write(heart_row,2,heart_total_rewards_W_min)
    heart_sheet.write(heart_row,3,heart_dices_W_min)
    heart_sheet.write(heart_row,4,heart_overlaps_W_min)
    heart_row += 1
    heart_sheet.write(heart_row,0,"max W")
    heart_sheet.write(heart_row,1,heart_dist_errors_W_max)
    heart_sheet.write(heart_row,2,heart_total_rewards_W_max)
    heart_sheet.write(heart_row,3,heart_dices_W_max)
    heart_sheet.write(heart_row,4,heart_overlaps_W_max)
    heart_row += 1
    heart_sheet.write(heart_row,0,"median W")
    heart_sheet.write(heart_row,1,heart_dist_errors_W_med)
    heart_sheet.write(heart_row,2,heart_total_rewards_W_med)
    heart_sheet.write(heart_row,3,heart_dices_W_med)
    heart_sheet.write(heart_row,4,heart_overlaps_W_med)
    heart_row += 1
    heart_sheet.write(heart_row,0,"mean W")
    heart_sheet.write(heart_row,1,heart_dist_errors_W_mean)
    heart_sheet.write(heart_row,2,heart_total_rewards_W_mean)
    heart_sheet.write(heart_row,3,heart_dices_W_mean)
    heart_sheet.write(heart_row,4,heart_overlaps_W_mean)
    heart_row += 1
    heart_sheet.write(heart_row,0,"stdev W")
    heart_sheet.write(heart_row,1,heart_dist_errors_W_stdev)
    heart_sheet.write(heart_row,2,heart_total_rewards_W_stdev)
    heart_sheet.write(heart_row,3,heart_dices_W_stdev)
    heart_sheet.write(heart_row,4,heart_overlaps_W_stdev)
    heart_row += 1

    heart_sheet.write(heart_row,0,"min T1 non fs")
    heart_sheet.write(heart_row,1,heart_dist_errors_T1_min)
    heart_sheet.write(heart_row,2,heart_total_rewards_T1_min)
    heart_sheet.write(heart_row,3,heart_dices_T1_min)
    heart_sheet.write(heart_row,4,heart_overlaps_T1_min)
    heart_row += 1
    heart_sheet.write(heart_row,0,"max T1 non fs")
    heart_sheet.write(heart_row,1,heart_dist_errors_T1_max)
    heart_sheet.write(heart_row,2,heart_total_rewards_T1_max)
    heart_sheet.write(heart_row,3,heart_dices_T1_max)
    heart_sheet.write(heart_row,4,heart_overlaps_T1_max)
    heart_row += 1
    heart_sheet.write(heart_row,0,"median T1 non fs")
    heart_sheet.write(heart_row,1,heart_dist_errors_T1_med)
    heart_sheet.write(heart_row,2,heart_total_rewards_T1_med)
    heart_sheet.write(heart_row,3,heart_dices_T1_med)
    heart_sheet.write(heart_row,4,heart_overlaps_T1_med)
    heart_row += 1
    heart_sheet.write(heart_row,0,"mean T1 non fs")
    heart_sheet.write(heart_row,1,heart_dist_errors_T1_mean)
    heart_sheet.write(heart_row,2,heart_total_rewards_T1_mean)
    heart_sheet.write(heart_row,3,heart_dices_T1_mean)
    heart_sheet.write(heart_row,4,heart_overlaps_T1_mean)
    heart_row += 1
    heart_sheet.write(heart_row,0,"stdev T1 non fs")
    heart_sheet.write(heart_row,1,heart_dist_errors_T1_stdev)
    heart_sheet.write(heart_row,2,heart_total_rewards_T1_stdev)
    heart_sheet.write(heart_row,3,heart_dices_T1_stdev)
    heart_sheet.write(heart_row,4,heart_overlaps_T1_stdev)
    heart_row += 1

    heart_sheet.write(heart_row,0,"min T2 haste tirm")
    heart_sheet.write(heart_row,1,heart_dist_errors_T2_min)
    heart_sheet.write(heart_row,2,heart_total_rewards_T2_min)
    heart_sheet.write(heart_row,3,heart_dices_T2_min)
    heart_sheet.write(heart_row,4,heart_overlaps_T2_min)
    heart_row += 1
    heart_sheet.write(heart_row,0,"max T2 haste tirm")
    heart_sheet.write(heart_row,1,heart_dist_errors_T2_max)
    heart_sheet.write(heart_row,2,heart_total_rewards_T2_max)
    heart_sheet.write(heart_row,3,heart_dices_T2_max)
    heart_sheet.write(heart_row,4,heart_overlaps_T2_max)
    heart_row += 1
    heart_sheet.write(heart_row,0,"median T2 haste tirm")
    heart_sheet.write(heart_row,1,heart_dist_errors_T2_med)
    heart_sheet.write(heart_row,2,heart_total_rewards_T2_med)
    heart_sheet.write(heart_row,3,heart_dices_T2_med)
    heart_sheet.write(heart_row,4,heart_overlaps_T2_med)
    heart_row += 1
    heart_sheet.write(heart_row,0,"mean T2 haste tirm")
    heart_sheet.write(heart_row,1,heart_dist_errors_T2_mean)
    heart_sheet.write(heart_row,2,heart_total_rewards_T2_mean)
    heart_sheet.write(heart_row,3,heart_dices_T2_mean)
    heart_sheet.write(heart_row,4,heart_overlaps_T2_mean)
    heart_row += 1
    heart_sheet.write(heart_row,0,"stdev T2 haste tirm")
    heart_sheet.write(heart_row,1,heart_dist_errors_T2_stdev)
    heart_sheet.write(heart_row,2,heart_total_rewards_T2_stdev)
    heart_sheet.write(heart_row,3,heart_dices_T2_stdev)
    heart_sheet.write(heart_row,4,heart_overlaps_T2_stdev)
    heart_row += 1




    ##################   kidney  #############################
    kidney_dist_errors_min = min(kidney_dist_errors)
    kidney_total_rewards_min = min(kidney_total_rewards)
    kidney_dices_min = min(kidney_dices)
    kidney_overlaps_min = min(kidney_overlaps)
    kidney_dist_errors_max = max(kidney_dist_errors)
    kidney_total_rewards_max = max(kidney_total_rewards)
    kidney_dices_max = max(kidney_dices)
    kidney_overlaps_max = max(kidney_overlaps)
    kidney_dist_errors_med = statistics.median(kidney_dist_errors)
    kidney_total_rewards_med = statistics.median(kidney_total_rewards)
    kidney_dices_med = statistics.median(kidney_dices)
    kidney_overlaps_med = statistics.median(kidney_overlaps)
    kidney_dist_errors_mean = statistics.mean(kidney_dist_errors)
    kidney_total_rewards_mean = statistics.mean(kidney_total_rewards)
    kidney_dices_mean = statistics.mean(kidney_dices)
    kidney_overlaps_mean = statistics.mean(kidney_overlaps)
    kidney_dist_errors_stdev = statistics.stdev(kidney_dist_errors)
    kidney_total_rewards_stdev = statistics.stdev(kidney_total_rewards)
    kidney_dices_stdev = statistics.stdev(kidney_dices)
    kidney_overlaps_stdev = statistics.stdev(kidney_overlaps)

    kidney_dist_errors_in_min = min(kidney_dist_errors_in)
    kidney_total_rewards_in_min = min(kidney_total_rewards_in)
    kidney_dices_in_min = min(kidney_dices_in)
    kidney_overlaps_in_min = min(kidney_overlaps_in)
    kidney_dist_errors_in_max = max(kidney_dist_errors_in)
    kidney_total_rewards_in_max = max(kidney_total_rewards_in)
    kidney_dices_in_max = max(kidney_dices_in)
    kidney_overlaps_in_max = max(kidney_overlaps_in)
    kidney_dist_errors_in_med = statistics.median(kidney_dist_errors_in)
    kidney_total_rewards_in_med = statistics.median(kidney_total_rewards_in)
    kidney_dices_in_med = statistics.median(kidney_dices_in)
    kidney_overlaps_in_med = statistics.median(kidney_overlaps_in)
    kidney_dist_errors_in_mean = statistics.mean(kidney_dist_errors_in)
    kidney_total_rewards_in_mean = statistics.mean(kidney_total_rewards_in)
    kidney_dices_in_mean = statistics.mean(kidney_dices_in)
    kidney_overlaps_in_mean = statistics.mean(kidney_overlaps_in)
    kidney_dist_errors_in_stdev = statistics.stdev(kidney_dist_errors_in)
    kidney_total_rewards_in_stdev = statistics.stdev(kidney_total_rewards_in)
    kidney_dices_in_stdev = statistics.stdev(kidney_dices_in)
    kidney_overlaps_in_stdev = statistics.stdev(kidney_overlaps_in)

    kidney_dist_errors_opp_min = min(kidney_dist_errors_opp)
    kidney_total_rewards_opp_min = min(kidney_total_rewards_opp)
    kidney_dices_opp_min = min(kidney_dices_opp)
    kidney_overlaps_opp_min = min(kidney_overlaps_opp)
    kidney_dist_errors_opp_max = max(kidney_dist_errors_opp)
    kidney_total_rewards_opp_max = max(kidney_total_rewards_opp)
    kidney_dices_opp_max = max(kidney_dices_opp)
    kidney_overlaps_opp_max = max(kidney_overlaps_opp)
    kidney_dist_errors_opp_med = statistics.median(kidney_dist_errors_opp)
    kidney_total_rewards_opp_med = statistics.median(kidney_total_rewards_opp)
    kidney_dices_opp_med = statistics.median(kidney_dices_opp)
    kidney_overlaps_opp_med = statistics.median(kidney_overlaps_opp)
    kidney_dist_errors_opp_mean = statistics.mean(kidney_dist_errors_opp)
    kidney_total_rewards_opp_mean = statistics.mean(kidney_total_rewards_opp)
    kidney_dices_opp_mean = statistics.mean(kidney_dices_opp)
    kidney_overlaps_opp_mean = statistics.mean(kidney_overlaps_opp)
    kidney_dist_errors_opp_stdev = statistics.stdev(kidney_dist_errors_opp)
    kidney_total_rewards_opp_stdev = statistics.stdev(kidney_total_rewards_opp)
    kidney_dices_opp_stdev = statistics.stdev(kidney_dices_opp)
    kidney_overlaps_opp_stdev = statistics.stdev(kidney_overlaps_opp)

    kidney_dist_errors_F_min = min(kidney_dist_errors_F)
    kidney_total_rewards_F_min = min(kidney_total_rewards_F)
    kidney_dices_F_min = min(kidney_dices_F)
    kidney_overlaps_F_min = min(kidney_overlaps_F)
    kidney_dist_errors_F_max = max(kidney_dist_errors_F)
    kidney_total_rewards_F_max = max(kidney_total_rewards_F)
    kidney_dices_F_max = max(kidney_dices_F)
    kidney_overlaps_F_max = max(kidney_overlaps_F)
    kidney_dist_errors_F_med = statistics.median(kidney_dist_errors_F)
    kidney_total_rewards_F_med = statistics.median(kidney_total_rewards_F)
    kidney_dices_F_med = statistics.median(kidney_dices_F)
    kidney_overlaps_F_med = statistics.median(kidney_overlaps_F)
    kidney_dist_errors_F_mean = statistics.mean(kidney_dist_errors_F)
    kidney_total_rewards_F_mean = statistics.mean(kidney_total_rewards_F)
    kidney_dices_F_mean = statistics.mean(kidney_dices_F)
    kidney_overlaps_F_mean = statistics.mean(kidney_overlaps_F)
    kidney_dist_errors_F_stdev = statistics.stdev(kidney_dist_errors_F)
    kidney_total_rewards_F_stdev = statistics.stdev(kidney_total_rewards_F)
    kidney_dices_F_stdev = statistics.stdev(kidney_dices_F)
    kidney_overlaps_F_stdev = statistics.stdev(kidney_overlaps_F)

    kidney_dist_errors_W_min = min(kidney_dist_errors_W)
    kidney_total_rewards_W_min = min(kidney_total_rewards_W)
    kidney_dices_W_min = min(kidney_dices_W)
    kidney_overlaps_W_min = min(kidney_overlaps_W)
    kidney_dist_errors_W_max = max(kidney_dist_errors_W)
    kidney_total_rewards_W_max = max(kidney_total_rewards_W)
    kidney_dices_W_max = max(kidney_dices_W)
    kidney_overlaps_W_max = max(kidney_overlaps_W)
    kidney_dist_errors_W_med = statistics.median(kidney_dist_errors_W)
    kidney_total_rewards_W_med = statistics.median(kidney_total_rewards_W)
    kidney_dices_W_med = statistics.median(kidney_dices_W)
    kidney_overlaps_W_med = statistics.median(kidney_overlaps_W)
    kidney_dist_errors_W_mean = statistics.mean(kidney_dist_errors_W)
    kidney_total_rewards_W_mean = statistics.mean(kidney_total_rewards_W)
    kidney_dices_W_mean = statistics.mean(kidney_dices_W)
    kidney_overlaps_W_mean = statistics.mean(kidney_overlaps_W)
    kidney_dist_errors_W_stdev = statistics.stdev(kidney_dist_errors_W)
    kidney_total_rewards_W_stdev = statistics.stdev(kidney_total_rewards_W)
    kidney_dices_W_stdev = statistics.stdev(kidney_dices_W)
    kidney_overlaps_W_stdev = statistics.stdev(kidney_overlaps_W)

    kidney_dist_errors_T1_min = min(kidney_dist_errors_T1)
    kidney_total_rewards_T1_min = min(kidney_total_rewards_T1)
    kidney_dices_T1_min = min(kidney_dices_T1)
    kidney_overlaps_T1_min = min(kidney_overlaps_T1)
    kidney_dist_errors_T1_max = max(kidney_dist_errors_T1)
    kidney_total_rewards_T1_max = max(kidney_total_rewards_T1)
    kidney_dices_T1_max = max(kidney_dices_T1)
    kidney_overlaps_T1_max = max(kidney_overlaps_T1)
    kidney_dist_errors_T1_med = statistics.median(kidney_dist_errors_T1)
    kidney_total_rewards_T1_med = statistics.median(kidney_total_rewards_T1)
    kidney_dices_T1_med = statistics.median(kidney_dices_T1)
    kidney_overlaps_T1_med = statistics.median(kidney_overlaps_T1)
    kidney_dist_errors_T1_mean = statistics.mean(kidney_dist_errors_T1)
    kidney_total_rewards_T1_mean = statistics.mean(kidney_total_rewards_T1)
    kidney_dices_T1_mean = statistics.mean(kidney_dices_T1)
    kidney_overlaps_T1_mean = statistics.mean(kidney_overlaps_T1)
    kidney_dist_errors_T1_stdev = statistics.stdev(kidney_dist_errors_T1)
    kidney_total_rewards_T1_stdev = statistics.stdev(kidney_total_rewards_T1)
    kidney_dices_T1_stdev = statistics.stdev(kidney_dices_T1)
    kidney_overlaps_T1_stdev = statistics.stdev(kidney_overlaps_T1)

    kidney_dist_errors_T2_min = min(kidney_dist_errors_T2)
    kidney_total_rewards_T2_min = min(kidney_total_rewards_T2)
    kidney_dices_T2_min = min(kidney_dices_T2)
    kidney_overlaps_T2_min = min(kidney_overlaps_T2)
    kidney_dist_errors_T2_max = max(kidney_dist_errors_T2)
    kidney_total_rewards_T2_max = max(kidney_total_rewards_T2)
    kidney_dices_T2_max = max(kidney_dices_T2)
    kidney_overlaps_T2_max = max(kidney_overlaps_T2)
    kidney_dist_errors_T2_med = statistics.median(kidney_dist_errors_T2)
    kidney_total_rewards_T2_med = statistics.median(kidney_total_rewards_T2)
    kidney_dices_T2_med = statistics.median(kidney_dices_T2)
    kidney_overlaps_T2_med = statistics.median(kidney_overlaps_T2)
    kidney_dist_errors_T2_mean = statistics.mean(kidney_dist_errors_T2)
    kidney_total_rewards_T2_mean = statistics.mean(kidney_total_rewards_T2)
    kidney_dices_T2_mean = statistics.mean(kidney_dices_T2)
    kidney_overlaps_T2_mean = statistics.mean(kidney_overlaps_T2)
    kidney_dist_errors_T2_stdev = statistics.stdev(kidney_dist_errors_T2)
    kidney_total_rewards_T2_stdev = statistics.stdev(kidney_total_rewards_T2)
    kidney_dices_T2_stdev = statistics.stdev(kidney_dices_T2)
    kidney_overlaps_T2_stdev = statistics.stdev(kidney_overlaps_T2)

    kidney_sheet.write(kidney_row,0,"min all kidney img params")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_min)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_min)
    kidney_sheet.write(kidney_row,3,kidney_dices_min)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_min)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"max all kidney img params")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_max)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_max)
    kidney_sheet.write(kidney_row,3,kidney_dices_max)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_max)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"median all kidney img params")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_med)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_med)
    kidney_sheet.write(kidney_row,3,kidney_dices_med)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_med)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"mean all kidney img params")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_mean)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_mean)
    kidney_sheet.write(kidney_row,3,kidney_dices_mean)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_mean)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"stdev all kidney img params")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_stdev)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_stdev)
    kidney_sheet.write(kidney_row,3,kidney_dices_stdev)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_stdev)
    kidney_row += 1

    kidney_sheet.write(kidney_row,0,"min in")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_in_min)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_in_min)
    kidney_sheet.write(kidney_row,3,kidney_dices_in_min)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_in_min)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"max in")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_in_max)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_in_max)
    kidney_sheet.write(kidney_row,3,kidney_dices_in_max)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_in_max)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"median in")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_in_med)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_in_med)
    kidney_sheet.write(kidney_row,3,kidney_dices_in_med)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_in_med)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"mean in")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_in_mean)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_in_mean)
    kidney_sheet.write(kidney_row,3,kidney_dices_in_mean)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_in_mean)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"stdev in")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_in_stdev)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_in_stdev)
    kidney_sheet.write(kidney_row,3,kidney_dices_in_stdev)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_in_stdev)
    kidney_row += 1

    kidney_sheet.write(kidney_row,0,"min opp")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_opp_min)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_opp_min)
    kidney_sheet.write(kidney_row,3,kidney_dices_opp_min)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_opp_min)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"max opp")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_opp_max)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_opp_max)
    kidney_sheet.write(kidney_row,3,kidney_dices_opp_max)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_opp_max)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"median opp")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_opp_med)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_opp_med)
    kidney_sheet.write(kidney_row,3,kidney_dices_opp_med)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_opp_med)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"mean opp")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_opp_mean)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_opp_mean)
    kidney_sheet.write(kidney_row,3,kidney_dices_opp_mean)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_opp_mean)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"stdev opp")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_opp_stdev)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_opp_stdev)
    kidney_sheet.write(kidney_row,3,kidney_dices_opp_stdev)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_opp_stdev)
    kidney_row += 1

    kidney_sheet.write(kidney_row,0,"min F")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_F_min)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_F_min)
    kidney_sheet.write(kidney_row,3,kidney_dices_F_min)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_F_min)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"max F")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_F_max)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_F_max)
    kidney_sheet.write(kidney_row,3,kidney_dices_F_max)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_F_max)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"median F")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_F_med)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_F_med)
    kidney_sheet.write(kidney_row,3,kidney_dices_F_med)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_F_med)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"mean F")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_F_mean)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_F_mean)
    kidney_sheet.write(kidney_row,3,kidney_dices_F_mean)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_F_mean)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"stdev F")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_F_stdev)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_F_stdev)
    kidney_sheet.write(kidney_row,3,kidney_dices_F_stdev)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_F_stdev)
    kidney_row += 1

    kidney_sheet.write(kidney_row,0,"min W")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_W_min)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_W_min)
    kidney_sheet.write(kidney_row,3,kidney_dices_W_min)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_W_min)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"max W")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_W_max)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_W_max)
    kidney_sheet.write(kidney_row,3,kidney_dices_W_max)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_W_max)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"median W")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_W_med)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_W_med)
    kidney_sheet.write(kidney_row,3,kidney_dices_W_med)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_W_med)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"mean W")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_W_mean)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_W_mean)
    kidney_sheet.write(kidney_row,3,kidney_dices_W_mean)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_W_mean)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"stdev W")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_W_stdev)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_W_stdev)
    kidney_sheet.write(kidney_row,3,kidney_dices_W_stdev)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_W_stdev)
    kidney_row += 1

    kidney_sheet.write(kidney_row,0,"min T1 non fs")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_T1_min)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_T1_min)
    kidney_sheet.write(kidney_row,3,kidney_dices_T1_min)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_T1_min)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"max T1 non fs")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_T1_max)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_T1_max)
    kidney_sheet.write(kidney_row,3,kidney_dices_T1_max)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_T1_max)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"median T1 non fs")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_T1_med)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_T1_med)
    kidney_sheet.write(kidney_row,3,kidney_dices_T1_med)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_T1_med)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"mean T1 non fs")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_T1_mean)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_T1_mean)
    kidney_sheet.write(kidney_row,3,kidney_dices_T1_mean)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_T1_mean)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"stdev T1 non fs")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_T1_stdev)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_T1_stdev)
    kidney_sheet.write(kidney_row,3,kidney_dices_T1_stdev)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_T1_stdev)
    kidney_row += 1

    kidney_sheet.write(kidney_row,0,"min T2 haste tirm")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_T2_min)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_T2_min)
    kidney_sheet.write(kidney_row,3,kidney_dices_T2_min)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_T2_min)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"max T2 haste tirm")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_T2_max)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_T2_max)
    kidney_sheet.write(kidney_row,3,kidney_dices_T2_max)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_T2_max)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"median T2 haste tirm")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_T2_med)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_T2_med)
    kidney_sheet.write(kidney_row,3,kidney_dices_T2_med)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_T2_med)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"mean T2 haste tirm")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_T2_mean)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_T2_mean)
    kidney_sheet.write(kidney_row,3,kidney_dices_T2_mean)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_T2_mean)
    kidney_row += 1
    kidney_sheet.write(kidney_row,0,"stdev T2 haste tirm")
    kidney_sheet.write(kidney_row,1,kidney_dist_errors_T2_stdev)
    kidney_sheet.write(kidney_row,2,kidney_total_rewards_T2_stdev)
    kidney_sheet.write(kidney_row,3,kidney_dices_T2_stdev)
    kidney_sheet.write(kidney_row,4,kidney_overlaps_T2_stdev)
    kidney_row += 1





    ################# trochanter ###################
    troch_dist_errors_min = min(troch_dist_errors)
    troch_total_rewards_min = min(troch_total_rewards)
    troch_dices_min = min(troch_dices)
    troch_overlaps_min = min(troch_overlaps)
    troch_dist_errors_max = max(troch_dist_errors)
    troch_total_rewards_max = max(troch_total_rewards)
    troch_dices_max = max(troch_dices)
    troch_overlaps_max = max(troch_overlaps)
    troch_dist_errors_med = statistics.median(troch_dist_errors)
    troch_total_rewards_med = statistics.median(troch_total_rewards)
    troch_dices_med = statistics.median(troch_dices)
    troch_overlaps_med = statistics.median(troch_overlaps)
    troch_dist_errors_mean = statistics.mean(troch_dist_errors)
    troch_total_rewards_mean = statistics.mean(troch_total_rewards)
    troch_dices_mean = statistics.mean(troch_dices)
    troch_overlaps_mean = statistics.mean(troch_overlaps)
    troch_dist_errors_stdev = statistics.stdev(troch_dist_errors)
    troch_total_rewards_stdev = statistics.stdev(troch_total_rewards)
    troch_dices_stdev = statistics.stdev(troch_dices)
    troch_overlaps_stdev = statistics.stdev(troch_overlaps)

    troch_dist_errors_in_min = min(troch_dist_errors_in)
    troch_total_rewards_in_min = min(troch_total_rewards_in)
    troch_dices_in_min = min(troch_dices_in)
    troch_overlaps_in_min = min(troch_overlaps_in)
    troch_dist_errors_in_max = max(troch_dist_errors_in)
    troch_total_rewards_in_max = max(troch_total_rewards_in)
    troch_dices_in_max = max(troch_dices_in)
    troch_overlaps_in_max = max(troch_overlaps_in)
    troch_dist_errors_in_med = statistics.median(troch_dist_errors_in)
    troch_total_rewards_in_med = statistics.median(troch_total_rewards_in)
    troch_dices_in_med = statistics.median(troch_dices_in)
    troch_overlaps_in_med = statistics.median(troch_overlaps_in)
    troch_dist_errors_in_mean = statistics.mean(troch_dist_errors_in)
    troch_total_rewards_in_mean = statistics.mean(troch_total_rewards_in)
    troch_dices_in_mean = statistics.mean(troch_dices_in)
    troch_overlaps_in_mean = statistics.mean(troch_overlaps_in)
    troch_dist_errors_in_stdev = statistics.stdev(troch_dist_errors_in)
    troch_total_rewards_in_stdev = statistics.stdev(troch_total_rewards_in)
    troch_dices_in_stdev = statistics.stdev(troch_dices_in)
    troch_overlaps_in_stdev = statistics.stdev(troch_overlaps_in)

    troch_dist_errors_opp_min = min(troch_dist_errors_opp)
    troch_total_rewards_opp_min = min(troch_total_rewards_opp)
    troch_dices_opp_min = min(troch_dices_opp)
    troch_overlaps_opp_min = min(troch_overlaps_opp)
    troch_dist_errors_opp_max = max(troch_dist_errors_opp)
    troch_total_rewards_opp_max = max(troch_total_rewards_opp)
    troch_dices_opp_max = max(troch_dices_opp)
    troch_overlaps_opp_max = max(troch_overlaps_opp)
    troch_dist_errors_opp_med = statistics.median(troch_dist_errors_opp)
    troch_total_rewards_opp_med = statistics.median(troch_total_rewards_opp)
    troch_dices_opp_med = statistics.median(troch_dices_opp)
    troch_overlaps_opp_med = statistics.median(troch_overlaps_opp)
    troch_dist_errors_opp_mean = statistics.mean(troch_dist_errors_opp)
    troch_total_rewards_opp_mean = statistics.mean(troch_total_rewards_opp)
    troch_dices_opp_mean = statistics.mean(troch_dices_opp)
    troch_overlaps_opp_mean = statistics.mean(troch_overlaps_opp)
    troch_dist_errors_opp_stdev = statistics.stdev(troch_dist_errors_opp)
    troch_total_rewards_opp_stdev = statistics.stdev(troch_total_rewards_opp)
    troch_dices_opp_stdev = statistics.stdev(troch_dices_opp)
    troch_overlaps_opp_stdev = statistics.stdev(troch_overlaps_opp)

    troch_dist_errors_F_min = min(troch_dist_errors_F)
    troch_total_rewards_F_min = min(troch_total_rewards_F)
    troch_dices_F_min = min(troch_dices_F)
    troch_overlaps_F_min = min(troch_overlaps_F)
    troch_dist_errors_F_max = max(troch_dist_errors_F)
    troch_total_rewards_F_max = max(troch_total_rewards_F)
    troch_dices_F_max = max(troch_dices_F)
    troch_overlaps_F_max = max(troch_overlaps_F)
    troch_dist_errors_F_med = statistics.median(troch_dist_errors_F)
    troch_total_rewards_F_med = statistics.median(troch_total_rewards_F)
    troch_dices_F_med = statistics.median(troch_dices_F)
    troch_overlaps_F_med = statistics.median(troch_overlaps_F)
    troch_dist_errors_F_mean = statistics.mean(troch_dist_errors_F)
    troch_total_rewards_F_mean = statistics.mean(troch_total_rewards_F)
    troch_dices_F_mean = statistics.mean(troch_dices_F)
    troch_overlaps_F_mean = statistics.mean(troch_overlaps_F)
    troch_dist_errors_F_stdev = statistics.stdev(troch_dist_errors_F)
    troch_total_rewards_F_stdev = statistics.stdev(troch_total_rewards_F)
    troch_dices_F_stdev = statistics.stdev(troch_dices_F)
    troch_overlaps_F_stdev = statistics.stdev(troch_overlaps_F)

    troch_dist_errors_W_min = min(troch_dist_errors_W)
    troch_total_rewards_W_min = min(troch_total_rewards_W)
    troch_dices_W_min = min(troch_dices_W)
    troch_overlaps_W_min = min(troch_overlaps_W)
    troch_dist_errors_W_max = max(troch_dist_errors_W)
    troch_total_rewards_W_max = max(troch_total_rewards_W)
    troch_dices_W_max = max(troch_dices_W)
    troch_overlaps_W_max = max(troch_overlaps_W)
    troch_dist_errors_W_med = statistics.median(troch_dist_errors_W)
    troch_total_rewards_W_med = statistics.median(troch_total_rewards_W)
    troch_dices_W_med = statistics.median(troch_dices_W)
    troch_overlaps_W_med = statistics.median(troch_overlaps_W)
    troch_dist_errors_W_mean = statistics.mean(troch_dist_errors_W)
    troch_total_rewards_W_mean = statistics.mean(troch_total_rewards_W)
    troch_dices_W_mean = statistics.mean(troch_dices_W)
    troch_overlaps_W_mean = statistics.mean(troch_overlaps_W)
    troch_dist_errors_W_stdev = statistics.stdev(troch_dist_errors_W)
    troch_total_rewards_W_stdev = statistics.stdev(troch_total_rewards_W)
    troch_dices_W_stdev = statistics.stdev(troch_dices_W)
    troch_overlaps_W_stdev = statistics.stdev(troch_overlaps_W)

    troch_dist_errors_T1_min = min(troch_dist_errors_T1)
    troch_total_rewards_T1_min = min(troch_total_rewards_T1)
    troch_dices_T1_min = min(troch_dices_T1)
    troch_overlaps_T1_min = min(troch_overlaps_T1)
    troch_dist_errors_T1_max = max(troch_dist_errors_T1)
    troch_total_rewards_T1_max = max(troch_total_rewards_T1)
    troch_dices_T1_max = max(troch_dices_T1)
    troch_overlaps_T1_max = max(troch_overlaps_T1)
    troch_dist_errors_T1_med = statistics.median(troch_dist_errors_T1)
    troch_total_rewards_T1_med = statistics.median(troch_total_rewards_T1)
    troch_dices_T1_med = statistics.median(troch_dices_T1)
    troch_overlaps_T1_med = statistics.median(troch_overlaps_T1)
    troch_dist_errors_T1_mean = statistics.mean(troch_dist_errors_T1)
    troch_total_rewards_T1_mean = statistics.mean(troch_total_rewards_T1)
    troch_dices_T1_mean = statistics.mean(troch_dices_T1)
    troch_overlaps_T1_mean = statistics.mean(troch_overlaps_T1)
    troch_dist_errors_T1_stdev = statistics.stdev(troch_dist_errors_T1)
    troch_total_rewards_T1_stdev = statistics.stdev(troch_total_rewards_T1)
    troch_dices_T1_stdev = statistics.stdev(troch_dices_T1)
    troch_overlaps_T1_stdev = statistics.stdev(troch_overlaps_T1)

    troch_dist_errors_T2_min = min(troch_dist_errors_T2)
    troch_total_rewards_T2_min = min(troch_total_rewards_T2)
    troch_dices_T2_min = min(troch_dices_T2)
    troch_overlaps_T2_min = min(troch_overlaps_T2)
    troch_dist_errors_T2_max = max(troch_dist_errors_T2)
    troch_total_rewards_T2_max = max(troch_total_rewards_T2)
    troch_dices_T2_max = max(troch_dices_T2)
    troch_overlaps_T2_max = max(troch_overlaps_T2)
    troch_dist_errors_T2_med = statistics.median(troch_dist_errors_T2)
    troch_total_rewards_T2_med = statistics.median(troch_total_rewards_T2)
    troch_dices_T2_med = statistics.median(troch_dices_T2)
    troch_overlaps_T2_med = statistics.median(troch_overlaps_T2)
    troch_dist_errors_T2_mean = statistics.mean(troch_dist_errors_T2)
    troch_total_rewards_T2_mean = statistics.mean(troch_total_rewards_T2)
    troch_dices_T2_mean = statistics.mean(troch_dices_T2)
    troch_overlaps_T2_mean = statistics.mean(troch_overlaps_T2)
    troch_dist_errors_T2_stdev = statistics.stdev(troch_dist_errors_T2)
    troch_total_rewards_T2_stdev = statistics.stdev(troch_total_rewards_T2)
    troch_dices_T2_stdev = statistics.stdev(troch_dices_T2)
    troch_overlaps_T2_stdev = statistics.stdev(troch_overlaps_T2)

    troch_sheet.write(troch_row,0,"min all troch img params")
    troch_sheet.write(troch_row,1,troch_dist_errors_min)
    troch_sheet.write(troch_row,2,troch_total_rewards_min)
    troch_sheet.write(troch_row,3,troch_dices_min)
    troch_sheet.write(troch_row,4,troch_overlaps_min)
    troch_row += 1
    troch_sheet.write(troch_row,0,"max all troch img params")
    troch_sheet.write(troch_row,1,troch_dist_errors_max)
    troch_sheet.write(troch_row,2,troch_total_rewards_max)
    troch_sheet.write(troch_row,3,troch_dices_max)
    troch_sheet.write(troch_row,4,troch_overlaps_max)
    troch_row += 1
    troch_sheet.write(troch_row,0,"median all troch img params")
    troch_sheet.write(troch_row,1,troch_dist_errors_med)
    troch_sheet.write(troch_row,2,troch_total_rewards_med)
    troch_sheet.write(troch_row,3,troch_dices_med)
    troch_sheet.write(troch_row,4,troch_overlaps_med)
    troch_row += 1
    troch_sheet.write(troch_row,0,"mean all troch img params")
    troch_sheet.write(troch_row,1,troch_dist_errors_mean)
    troch_sheet.write(troch_row,2,troch_total_rewards_mean)
    troch_sheet.write(troch_row,3,troch_dices_mean)
    troch_sheet.write(troch_row,4,troch_overlaps_mean)
    troch_row += 1
    troch_sheet.write(troch_row,0,"stdev all trochimg params")
    troch_sheet.write(troch_row,1,troch_dist_errors_stdev)
    troch_sheet.write(troch_row,2,troch_total_rewards_stdev)
    troch_sheet.write(troch_row,3,troch_dices_stdev)
    troch_sheet.write(troch_row,4,troch_overlaps_stdev)
    troch_row += 1

    troch_sheet.write(troch_row,0,"min in")
    troch_sheet.write(troch_row,1,troch_dist_errors_in_min)
    troch_sheet.write(troch_row,2,troch_total_rewards_in_min)
    troch_sheet.write(troch_row,3,troch_dices_in_min)
    troch_sheet.write(troch_row,4,troch_overlaps_in_min)
    troch_row += 1
    troch_sheet.write(troch_row,0,"max in")
    troch_sheet.write(troch_row,1,troch_dist_errors_in_max)
    troch_sheet.write(troch_row,2,troch_total_rewards_in_max)
    troch_sheet.write(troch_row,3,troch_dices_in_max)
    troch_sheet.write(troch_row,4,troch_overlaps_in_max)
    troch_row += 1
    troch_sheet.write(troch_row,0,"median in")
    troch_sheet.write(troch_row,1,troch_dist_errors_in_med)
    troch_sheet.write(troch_row,2,troch_total_rewards_in_med)
    troch_sheet.write(troch_row,3,troch_dices_in_med)
    troch_sheet.write(troch_row,4,troch_overlaps_in_med)
    troch_row += 1
    troch_sheet.write(troch_row,0,"mean in")
    troch_sheet.write(troch_row,1,troch_dist_errors_in_mean)
    troch_sheet.write(troch_row,2,troch_total_rewards_in_mean)
    troch_sheet.write(troch_row,3,troch_dices_in_mean)
    troch_sheet.write(troch_row,4,troch_overlaps_in_mean)
    troch_row += 1
    troch_sheet.write(troch_row,0,"stdev in")
    troch_sheet.write(troch_row,1,troch_dist_errors_in_stdev)
    troch_sheet.write(troch_row,2,troch_total_rewards_in_stdev)
    troch_sheet.write(troch_row,3,troch_dices_in_stdev)
    troch_sheet.write(troch_row,4,troch_overlaps_in_stdev)
    troch_row += 1

    troch_sheet.write(troch_row,0,"min opp")
    troch_sheet.write(troch_row,1,troch_dist_errors_opp_min)
    troch_sheet.write(troch_row,2,troch_total_rewards_opp_min)
    troch_sheet.write(troch_row,3,troch_dices_opp_min)
    troch_sheet.write(troch_row,4,troch_overlaps_opp_min)
    troch_row += 1
    troch_sheet.write(troch_row,0,"max opp")
    troch_sheet.write(troch_row,1,troch_dist_errors_opp_max)
    troch_sheet.write(troch_row,2,troch_total_rewards_opp_max)
    troch_sheet.write(troch_row,3,troch_dices_opp_max)
    troch_sheet.write(troch_row,4,troch_overlaps_opp_max)
    troch_row += 1
    troch_sheet.write(troch_row,0,"median opp")
    troch_sheet.write(troch_row,1,troch_dist_errors_opp_med)
    troch_sheet.write(troch_row,2,troch_total_rewards_opp_med)
    troch_sheet.write(troch_row,3,troch_dices_opp_med)
    troch_sheet.write(troch_row,4,troch_overlaps_opp_med)
    troch_row += 1
    troch_sheet.write(troch_row,0,"mean opp")
    troch_sheet.write(troch_row,1,troch_dist_errors_opp_mean)
    troch_sheet.write(troch_row,2,troch_total_rewards_opp_mean)
    troch_sheet.write(troch_row,3,troch_dices_opp_mean)
    troch_sheet.write(troch_row,4,troch_overlaps_opp_mean)
    troch_row += 1
    troch_sheet.write(troch_row,0,"stdev opp")
    troch_sheet.write(troch_row,1,troch_dist_errors_opp_stdev)
    troch_sheet.write(troch_row,2,troch_total_rewards_opp_stdev)
    troch_sheet.write(troch_row,3,troch_dices_opp_stdev)
    troch_sheet.write(troch_row,4,troch_overlaps_opp_stdev)
    troch_row += 1

    troch_sheet.write(troch_row,0,"min F")
    troch_sheet.write(troch_row,1,troch_dist_errors_F_min)
    troch_sheet.write(troch_row,2,troch_total_rewards_F_min)
    troch_sheet.write(troch_row,3,troch_dices_F_min)
    troch_sheet.write(troch_row,4,troch_overlaps_F_min)
    troch_row += 1
    troch_sheet.write(troch_row,0,"max F")
    troch_sheet.write(troch_row,1,troch_dist_errors_F_max)
    troch_sheet.write(troch_row,2,troch_total_rewards_F_max)
    troch_sheet.write(troch_row,3,troch_dices_F_max)
    troch_sheet.write(troch_row,4,troch_overlaps_F_max)
    troch_row += 1
    troch_sheet.write(troch_row,0,"median F")
    troch_sheet.write(troch_row,1,troch_dist_errors_F_med)
    troch_sheet.write(troch_row,2,troch_total_rewards_F_med)
    troch_sheet.write(troch_row,3,troch_dices_F_med)
    troch_sheet.write(troch_row,4,troch_overlaps_F_med)
    troch_row += 1
    troch_sheet.write(troch_row,0,"mean F")
    troch_sheet.write(troch_row,1,troch_dist_errors_F_mean)
    troch_sheet.write(troch_row,2,troch_total_rewards_F_mean)
    troch_sheet.write(troch_row,3,troch_dices_F_mean)
    troch_sheet.write(troch_row,4,troch_overlaps_F_mean)
    troch_row += 1
    troch_sheet.write(troch_row,0,"stdev F")
    troch_sheet.write(troch_row,1,troch_dist_errors_F_stdev)
    troch_sheet.write(troch_row,2,troch_total_rewards_F_stdev)
    troch_sheet.write(troch_row,3,troch_dices_F_stdev)
    troch_sheet.write(troch_row,4,troch_overlaps_F_stdev)
    troch_row += 1

    troch_sheet.write(troch_row,0,"min W")
    troch_sheet.write(troch_row,1,troch_dist_errors_W_min)
    troch_sheet.write(troch_row,2,troch_total_rewards_W_min)
    troch_sheet.write(troch_row,3,troch_dices_W_min)
    troch_sheet.write(troch_row,4,troch_overlaps_W_min)
    troch_row += 1
    troch_sheet.write(troch_row,0,"max W")
    troch_sheet.write(troch_row,1,troch_dist_errors_W_max)
    troch_sheet.write(troch_row,2,troch_total_rewards_W_max)
    troch_sheet.write(troch_row,3,troch_dices_W_max)
    troch_sheet.write(troch_row,4,troch_overlaps_W_max)
    troch_row += 1
    troch_sheet.write(troch_row,0,"median W")
    troch_sheet.write(troch_row,1,troch_dist_errors_W_med)
    troch_sheet.write(troch_row,2,troch_total_rewards_W_med)
    troch_sheet.write(troch_row,3,troch_dices_W_med)
    troch_sheet.write(troch_row,4,troch_overlaps_W_med)
    troch_row += 1
    troch_sheet.write(troch_row,0,"mean W")
    troch_sheet.write(troch_row,1,troch_dist_errors_W_mean)
    troch_sheet.write(troch_row,2,troch_total_rewards_W_mean)
    troch_sheet.write(troch_row,3,troch_dices_W_mean)
    troch_sheet.write(troch_row,4,troch_overlaps_W_mean)
    troch_row += 1
    troch_sheet.write(troch_row,0,"stdev W")
    troch_sheet.write(troch_row,1,troch_dist_errors_W_stdev)
    troch_sheet.write(troch_row,2,troch_total_rewards_W_stdev)
    troch_sheet.write(troch_row,3,troch_dices_W_stdev)
    troch_sheet.write(troch_row,4,troch_overlaps_W_stdev)
    troch_row += 1

    troch_sheet.write(troch_row,0,"min T1 non fs")
    troch_sheet.write(troch_row,1,troch_dist_errors_T1_min)
    troch_sheet.write(troch_row,2,troch_total_rewards_T1_min)
    troch_sheet.write(troch_row,3,troch_dices_T1_min)
    troch_sheet.write(troch_row,4,troch_overlaps_T1_min)
    troch_row += 1
    troch_sheet.write(troch_row,0,"max T1 non fs")
    troch_sheet.write(troch_row,1,troch_dist_errors_T1_max)
    troch_sheet.write(troch_row,2,troch_total_rewards_T1_max)
    troch_sheet.write(troch_row,3,troch_dices_T1_max)
    troch_sheet.write(troch_row,4,troch_overlaps_T1_max)
    troch_row += 1
    troch_sheet.write(troch_row,0,"median T1 non fs")
    troch_sheet.write(troch_row,1,troch_dist_errors_T1_med)
    troch_sheet.write(troch_row,2,troch_total_rewards_T1_med)
    troch_sheet.write(troch_row,3,troch_dices_T1_med)
    troch_sheet.write(troch_row,4,troch_overlaps_T1_med)
    troch_row += 1
    troch_sheet.write(troch_row,0,"mean T1 non fs")
    troch_sheet.write(troch_row,1,troch_dist_errors_T1_mean)
    troch_sheet.write(troch_row,2,troch_total_rewards_T1_mean)
    troch_sheet.write(troch_row,3,troch_dices_T1_mean)
    troch_sheet.write(troch_row,4,troch_overlaps_T1_mean)
    troch_row += 1
    troch_sheet.write(troch_row,0,"stdev T1 non fs")
    troch_sheet.write(troch_row,1,troch_dist_errors_T1_stdev)
    troch_sheet.write(troch_row,2,troch_total_rewards_T1_stdev)
    troch_sheet.write(troch_row,3,troch_dices_T1_stdev)
    troch_sheet.write(troch_row,4,troch_overlaps_T1_stdev)
    troch_row += 1

    troch_sheet.write(troch_row,0,"min T2 haste tirm")
    troch_sheet.write(troch_row,1,troch_dist_errors_T2_min)
    troch_sheet.write(troch_row,2,troch_total_rewards_T2_min)
    troch_sheet.write(troch_row,3,troch_dices_T2_min)
    troch_sheet.write(troch_row,4,troch_overlaps_T2_min)
    troch_row += 1
    troch_sheet.write(troch_row,0,"max T2 haste tirm")
    troch_sheet.write(troch_row,1,troch_dist_errors_T2_max)
    troch_sheet.write(troch_row,2,troch_total_rewards_T2_max)
    troch_sheet.write(troch_row,3,troch_dices_T2_max)
    troch_sheet.write(troch_row,4,troch_overlaps_T2_max)
    troch_row += 1
    troch_sheet.write(troch_row,0,"median T2 haste tirm")
    troch_sheet.write(troch_row,1,troch_dist_errors_T2_med)
    troch_sheet.write(troch_row,2,troch_total_rewards_T2_med)
    troch_sheet.write(troch_row,3,troch_dices_T2_med)
    troch_sheet.write(troch_row,4,troch_overlaps_T2_med)
    troch_row += 1
    troch_sheet.write(troch_row,0,"mean T2 haste tirm")
    troch_sheet.write(troch_row,1,troch_dist_errors_T2_mean)
    troch_sheet.write(troch_row,2,troch_total_rewards_T2_mean)
    troch_sheet.write(troch_row,3,troch_dices_T2_mean)
    troch_sheet.write(troch_row,4,troch_overlaps_T2_mean)
    troch_row += 1
    troch_sheet.write(troch_row,0,"stdev T2 haste tirm")
    troch_sheet.write(troch_row,1,troch_dist_errors_T2_stdev)
    troch_sheet.write(troch_row,2,troch_total_rewards_T2_stdev)
    troch_sheet.write(troch_row,3,troch_dices_T2_stdev)
    troch_sheet.write(troch_row,4,troch_overlaps_T2_stdev)
    troch_row += 1




    ################## knee ###################
    knee_dist_errors_min = min(knee_dist_errors)
    knee_total_rewards_min = min(knee_total_rewards)
    knee_dices_min = min(knee_dices)
    knee_overlaps_min = min(knee_overlaps)
    knee_dist_errors_max = max(knee_dist_errors)
    knee_total_rewards_max = max(knee_total_rewards)
    knee_dices_max = max(knee_dices)
    knee_overlaps_max = max(knee_overlaps)
    knee_dist_errors_med = statistics.median(knee_dist_errors)
    knee_total_rewards_med = statistics.median(knee_total_rewards)
    knee_dices_med = statistics.median(knee_dices)
    knee_overlaps_med = statistics.median(knee_overlaps)
    knee_dist_errors_mean = statistics.mean(knee_dist_errors)
    knee_total_rewards_mean = statistics.mean(knee_total_rewards)
    knee_dices_mean = statistics.mean(knee_dices)
    knee_overlaps_mean = statistics.mean(knee_overlaps)
    knee_dist_errors_stdev = statistics.stdev(knee_dist_errors)
    knee_total_rewards_stdev = statistics.stdev(knee_total_rewards)
    knee_dices_stdev = statistics.stdev(knee_dices)
    knee_overlaps_stdev = statistics.stdev(knee_overlaps)

    knee_dist_errors_in_min = min(knee_dist_errors_in)
    knee_total_rewards_in_min = min(knee_total_rewards_in)
    knee_dices_in_min = min(knee_dices_in)
    knee_overlaps_in_min = min(knee_overlaps_in)
    knee_dist_errors_in_max = max(knee_dist_errors_in)
    knee_total_rewards_in_max = max(knee_total_rewards_in)
    knee_dices_in_max = max(knee_dices_in)
    knee_overlaps_in_max = max(knee_overlaps_in)
    knee_dist_errors_in_med = statistics.median(knee_dist_errors_in)
    knee_total_rewards_in_med = statistics.median(knee_total_rewards_in)
    knee_dices_in_med = statistics.median(knee_dices_in)
    knee_overlaps_in_med = statistics.median(knee_overlaps_in)
    knee_dist_errors_in_mean = statistics.mean(knee_dist_errors_in)
    knee_total_rewards_in_mean = statistics.mean(knee_total_rewards_in)
    knee_dices_in_mean = statistics.mean(knee_dices_in)
    knee_overlaps_in_mean = statistics.mean(knee_overlaps_in)
    knee_dist_errors_in_stdev = statistics.stdev(knee_dist_errors_in)
    knee_total_rewards_in_stdev = statistics.stdev(knee_total_rewards_in)
    knee_dices_in_stdev = statistics.stdev(knee_dices_in)
    knee_overlaps_in_stdev = statistics.stdev(knee_overlaps_in)

    knee_dist_errors_opp_min = min(knee_dist_errors_opp)
    knee_total_rewards_opp_min = min(knee_total_rewards_opp)
    knee_dices_opp_min = min(knee_dices_opp)
    knee_overlaps_opp_min = min(knee_overlaps_opp)
    knee_dist_errors_opp_max = max(knee_dist_errors_opp)
    knee_total_rewards_opp_max = max(knee_total_rewards_opp)
    knee_dices_opp_max = max(knee_dices_opp)
    knee_overlaps_opp_max = max(knee_overlaps_opp)
    knee_dist_errors_opp_med = statistics.median(knee_dist_errors_opp)
    knee_total_rewards_opp_med = statistics.median(knee_total_rewards_opp)
    knee_dices_opp_med = statistics.median(knee_dices_opp)
    knee_overlaps_opp_med = statistics.median(knee_overlaps_opp)
    knee_dist_errors_opp_mean = statistics.mean(knee_dist_errors_opp)
    knee_total_rewards_opp_mean = statistics.mean(knee_total_rewards_opp)
    knee_dices_opp_mean = statistics.mean(knee_dices_opp)
    knee_overlaps_opp_mean = statistics.mean(knee_overlaps_opp)
    knee_dist_errors_opp_stdev = statistics.stdev(knee_dist_errors_opp)
    knee_total_rewards_opp_stdev = statistics.stdev(knee_total_rewards_opp)
    knee_dices_opp_stdev = statistics.stdev(knee_dices_opp)
    knee_overlaps_opp_stdev = statistics.stdev(knee_overlaps_opp)

    knee_dist_errors_F_min = min(knee_dist_errors_F)
    knee_total_rewards_F_min = min(knee_total_rewards_F)
    knee_dices_F_min = min(knee_dices_F)
    knee_overlaps_F_min = min(knee_overlaps_F)
    knee_dist_errors_F_max = max(knee_dist_errors_F)
    knee_total_rewards_F_max = max(knee_total_rewards_F)
    knee_dices_F_max = max(knee_dices_F)
    knee_overlaps_F_max = max(knee_overlaps_F)
    knee_dist_errors_F_med = statistics.median(knee_dist_errors_F)
    knee_total_rewards_F_med = statistics.median(knee_total_rewards_F)
    knee_dices_F_med = statistics.median(knee_dices_F)
    knee_overlaps_F_med = statistics.median(knee_overlaps_F)
    knee_dist_errors_F_mean = statistics.mean(knee_dist_errors_F)
    knee_total_rewards_F_mean = statistics.mean(knee_total_rewards_F)
    knee_dices_F_mean = statistics.mean(knee_dices_F)
    knee_overlaps_F_mean = statistics.mean(knee_overlaps_F)
    knee_dist_errors_F_stdev = statistics.stdev(knee_dist_errors_F)
    knee_total_rewards_F_stdev = statistics.stdev(knee_total_rewards_F)
    knee_dices_F_stdev = statistics.stdev(knee_dices_F)
    knee_overlaps_F_stdev = statistics.stdev(knee_overlaps_F)

    knee_dist_errors_W_min = min(knee_dist_errors_W)
    knee_total_rewards_W_min = min(knee_total_rewards_W)
    knee_dices_W_min = min(knee_dices_W)
    knee_overlaps_W_min = min(knee_overlaps_W)
    knee_dist_errors_W_max = max(knee_dist_errors_W)
    knee_total_rewards_W_max = max(knee_total_rewards_W)
    knee_dices_W_max = max(knee_dices_W)
    knee_overlaps_W_max = max(knee_overlaps_W)
    knee_dist_errors_W_med = statistics.median(knee_dist_errors_W)
    knee_total_rewards_W_med = statistics.median(knee_total_rewards_W)
    knee_dices_W_med = statistics.median(knee_dices_W)
    knee_overlaps_W_med = statistics.median(knee_overlaps_W)
    knee_dist_errors_W_mean = statistics.mean(knee_dist_errors_W)
    knee_total_rewards_W_mean = statistics.mean(knee_total_rewards_W)
    knee_dices_W_mean = statistics.mean(knee_dices_W)
    knee_overlaps_W_mean = statistics.mean(knee_overlaps_W)
    knee_dist_errors_W_stdev = statistics.stdev(knee_dist_errors_W)
    knee_total_rewards_W_stdev = statistics.stdev(knee_total_rewards_W)
    knee_dices_W_stdev = statistics.stdev(knee_dices_W)
    knee_overlaps_W_stdev = statistics.stdev(knee_overlaps_W)

    knee_dist_errors_T1_min = min(knee_dist_errors_T1)
    knee_total_rewards_T1_min = min(knee_total_rewards_T1)
    knee_dices_T1_min = min(knee_dices_T1)
    knee_overlaps_T1_min = min(knee_overlaps_T1)
    knee_dist_errors_T1_max = max(knee_dist_errors_T1)
    knee_total_rewards_T1_max = max(knee_total_rewards_T1)
    knee_dices_T1_max = max(knee_dices_T1)
    knee_overlaps_T1_max = max(knee_overlaps_T1)
    knee_dist_errors_T1_med = statistics.median(knee_dist_errors_T1)
    knee_total_rewards_T1_med = statistics.median(knee_total_rewards_T1)
    knee_dices_T1_med = statistics.median(knee_dices_T1)
    knee_overlaps_T1_med = statistics.median(knee_overlaps_T1)
    knee_dist_errors_T1_mean = statistics.mean(knee_dist_errors_T1)
    knee_total_rewards_T1_mean = statistics.mean(knee_total_rewards_T1)
    knee_dices_T1_mean = statistics.mean(knee_dices_T1)
    knee_overlaps_T1_mean = statistics.mean(knee_overlaps_T1)
    knee_dist_errors_T1_stdev = statistics.stdev(knee_dist_errors_T1)
    knee_total_rewards_T1_stdev = statistics.stdev(knee_total_rewards_T1)
    knee_dices_T1_stdev = statistics.stdev(knee_dices_T1)
    knee_overlaps_T1_stdev = statistics.stdev(knee_overlaps_T1)

    knee_dist_errors_T2_min = min(knee_dist_errors_T2)
    knee_total_rewards_T2_min = min(knee_total_rewards_T2)
    knee_dices_T2_min = min(knee_dices_T2)
    knee_overlaps_T2_min = min(knee_overlaps_T2)
    knee_dist_errors_T2_max = max(knee_dist_errors_T2)
    knee_total_rewards_T2_max = max(knee_total_rewards_T2)
    knee_dices_T2_max = max(knee_dices_T2)
    knee_overlaps_T2_max = max(knee_overlaps_T2)
    knee_dist_errors_T2_med = statistics.median(knee_dist_errors_T2)
    knee_total_rewards_T2_med = statistics.median(knee_total_rewards_T2)
    knee_dices_T2_med = statistics.median(knee_dices_T2)
    knee_overlaps_T2_med = statistics.median(knee_overlaps_T2)
    knee_dist_errors_T2_mean = statistics.mean(knee_dist_errors_T2)
    knee_total_rewards_T2_mean = statistics.mean(knee_total_rewards_T2)
    knee_dices_T2_mean = statistics.mean(knee_dices_T2)
    knee_overlaps_T2_mean = statistics.mean(knee_overlaps_T2)
    knee_dist_errors_T2_stdev = statistics.stdev(knee_dist_errors_T2)
    knee_total_rewards_T2_stdev = statistics.stdev(knee_total_rewards_T2)
    knee_dices_T2_stdev = statistics.stdev(knee_dices_T2)
    knee_overlaps_T2_stdev = statistics.stdev(knee_overlaps_T2)

    knee_sheet.write(knee_row,0,"min all knee img params")
    knee_sheet.write(knee_row,1,knee_dist_errors_min)
    knee_sheet.write(knee_row,2,knee_total_rewards_min)
    knee_sheet.write(knee_row,3,knee_dices_min)
    knee_sheet.write(knee_row,4,knee_overlaps_min)
    knee_row += 1
    knee_sheet.write(knee_row,0,"max all knee img params")
    knee_sheet.write(knee_row,1,knee_dist_errors_max)
    knee_sheet.write(knee_row,2,knee_total_rewards_max)
    knee_sheet.write(knee_row,3,knee_dices_max)
    knee_sheet.write(knee_row,4,knee_overlaps_max)
    knee_row += 1
    knee_sheet.write(knee_row,0,"median all knee img params")
    knee_sheet.write(knee_row,1,knee_dist_errors_med)
    knee_sheet.write(knee_row,2,knee_total_rewards_med)
    knee_sheet.write(knee_row,3,knee_dices_med)
    knee_sheet.write(knee_row,4,knee_overlaps_med)
    knee_row += 1
    knee_sheet.write(knee_row,0,"mean all knee img params")
    knee_sheet.write(knee_row,1,knee_dist_errors_mean)
    knee_sheet.write(knee_row,2,knee_total_rewards_mean)
    knee_sheet.write(knee_row,3,knee_dices_mean)
    knee_sheet.write(knee_row,4,knee_overlaps_mean)
    knee_row += 1
    knee_sheet.write(knee_row,0,"stdev all knee img params")
    knee_sheet.write(knee_row,1,knee_dist_errors_stdev)
    knee_sheet.write(knee_row,2,knee_total_rewards_stdev)
    knee_sheet.write(knee_row,3,knee_dices_stdev)
    knee_sheet.write(knee_row,4,knee_overlaps_stdev)
    knee_row += 1

    knee_sheet.write(knee_row,0,"min in")
    knee_sheet.write(knee_row,1,knee_dist_errors_in_min)
    knee_sheet.write(knee_row,2,knee_total_rewards_in_min)
    knee_sheet.write(knee_row,3,knee_dices_in_min)
    knee_sheet.write(knee_row,4,knee_overlaps_in_min)
    knee_row += 1
    knee_sheet.write(knee_row,0,"max in")
    knee_sheet.write(knee_row,1,knee_dist_errors_in_max)
    knee_sheet.write(knee_row,2,knee_total_rewards_in_max)
    knee_sheet.write(knee_row,3,knee_dices_in_max)
    knee_sheet.write(knee_row,4,knee_overlaps_in_max)
    knee_row += 1
    knee_sheet.write(knee_row,0,"median in")
    knee_sheet.write(knee_row,1,knee_dist_errors_in_med)
    knee_sheet.write(knee_row,2,knee_total_rewards_in_med)
    knee_sheet.write(knee_row,3,knee_dices_in_med)
    knee_sheet.write(knee_row,4,knee_overlaps_in_med)
    knee_row += 1
    knee_sheet.write(knee_row,0,"mean in")
    knee_sheet.write(knee_row,1,knee_dist_errors_in_mean)
    knee_sheet.write(knee_row,2,knee_total_rewards_in_mean)
    knee_sheet.write(knee_row,3,knee_dices_in_mean)
    knee_sheet.write(knee_row,4,knee_overlaps_in_mean)
    knee_row += 1
    knee_sheet.write(knee_row,0,"stdev in")
    knee_sheet.write(knee_row,1,knee_dist_errors_in_stdev)
    knee_sheet.write(knee_row,2,knee_total_rewards_in_stdev)
    knee_sheet.write(knee_row,3,knee_dices_in_stdev)
    knee_sheet.write(knee_row,4,knee_overlaps_in_stdev)
    knee_row += 1

    knee_sheet.write(knee_row,0,"min opp")
    knee_sheet.write(knee_row,1,knee_dist_errors_opp_min)
    knee_sheet.write(knee_row,2,knee_total_rewards_opp_min)
    knee_sheet.write(knee_row,3,knee_dices_opp_min)
    knee_sheet.write(knee_row,4,knee_overlaps_opp_min)
    knee_row += 1
    knee_sheet.write(knee_row,0,"max opp")
    knee_sheet.write(knee_row,1,knee_dist_errors_opp_max)
    knee_sheet.write(knee_row,2,knee_total_rewards_opp_max)
    knee_sheet.write(knee_row,3,knee_dices_opp_max)
    knee_sheet.write(knee_row,4,knee_overlaps_opp_max)
    knee_row += 1
    knee_sheet.write(knee_row,0,"median opp")
    knee_sheet.write(knee_row,1,knee_dist_errors_opp_med)
    knee_sheet.write(knee_row,2,knee_total_rewards_opp_med)
    knee_sheet.write(knee_row,3,knee_dices_opp_med)
    knee_sheet.write(knee_row,4,knee_overlaps_opp_med)
    knee_row += 1
    knee_sheet.write(knee_row,0,"mean opp")
    knee_sheet.write(knee_row,1,knee_dist_errors_opp_mean)
    knee_sheet.write(knee_row,2,knee_total_rewards_opp_mean)
    knee_sheet.write(knee_row,3,knee_dices_opp_mean)
    knee_sheet.write(knee_row,4,knee_overlaps_opp_mean)
    knee_row += 1
    knee_sheet.write(knee_row,0,"stdev opp")
    knee_sheet.write(knee_row,1,knee_dist_errors_opp_stdev)
    knee_sheet.write(knee_row,2,knee_total_rewards_opp_stdev)
    knee_sheet.write(knee_row,3,knee_dices_opp_stdev)
    knee_sheet.write(knee_row,4,knee_overlaps_opp_stdev)
    knee_row += 1

    knee_sheet.write(knee_row,0,"min F")
    knee_sheet.write(knee_row,1,knee_dist_errors_F_min)
    knee_sheet.write(knee_row,2,knee_total_rewards_F_min)
    knee_sheet.write(knee_row,3,knee_dices_F_min)
    knee_sheet.write(knee_row,4,knee_overlaps_F_min)
    knee_row += 1
    knee_sheet.write(knee_row,0,"max F")
    knee_sheet.write(knee_row,1,knee_dist_errors_F_max)
    knee_sheet.write(knee_row,2,knee_total_rewards_F_max)
    knee_sheet.write(knee_row,3,knee_dices_F_max)
    knee_sheet.write(knee_row,4,knee_overlaps_F_max)
    knee_row += 1
    knee_sheet.write(knee_row,0,"median F")
    knee_sheet.write(knee_row,1,knee_dist_errors_F_med)
    knee_sheet.write(knee_row,2,knee_total_rewards_F_med)
    knee_sheet.write(knee_row,3,knee_dices_F_med)
    knee_sheet.write(knee_row,4,knee_overlaps_F_med)
    knee_row += 1
    knee_sheet.write(knee_row,0,"mean F")
    knee_sheet.write(knee_row,1,knee_dist_errors_F_mean)
    knee_sheet.write(knee_row,2,knee_total_rewards_F_mean)
    knee_sheet.write(knee_row,3,knee_dices_F_mean)
    knee_sheet.write(knee_row,4,knee_overlaps_F_mean)
    knee_row += 1
    knee_sheet.write(knee_row,0,"stdev F")
    knee_sheet.write(knee_row,1,knee_dist_errors_F_stdev)
    knee_sheet.write(knee_row,2,knee_total_rewards_F_stdev)
    knee_sheet.write(knee_row,3,knee_dices_F_stdev)
    knee_sheet.write(knee_row,4,knee_overlaps_F_stdev)
    knee_row += 1

    knee_sheet.write(knee_row,0,"min W")
    knee_sheet.write(knee_row,1,knee_dist_errors_W_min)
    knee_sheet.write(knee_row,2,knee_total_rewards_W_min)
    knee_sheet.write(knee_row,3,knee_dices_W_min)
    knee_sheet.write(knee_row,4,knee_overlaps_W_min)
    knee_row += 1
    knee_sheet.write(knee_row,0,"max W")
    knee_sheet.write(knee_row,1,knee_dist_errors_W_max)
    knee_sheet.write(knee_row,2,knee_total_rewards_W_max)
    knee_sheet.write(knee_row,3,knee_dices_W_max)
    knee_sheet.write(knee_row,4,knee_overlaps_W_max)
    knee_row += 1
    knee_sheet.write(knee_row,0,"median W")
    knee_sheet.write(knee_row,1,knee_dist_errors_W_med)
    knee_sheet.write(knee_row,2,knee_total_rewards_W_med)
    knee_sheet.write(knee_row,3,knee_dices_W_med)
    knee_sheet.write(knee_row,4,knee_overlaps_W_med)
    knee_row += 1
    knee_sheet.write(knee_row,0,"mean W")
    knee_sheet.write(knee_row,1,knee_dist_errors_W_mean)
    knee_sheet.write(knee_row,2,knee_total_rewards_W_mean)
    knee_sheet.write(knee_row,3,knee_dices_W_mean)
    knee_sheet.write(knee_row,4,knee_overlaps_W_mean)
    knee_row += 1
    knee_sheet.write(knee_row,0,"stdev W")
    knee_sheet.write(knee_row,1,knee_dist_errors_W_stdev)
    knee_sheet.write(knee_row,2,knee_total_rewards_W_stdev)
    knee_sheet.write(knee_row,3,knee_dices_W_stdev)
    knee_sheet.write(knee_row,4,knee_overlaps_W_stdev)
    knee_row += 1

    knee_sheet.write(knee_row,0,"min T1 non fs")
    knee_sheet.write(knee_row,1,knee_dist_errors_T1_min)
    knee_sheet.write(knee_row,2,knee_total_rewards_T1_min)
    knee_sheet.write(knee_row,3,knee_dices_T1_min)
    knee_sheet.write(knee_row,4,knee_overlaps_T1_min)
    knee_row += 1
    knee_sheet.write(knee_row,0,"max T1 non fs")
    knee_sheet.write(knee_row,1,knee_dist_errors_T1_max)
    knee_sheet.write(knee_row,2,knee_total_rewards_T1_max)
    knee_sheet.write(knee_row,3,knee_dices_T1_max)
    knee_sheet.write(knee_row,4,knee_overlaps_T1_max)
    knee_row += 1
    knee_sheet.write(knee_row,0,"median T1 non fs")
    knee_sheet.write(knee_row,1,knee_dist_errors_T1_med)
    knee_sheet.write(knee_row,2,knee_total_rewards_T1_med)
    knee_sheet.write(knee_row,3,knee_dices_T1_med)
    knee_sheet.write(knee_row,4,knee_overlaps_T1_med)
    knee_row += 1
    knee_sheet.write(knee_row,0,"mean T1 non fs")
    knee_sheet.write(knee_row,1,knee_dist_errors_T1_mean)
    knee_sheet.write(knee_row,2,knee_total_rewards_T1_mean)
    knee_sheet.write(knee_row,3,knee_dices_T1_mean)
    knee_sheet.write(knee_row,4,knee_overlaps_T1_mean)
    knee_row += 1
    knee_sheet.write(knee_row,0,"stdev T1 non fs")
    knee_sheet.write(knee_row,1,knee_dist_errors_T1_stdev)
    knee_sheet.write(knee_row,2,knee_total_rewards_T1_stdev)
    knee_sheet.write(knee_row,3,knee_dices_T1_stdev)
    knee_sheet.write(knee_row,4,knee_overlaps_T1_stdev)
    knee_row += 1

    knee_sheet.write(knee_row,0,"min T2 haste tirm")
    knee_sheet.write(knee_row,1,knee_dist_errors_T2_min)
    knee_sheet.write(knee_row,2,knee_total_rewards_T2_min)
    knee_sheet.write(knee_row,3,knee_dices_T2_min)
    knee_sheet.write(knee_row,4,knee_overlaps_T2_min)
    knee_row += 1
    knee_sheet.write(knee_row,0,"max T2 haste tirm")
    knee_sheet.write(knee_row,1,knee_dist_errors_T2_max)
    knee_sheet.write(knee_row,2,knee_total_rewards_T2_max)
    knee_sheet.write(knee_row,3,knee_dices_T2_max)
    knee_sheet.write(knee_row,4,knee_overlaps_T2_max)
    knee_row += 1
    knee_sheet.write(knee_row,0,"median T2 haste tirm")
    knee_sheet.write(knee_row,1,knee_dist_errors_T2_med)
    knee_sheet.write(knee_row,2,knee_total_rewards_T2_med)
    knee_sheet.write(knee_row,3,knee_dices_T2_med)
    knee_sheet.write(knee_row,4,knee_overlaps_T2_med)
    knee_row += 1
    knee_sheet.write(knee_row,0,"mean T2 haste tirm")
    knee_sheet.write(knee_row,1,knee_dist_errors_T2_mean)
    knee_sheet.write(knee_row,2,knee_total_rewards_T2_mean)
    knee_sheet.write(knee_row,3,knee_dices_T2_mean)
    knee_sheet.write(knee_row,4,knee_overlaps_T2_mean)
    knee_row += 1
    knee_sheet.write(knee_row,0,"stdev T2 haste tirm")
    knee_sheet.write(knee_row,1,knee_dist_errors_T2_stdev)
    knee_sheet.write(knee_row,2,knee_total_rewards_T2_stdev)
    knee_sheet.write(knee_row,3,knee_dices_T2_stdev)
    knee_sheet.write(knee_row,4,knee_overlaps_T2_stdev)
    knee_row += 1



    ################ prostate ##################
    pros_dist_errors_min = min(pros_dist_errors)
    pros_total_rewards_min = min(pros_total_rewards)
    pros_dices_min = min(pros_dices)
    pros_overlaps_min = min(pros_overlaps)
    pros_dist_errors_max = max(pros_dist_errors)
    pros_total_rewards_max = max(pros_total_rewards)
    pros_dices_max = max(pros_dices)
    pros_overlaps_max = max(pros_overlaps)
    pros_dist_errors_med = statistics.median(pros_dist_errors)
    pros_total_rewards_med = statistics.median(pros_total_rewards)
    pros_dices_med = statistics.median(pros_dices)
    pros_overlaps_med = statistics.median(pros_overlaps)
    pros_dist_errors_mean = statistics.mean(pros_dist_errors)
    pros_total_rewards_mean = statistics.mean(pros_total_rewards)
    pros_dices_mean = statistics.mean(pros_dices)
    pros_overlaps_mean = statistics.mean(pros_overlaps)
    pros_dist_errors_stdev = statistics.stdev(pros_dist_errors)
    pros_total_rewards_stdev = statistics.stdev(pros_total_rewards)
    pros_dices_stdev = statistics.stdev(pros_dices)
    pros_overlaps_stdev = statistics.stdev(pros_overlaps)

    pros_dist_errors_adc_min = min(pros_dist_errors_adc)
    pros_total_rewards_adc_min = min(pros_total_rewards_adc)
    pros_dices_adc_min = min(pros_dices_adc)
    pros_overlaps_adc_min = min(pros_overlaps_adc)
    pros_dist_errors_adc_max = max(pros_dist_errors_adc)
    pros_total_rewards_adc_max = max(pros_total_rewards_adc)
    pros_dices_adc_max = max(pros_dices_adc)
    pros_overlaps_adc_max = max(pros_overlaps_adc)
    pros_dist_errors_adc_med = statistics.median(pros_dist_errors_adc)
    pros_total_rewards_adc_med = statistics.median(pros_total_rewards_adc)
    pros_dices_adc_med = statistics.median(pros_dices_adc)
    pros_overlaps_adc_med = statistics.median(pros_overlaps_adc)
    pros_dist_errors_adc_mean = statistics.mean(pros_dist_errors_adc)
    pros_total_rewards_adc_mean = statistics.mean(pros_total_rewards_adc)
    pros_dices_adc_mean = statistics.mean(pros_dices_adc)
    pros_overlaps_adc_mean = statistics.mean(pros_overlaps_adc)
    pros_dist_errors_adc_stdev = statistics.stdev(pros_dist_errors_adc)
    pros_total_rewards_adc_stdev = statistics.stdev(pros_total_rewards_adc)
    pros_dices_adc_stdev = statistics.stdev(pros_dices_adc)
    pros_overlaps_adc_stdev = statistics.stdev(pros_overlaps_adc)

    pros_dist_errors_T2_min = min(pros_dist_errors_T2)
    pros_total_rewards_T2_min = min(pros_total_rewards_T2)
    pros_dices_T2_min = min(pros_dices_T2)
    pros_overlaps_T2_min = min(pros_overlaps_T2)
    pros_dist_errors_T2_max = max(pros_dist_errors_T2)
    pros_total_rewards_T2_max = max(pros_total_rewards_T2)
    pros_dices_T2_max = max(pros_dices_T2)
    pros_overlaps_T2_max = max(pros_overlaps_T2)
    pros_dist_errors_T2_med = statistics.median(pros_dist_errors_T2)
    pros_total_rewards_T2_med = statistics.median(pros_total_rewards_T2)
    pros_dices_T2_med = statistics.median(pros_dices_T2)
    pros_overlaps_T2_med = statistics.median(pros_overlaps_T2)
    pros_dist_errors_T2_mean = statistics.mean(pros_dist_errors_T2)
    pros_total_rewards_T2_mean = statistics.mean(pros_total_rewards_T2)
    pros_dices_T2_mean = statistics.mean(pros_dices_T2)
    pros_overlaps_T2_mean = statistics.mean(pros_overlaps_T2)
    pros_dist_errors_T2_stdev = statistics.stdev(pros_dist_errors_T2)
    pros_total_rewards_T2_stdev = statistics.stdev(pros_total_rewards_T2)
    pros_dices_T2_stdev = statistics.stdev(pros_dices_T2)
    pros_overlaps_T2_stdev = statistics.stdev(pros_overlaps_T2)

    pros_sheet.write(pros_row,0,"min all pros img params")
    pros_sheet.write(pros_row,1,pros_dist_errors_min)
    pros_sheet.write(pros_row,2,pros_total_rewards_min)
    pros_sheet.write(pros_row,3,pros_dices_min)
    pros_sheet.write(pros_row,4,pros_overlaps_min)
    pros_row += 1
    pros_sheet.write(pros_row,0,"max all pros img params")
    pros_sheet.write(pros_row,1,pros_dist_errors_max)
    pros_sheet.write(pros_row,2,pros_total_rewards_max)
    pros_sheet.write(pros_row,3,pros_dices_max)
    pros_sheet.write(pros_row,4,pros_overlaps_max)
    pros_row += 1
    pros_sheet.write(pros_row,0,"median all pros img params")
    pros_sheet.write(pros_row,1,pros_dist_errors_med)
    pros_sheet.write(pros_row,2,pros_total_rewards_med)
    pros_sheet.write(pros_row,3,pros_dices_med)
    pros_sheet.write(pros_row,4,pros_overlaps_med)
    pros_row += 1
    pros_sheet.write(pros_row,0,"mean all pros img params")
    pros_sheet.write(pros_row,1,pros_dist_errors_mean)
    pros_sheet.write(pros_row,2,pros_total_rewards_mean)
    pros_sheet.write(pros_row,3,pros_dices_mean)
    pros_sheet.write(pros_row,4,pros_overlaps_mean)
    pros_row += 1
    pros_sheet.write(pros_row,0,"stdev all pros img params")
    pros_sheet.write(pros_row,1,pros_dist_errors_stdev)
    pros_sheet.write(pros_row,2,pros_total_rewards_stdev)
    pros_sheet.write(pros_row,3,pros_dices_stdev)
    pros_sheet.write(pros_row,4,pros_overlaps_stdev)
    pros_row += 1

    pros_sheet.write(pros_row,0,"min adc")
    pros_sheet.write(pros_row,1,pros_dist_errors_adc_min)
    pros_sheet.write(pros_row,2,pros_total_rewards_adc_min)
    pros_sheet.write(pros_row,3,pros_dices_adc_min)
    pros_sheet.write(pros_row,4,pros_overlaps_adc_min)
    pros_row += 1
    pros_sheet.write(pros_row,0,"max adc")
    pros_sheet.write(pros_row,1,pros_dist_errors_adc_max)
    pros_sheet.write(pros_row,2,pros_total_rewards_adc_max)
    pros_sheet.write(pros_row,3,pros_dices_adc_max)
    pros_sheet.write(pros_row,4,pros_overlaps_adc_max)
    pros_row += 1
    pros_sheet.write(pros_row,0,"median adc")
    pros_sheet.write(pros_row,1,pros_dist_errors_adc_med)
    pros_sheet.write(pros_row,2,pros_total_rewards_adc_med)
    pros_sheet.write(pros_row,3,pros_dices_adc_med)
    pros_sheet.write(pros_row,4,pros_overlaps_adc_med)
    pros_row += 1
    pros_sheet.write(pros_row,0,"mean adc")
    pros_sheet.write(pros_row,1,pros_dist_errors_adc_mean)
    pros_sheet.write(pros_row,2,pros_total_rewards_adc_mean)
    pros_sheet.write(pros_row,3,pros_dices_adc_mean)
    pros_sheet.write(pros_row,4,pros_overlaps_adc_mean)
    pros_row += 1
    pros_sheet.write(pros_row,0,"stdev adc")
    pros_sheet.write(pros_row,1,pros_dist_errors_adc_stdev)
    pros_sheet.write(pros_row,2,pros_total_rewards_adc_stdev)
    pros_sheet.write(pros_row,3,pros_dices_adc_stdev)
    pros_sheet.write(pros_row,4,pros_overlaps_adc_stdev)
    pros_row += 1

    pros_sheet.write(pros_row,0,"min T2")
    pros_sheet.write(pros_row,1,pros_dist_errors_T2_min)
    pros_sheet.write(pros_row,2,pros_total_rewards_T2_min)
    pros_sheet.write(pros_row,3,pros_dices_T2_min)
    pros_sheet.write(pros_row,4,pros_overlaps_T2_min)
    pros_row += 1
    pros_sheet.write(pros_row,0,"max T2")
    pros_sheet.write(pros_row,1,pros_dist_errors_T2_max)
    pros_sheet.write(pros_row,2,pros_total_rewards_T2_max)
    pros_sheet.write(pros_row,3,pros_dices_T2_max)
    pros_sheet.write(pros_row,4,pros_overlaps_T2_max)
    pros_row += 1
    pros_sheet.write(pros_row,0,"median T2")
    pros_sheet.write(pros_row,1,pros_dist_errors_T2_med)
    pros_sheet.write(pros_row,2,pros_total_rewards_T2_med)
    pros_sheet.write(pros_row,3,pros_dices_T2_med)
    pros_sheet.write(pros_row,4,pros_overlaps_T2_med)
    pros_row += 1
    pros_sheet.write(pros_row,0,"mean T2")
    pros_sheet.write(pros_row,1,pros_dist_errors_T2_mean)
    pros_sheet.write(pros_row,2,pros_total_rewards_T2_mean)
    pros_sheet.write(pros_row,3,pros_dices_T2_mean)
    pros_sheet.write(pros_row,4,pros_overlaps_T2_mean)
    pros_row += 1
    pros_sheet.write(pros_row,0,"stdev T2")
    pros_sheet.write(pros_row,1,pros_dist_errors_T2_stdev)
    pros_sheet.write(pros_row,2,pros_total_rewards_T2_stdev)
    pros_sheet.write(pros_row,3,pros_dices_T2_stdev)
    pros_sheet.write(pros_row,4,pros_overlaps_T2_stdev)
    pros_row += 1



    ################## breast ###################
    breast_dist_errors_min = min(breast_dist_errors)
    breast_total_rewards_min = min(breast_total_rewards)
    breast_dices_min = min(breast_dices)
    breast_overlaps_min = min(breast_overlaps)
    breast_dist_errors_max = max(breast_dist_errors)
    breast_total_rewards_max = max(breast_total_rewards)
    breast_dices_max = max(breast_dices)
    breast_overlaps_max = max(breast_overlaps)
    breast_dist_errors_med = statistics.median(breast_dist_errors)
    breast_total_rewards_med = statistics.median(breast_total_rewards)
    breast_dices_med = statistics.median(breast_dices)
    breast_overlaps_med = statistics.median(breast_overlaps)
    breast_dist_errors_mean = statistics.mean(breast_dist_errors)
    breast_total_rewards_mean = statistics.mean(breast_total_rewards)
    breast_dices_mean = statistics.mean(breast_dices)
    breast_overlaps_mean = statistics.mean(breast_overlaps)
    breast_dist_errors_stdev = statistics.stdev(breast_dist_errors)
    breast_total_rewards_stdev = statistics.stdev(breast_total_rewards)
    breast_dices_stdev = statistics.stdev(breast_dices)
    breast_overlaps_stdev = statistics.stdev(breast_overlaps)

    breast_dist_errors_Post_min = min(breast_dist_errors_Post)
    breast_total_rewards_Post_min = min(breast_total_rewards_Post)
    breast_dices_Post_min = min(breast_dices_Post)
    breast_overlaps_Post_min = min(breast_overlaps_Post)
    breast_dist_errors_Post_max = max(breast_dist_errors_Post)
    breast_total_rewards_Post_max = max(breast_total_rewards_Post)
    breast_dices_Post_max = max(breast_dices_Post)
    breast_overlaps_Post_max = max(breast_overlaps_Post)
    breast_dist_errors_Post_med = statistics.median(breast_dist_errors_Post)
    breast_total_rewards_Post_med = statistics.median(breast_total_rewards_Post)
    breast_dices_Post_med = statistics.median(breast_dices_Post)
    breast_overlaps_Post_med = statistics.median(breast_overlaps_Post)
    breast_dist_errors_Post_mean = statistics.mean(breast_dist_errors_Post)
    breast_total_rewards_Post_mean = statistics.mean(breast_total_rewards_Post)
    breast_dices_Post_mean = statistics.mean(breast_dices_Post)
    breast_overlaps_Post_mean = statistics.mean(breast_overlaps_Post)
    breast_dist_errors_Post_stdev = statistics.stdev(breast_dist_errors_Post)
    breast_total_rewards_Post_stdev = statistics.stdev(breast_total_rewards_Post)
    breast_dices_Post_stdev = statistics.stdev(breast_dices_Post)
    breast_overlaps_Post_stdev = statistics.stdev(breast_overlaps_Post)

    breast_dist_errors_Pre_min = min(breast_dist_errors_Pre)
    breast_total_rewards_Pre_min = min(breast_total_rewards_Pre)
    breast_dices_Pre_min = min(breast_dices_Pre)
    breast_overlaps_Pre_min = min(breast_overlaps_Pre)
    breast_dist_errors_Pre_max = max(breast_dist_errors_Pre)
    breast_total_rewards_Pre_max = max(breast_total_rewards_Pre)
    breast_dices_Pre_max = max(breast_dices_Pre)
    breast_overlaps_Pre_max = max(breast_overlaps_Pre)
    breast_dist_errors_Pre_med = statistics.median(breast_dist_errors_Pre)
    breast_total_rewards_Pre_med = statistics.median(breast_total_rewards_Pre)
    breast_dices_Pre_med = statistics.median(breast_dices_Pre)
    breast_overlaps_Pre_med = statistics.median(breast_overlaps_Pre)
    breast_dist_errors_Pre_mean = statistics.mean(breast_dist_errors_Pre)
    breast_total_rewards_Pre_mean = statistics.mean(breast_total_rewards_Pre)
    breast_dices_Pre_mean = statistics.mean(breast_dices_Pre)
    breast_overlaps_Pre_mean = statistics.mean(breast_overlaps_Pre)
    breast_dist_errors_Pre_stdev = statistics.stdev(breast_dist_errors_Pre)
    breast_total_rewards_Pre_stdev = statistics.stdev(breast_total_rewards_Pre)
    breast_dices_Pre_stdev = statistics.stdev(breast_dices_Pre)
    breast_overlaps_Pre_stdev = statistics.stdev(breast_overlaps_Pre)

    breast_dist_errors_SUB_min = min(breast_dist_errors_SUB)
    breast_total_rewards_SUB_min = min(breast_total_rewards_SUB)
    breast_dices_SUB_min = min(breast_dices_SUB)
    breast_overlaps_SUB_min = min(breast_overlaps_SUB)
    breast_dist_errors_SUB_max = max(breast_dist_errors_SUB)
    breast_total_rewards_SUB_max = max(breast_total_rewards_SUB)
    breast_dices_SUB_max = max(breast_dices_SUB)
    breast_overlaps_SUB_max = max(breast_overlaps_SUB)
    breast_dist_errors_SUB_med = statistics.median(breast_dist_errors_SUB)
    breast_total_rewards_SUB_med = statistics.median(breast_total_rewards_SUB)
    breast_dices_SUB_med = statistics.median(breast_dices_SUB)
    breast_overlaps_SUB_med = statistics.median(breast_overlaps_SUB)
    breast_dist_errors_SUB_mean = statistics.mean(breast_dist_errors_SUB)
    breast_total_rewards_SUB_mean = statistics.mean(breast_total_rewards_SUB)
    breast_dices_SUB_mean = statistics.mean(breast_dices_SUB)
    breast_overlaps_SUB_mean = statistics.mean(breast_overlaps_SUB)
    breast_dist_errors_SUB_stdev = statistics.stdev(breast_dist_errors_SUB)
    breast_total_rewards_SUB_stdev = statistics.stdev(breast_total_rewards_SUB)
    breast_dices_SUB_stdev = statistics.stdev(breast_dices_SUB)
    breast_overlaps_SUB_stdev = statistics.stdev(breast_overlaps_SUB)

    breast_dist_errors_T1_min = min(breast_dist_errors_T1)
    breast_total_rewards_T1_min = min(breast_total_rewards_T1)
    breast_dices_T1_min = min(breast_dices_T1)
    breast_overlaps_T1_min = min(breast_overlaps_T1)
    breast_dist_errors_T1_max = max(breast_dist_errors_T1)
    breast_total_rewards_T1_max = max(breast_total_rewards_T1)
    breast_dices_T1_max = max(breast_dices_T1)
    breast_overlaps_T1_max = max(breast_overlaps_T1)
    breast_dist_errors_T1_med = statistics.median(breast_dist_errors_T1)
    breast_total_rewards_T1_med = statistics.median(breast_total_rewards_T1)
    breast_dices_T1_med = statistics.median(breast_dices_T1)
    breast_overlaps_T1_med = statistics.median(breast_overlaps_T1)
    breast_dist_errors_T1_mean = statistics.mean(breast_dist_errors_T1)
    breast_total_rewards_T1_mean = statistics.mean(breast_total_rewards_T1)
    breast_dices_T1_mean = statistics.mean(breast_dices_T1)
    breast_overlaps_T1_mean = statistics.mean(breast_overlaps_T1)
    breast_dist_errors_T1_stdev = statistics.stdev(breast_dist_errors_T1)
    breast_total_rewards_T1_stdev = statistics.stdev(breast_total_rewards_T1)
    breast_dices_T1_stdev = statistics.stdev(breast_dices_T1)
    breast_overlaps_T1_stdev = statistics.stdev(breast_overlaps_T1)

    breast_dist_errors_T2_min = min(breast_dist_errors_T2)
    breast_total_rewards_T2_min = min(breast_total_rewards_T2)
    breast_dices_T2_min = min(breast_dices_T2)
    breast_overlaps_T2_min = min(breast_overlaps_T2)
    breast_dist_errors_T2_max = max(breast_dist_errors_T2)
    breast_total_rewards_T2_max = max(breast_total_rewards_T2)
    breast_dices_T2_max = max(breast_dices_T2)
    breast_overlaps_T2_max = max(breast_overlaps_T2)
    breast_dist_errors_T2_med = statistics.median(breast_dist_errors_T2)
    breast_total_rewards_T2_med = statistics.median(breast_total_rewards_T2)
    breast_dices_T2_med = statistics.median(breast_dices_T2)
    breast_overlaps_T2_med = statistics.median(breast_overlaps_T2)
    breast_dist_errors_T2_mean = statistics.mean(breast_dist_errors_T2)
    breast_total_rewards_T2_mean = statistics.mean(breast_total_rewards_T2)
    breast_dices_T2_mean = statistics.mean(breast_dices_T2)
    breast_overlaps_T2_mean = statistics.mean(breast_overlaps_T2)
    breast_dist_errors_T2_stdev = statistics.stdev(breast_dist_errors_T2)
    breast_total_rewards_T2_stdev = statistics.stdev(breast_total_rewards_T2)
    breast_dices_T2_stdev = statistics.stdev(breast_dices_T2)
    breast_overlaps_T2_stdev = statistics.stdev(breast_overlaps_T2)

    breast_sheet.write(breast_row,0,"min all breast img params")
    breast_sheet.write(breast_row,1,breast_dist_errors_min)
    breast_sheet.write(breast_row,2,breast_total_rewards_min)
    breast_sheet.write(breast_row,3,breast_dices_min)
    breast_sheet.write(breast_row,4,breast_overlaps_min)
    breast_row += 1
    breast_sheet.write(breast_row,0,"max all breast img params")
    breast_sheet.write(breast_row,1,breast_dist_errors_max)
    breast_sheet.write(breast_row,2,breast_total_rewards_max)
    breast_sheet.write(breast_row,3,breast_dices_max)
    breast_sheet.write(breast_row,4,breast_overlaps_max)
    breast_row += 1
    breast_sheet.write(breast_row,0,"median all breast img params")
    breast_sheet.write(breast_row,1,breast_dist_errors_med)
    breast_sheet.write(breast_row,2,breast_total_rewards_med)
    breast_sheet.write(breast_row,3,breast_dices_med)
    breast_sheet.write(breast_row,4,breast_overlaps_med)
    breast_row += 1
    breast_sheet.write(breast_row,0,"mean all breast img params")
    breast_sheet.write(breast_row,1,breast_dist_errors_mean)
    breast_sheet.write(breast_row,2,breast_total_rewards_mean)
    breast_sheet.write(breast_row,3,breast_dices_mean)
    breast_sheet.write(breast_row,4,breast_overlaps_mean)
    breast_row += 1
    breast_sheet.write(breast_row,0,"stdev all breast img params")
    breast_sheet.write(breast_row,1,breast_dist_errors_stdev)
    breast_sheet.write(breast_row,2,breast_total_rewards_stdev)
    breast_sheet.write(breast_row,3,breast_dices_stdev)
    breast_sheet.write(breast_row,4,breast_overlaps_stdev)
    breast_row += 1

    breast_sheet.write(breast_row,0,"min Post")
    breast_sheet.write(breast_row,1,breast_dist_errors_Post_min)
    breast_sheet.write(breast_row,2,breast_total_rewards_Post_min)
    breast_sheet.write(breast_row,3,breast_dices_Post_min)
    breast_sheet.write(breast_row,4,breast_overlaps_Post_min)
    breast_row += 1
    breast_sheet.write(breast_row,0,"max Post")
    breast_sheet.write(breast_row,1,breast_dist_errors_Post_max)
    breast_sheet.write(breast_row,2,breast_total_rewards_Post_max)
    breast_sheet.write(breast_row,3,breast_dices_Post_max)
    breast_sheet.write(breast_row,4,breast_overlaps_Post_max)
    breast_row += 1
    breast_sheet.write(breast_row,0,"median Post")
    breast_sheet.write(breast_row,1,breast_dist_errors_Post_med)
    breast_sheet.write(breast_row,2,breast_total_rewards_Post_med)
    breast_sheet.write(breast_row,3,breast_dices_Post_med)
    breast_sheet.write(breast_row,4,breast_overlaps_Post_med)
    breast_row += 1
    breast_sheet.write(breast_row,0,"mean Post")
    breast_sheet.write(breast_row,1,breast_dist_errors_Post_mean)
    breast_sheet.write(breast_row,2,breast_total_rewards_Post_mean)
    breast_sheet.write(breast_row,3,breast_dices_Post_mean)
    breast_sheet.write(breast_row,4,breast_overlaps_Post_mean)
    breast_row += 1
    breast_sheet.write(breast_row,0,"stdev Post")
    breast_sheet.write(breast_row,1,breast_dist_errors_Post_stdev)
    breast_sheet.write(breast_row,2,breast_total_rewards_Post_stdev)
    breast_sheet.write(breast_row,3,breast_dices_Post_stdev)
    breast_sheet.write(breast_row,4,breast_overlaps_Post_stdev)
    breast_row += 1

    breast_sheet.write(breast_row,0,"min Pre")
    breast_sheet.write(breast_row,1,breast_dist_errors_Pre_min)
    breast_sheet.write(breast_row,2,breast_total_rewards_Pre_min)
    breast_sheet.write(breast_row,3,breast_dices_Pre_min)
    breast_sheet.write(breast_row,4,breast_overlaps_Pre_min)
    breast_row += 1
    breast_sheet.write(breast_row,0,"max Pre")
    breast_sheet.write(breast_row,1,breast_dist_errors_Pre_max)
    breast_sheet.write(breast_row,2,breast_total_rewards_Pre_max)
    breast_sheet.write(breast_row,3,breast_dices_Pre_max)
    breast_sheet.write(breast_row,4,breast_overlaps_Pre_max)
    breast_row += 1
    breast_sheet.write(breast_row,0,"median Pre")
    breast_sheet.write(breast_row,1,breast_dist_errors_Pre_med)
    breast_sheet.write(breast_row,2,breast_total_rewards_Pre_med)
    breast_sheet.write(breast_row,3,breast_dices_Pre_med)
    breast_sheet.write(breast_row,4,breast_overlaps_Pre_med)
    breast_row += 1
    breast_sheet.write(breast_row,0,"mean Pre")
    breast_sheet.write(breast_row,1,breast_dist_errors_Pre_mean)
    breast_sheet.write(breast_row,2,breast_total_rewards_Pre_mean)
    breast_sheet.write(breast_row,3,breast_dices_Pre_mean)
    breast_sheet.write(breast_row,4,breast_overlaps_Pre_mean)
    breast_row += 1
    breast_sheet.write(breast_row,0,"stdev Pre")
    breast_sheet.write(breast_row,1,breast_dist_errors_Pre_stdev)
    breast_sheet.write(breast_row,2,breast_total_rewards_Pre_stdev)
    breast_sheet.write(breast_row,3,breast_dices_Pre_stdev)
    breast_sheet.write(breast_row,4,breast_overlaps_Pre_stdev)
    breast_row += 1

    breast_sheet.write(breast_row,0,"min SUB")
    breast_sheet.write(breast_row,1,breast_dist_errors_SUB_min)
    breast_sheet.write(breast_row,2,breast_total_rewards_SUB_min)
    breast_sheet.write(breast_row,3,breast_dices_SUB_min)
    breast_sheet.write(breast_row,4,breast_overlaps_SUB_min)
    breast_row += 1
    breast_sheet.write(breast_row,0,"max SUB")
    breast_sheet.write(breast_row,1,breast_dist_errors_SUB_max)
    breast_sheet.write(breast_row,2,breast_total_rewards_SUB_max)
    breast_sheet.write(breast_row,3,breast_dices_SUB_max)
    breast_sheet.write(breast_row,4,breast_overlaps_SUB_max)
    breast_row += 1
    breast_sheet.write(breast_row,0,"median SUB")
    breast_sheet.write(breast_row,1,breast_dist_errors_SUB_med)
    breast_sheet.write(breast_row,2,breast_total_rewards_SUB_med)
    breast_sheet.write(breast_row,3,breast_dices_SUB_med)
    breast_sheet.write(breast_row,4,breast_overlaps_SUB_med)
    breast_row += 1
    breast_sheet.write(breast_row,0,"mean SUB")
    breast_sheet.write(breast_row,1,breast_dist_errors_SUB_mean)
    breast_sheet.write(breast_row,2,breast_total_rewards_SUB_mean)
    breast_sheet.write(breast_row,3,breast_dices_SUB_mean)
    breast_sheet.write(breast_row,4,breast_overlaps_SUB_mean)
    breast_row += 1
    breast_sheet.write(breast_row,0,"stdev SUB")
    breast_sheet.write(breast_row,1,breast_dist_errors_SUB_stdev)
    breast_sheet.write(breast_row,2,breast_total_rewards_SUB_stdev)
    breast_sheet.write(breast_row,3,breast_dices_SUB_stdev)
    breast_sheet.write(breast_row,4,breast_overlaps_SUB_stdev)
    breast_row += 1

    breast_sheet.write(breast_row,0,"min T1")
    breast_sheet.write(breast_row,1,breast_dist_errors_T1_min)
    breast_sheet.write(breast_row,2,breast_total_rewards_T1_min)
    breast_sheet.write(breast_row,3,breast_dices_T1_min)
    breast_sheet.write(breast_row,4,breast_overlaps_T1_min)
    breast_row += 1
    breast_sheet.write(breast_row,0,"max T1")
    breast_sheet.write(breast_row,1,breast_dist_errors_T1_max)
    breast_sheet.write(breast_row,2,breast_total_rewards_T1_max)
    breast_sheet.write(breast_row,3,breast_dices_T1_max)
    breast_sheet.write(breast_row,4,breast_overlaps_T1_max)
    breast_row += 1
    breast_sheet.write(breast_row,0,"median T1")
    breast_sheet.write(breast_row,1,breast_dist_errors_T1_med)
    breast_sheet.write(breast_row,2,breast_total_rewards_T1_med)
    breast_sheet.write(breast_row,3,breast_dices_T1_med)
    breast_sheet.write(breast_row,4,breast_overlaps_T1_med)
    breast_row += 1
    breast_sheet.write(breast_row,0,"mean T1")
    breast_sheet.write(breast_row,1,breast_dist_errors_T1_mean)
    breast_sheet.write(breast_row,2,breast_total_rewards_T1_mean)
    breast_sheet.write(breast_row,3,breast_dices_T1_mean)
    breast_sheet.write(breast_row,4,breast_overlaps_T1_mean)
    breast_row += 1
    breast_sheet.write(breast_row,0,"stdev T1")
    breast_sheet.write(breast_row,1,breast_dist_errors_T1_stdev)
    breast_sheet.write(breast_row,2,breast_total_rewards_T1_stdev)
    breast_sheet.write(breast_row,3,breast_dices_T1_stdev)
    breast_sheet.write(breast_row,4,breast_overlaps_T1_stdev)
    breast_row += 1

    breast_sheet.write(breast_row,0,"min T2")
    breast_sheet.write(breast_row,1,breast_dist_errors_T2_min)
    breast_sheet.write(breast_row,2,breast_total_rewards_T2_min)
    breast_sheet.write(breast_row,3,breast_dices_T2_min)
    breast_sheet.write(breast_row,4,breast_overlaps_T2_min)
    breast_row += 1
    breast_sheet.write(breast_row,0,"max T2")
    breast_sheet.write(breast_row,1,breast_dist_errors_T2_max)
    breast_sheet.write(breast_row,2,breast_total_rewards_T2_max)
    breast_sheet.write(breast_row,3,breast_dices_T2_max)
    breast_sheet.write(breast_row,4,breast_overlaps_T2_max)
    breast_row += 1
    breast_sheet.write(breast_row,0,"median T2")
    breast_sheet.write(breast_row,1,breast_dist_errors_T2_med)
    breast_sheet.write(breast_row,2,breast_total_rewards_T2_med)
    breast_sheet.write(breast_row,3,breast_dices_T2_med)
    breast_sheet.write(breast_row,4,breast_overlaps_T2_med)
    breast_row += 1
    breast_sheet.write(breast_row,0,"mean T2")
    breast_sheet.write(breast_row,1,breast_dist_errors_T2_mean)
    breast_sheet.write(breast_row,2,breast_total_rewards_T2_mean)
    breast_sheet.write(breast_row,3,breast_dices_T2_mean)
    breast_sheet.write(breast_row,4,breast_overlaps_T2_mean)
    breast_row += 1
    breast_sheet.write(breast_row,0,"stdev T2")
    breast_sheet.write(breast_row,1,breast_dist_errors_T2_stdev)
    breast_sheet.write(breast_row,2,breast_total_rewards_T2_stdev)
    breast_sheet.write(breast_row,3,breast_dices_T2_stdev)
    breast_sheet.write(breast_row,4,breast_overlaps_T2_stdev)
    breast_row += 1
    


        #print(filename)
        #sys.exit()
    wb.save('MetricsResults.xls')
    return locs

###############################################################################

def eval_with_funcs(predictors, nr_eval, get_player_fn, files_list=None):
    """
    Args:
        predictors ([PredictorBase])

    Runs episodes in parallel, returning statistics about the model performance.
    """

    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue, distErrorQueue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue
            self.q_dist = distErrorQueue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(task=False,
                                       files_list=files_list)
                while not self.stopped():
                    try:
                        score, filename, ditance_error, q_values, location = play_one_episode(player, self.func)
                        # print("Score, ", score)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)
                    self.queue_put_stoppable(self.q_dist, ditance_error)

    q = queue.Queue()
    q_dist = queue.Queue()

    threads = [Worker(f, q, q_dist) for f in predictors]

    # start all workers
    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()
    dist_stat = StatCounter()

    # show progress bar w/ tqdm
    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        r = q.get()
        stat.feed(r)
        dist = q_dist.get()
        dist_stat.feed(dist)

    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        r = q.get()
        stat.feed(r)

    while q_dist.qsize():
        dist = q_dist.get()
        dist_stat.feed(dist)

    if stat.count > 0:
        return (stat.average, stat.max, dist_stat.average, dist_stat.max)
    return (0, 0, 0, 0)


###############################################################################

def eval_model_multithread(pred, nr_eval, get_player_fn, files_list):
    """
    Args:
        pred (OfflinePredictor): state -> Qvalue

    Evaluate pretrained models, or checkpoints of models during training
    """
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    with pred.sess.as_default():
        mean_score, max_score, mean_dist, max_dist = eval_with_funcs(
            [pred] * NR_PROC, nr_eval, get_player_fn, files_list)
    logger.info("Average Score: {}; Max Score: {}; Average Distance: {}; Max Distance: {}".format(mean_score, max_score, mean_dist, max_dist))

###############################################################################

class Evaluator(Callback):

    def __init__(self, nr_eval, input_names, output_names,
                 get_player_fn, files_list=None):
        self.files_list = files_list
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        """triggered by Trainer"""
        t = time.time()
        mean_score, max_score, mean_dist, max_dist = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, self.files_list)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)

        # log scores
        self.trainer.monitors.put_scalar('mean_score', mean_score)
        self.trainer.monitors.put_scalar('max_score', max_score)
        self.trainer.monitors.put_scalar('mean_distance', mean_dist)
        self.trainer.monitors.put_scalar('max_distance', max_dist)

###############################################################################
