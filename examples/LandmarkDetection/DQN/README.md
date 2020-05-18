# Anatomical Landmark Detection

A DQN is trained for locating landmarks in 2D slices of different modalities and imaging parameters.


---
## Results
Examples of the single 2D agent locating different landmarks in 2D slices. Red is the target bounding box, yellow is the agent bounding box.

* Nipple
<p align="center">
<img src="./images/ald_tp1_Pre.gif" width="400">
</p>

* Prostate
<p align="center">
<img src="./images/ADC_Resliced_0019.gif" width="400">
</p>

* Heart
<p align="center">
<img src="./images/normal6_W_159.gif" width="400">
</p>

* Kidney
<p align="center">
<img src="./images/normal6_W_141.gif" width="400">
</p>

* Trochanter
<p align="center">
<img src="./images/normal6_W_97.gif" width="400">
</p>

* Knee
<p align="center">
<img src="./images/normal6_W_27.gif" width="400">
</p>


---


## 
```
usage: DQN.py [-h] [--gpu GPU] [--load LOAD] [--task {play,eval,train}]
              [--algo {DQN,Double,Dueling,DuelingDouble}]
              [--files FILES [FILES ...]] [--saveGif] [--saveVideo]
              [--logDir LOGDIR] [--name NAME]

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             comma separated list of GPU(s) to use.
  --load LOAD           load model
  --task {play,eval,train}
                        task to perform. Must load a pretrained model if task
                        is "play" or "eval"
  --algo {DQN,Double,Dueling,DuelingDouble}
                        algorithm
  --files FILES [FILES ...]
                        Filepath to the text file that comtains list of
                        images. Each line of this file is a full path to an
                        image scan. For (task == train or eval) there should
                        be two input files ['images', 'landmarks']
  --saveGif             save gif image of the game
  --saveVideo           save video of the game
  --logDir LOGDIR       store logs in this directory during training
  --name NAME           name of current experiment for logs

```

### Train
```
 python DQN.py --task train --algo DQN --gpu 0 --files './data/filenames/image_files.txt' './data/filenames/landmark_files.txt'
```

### Evaluate
```
python DQN.py --task eval --algo DQN --gpu 0 --load data/models/DQN_multiscale_brain_mri_point_pc_ROI_45_45_45/model-600000 --files './data/filenames/image_files.txt' './data/filenames/landmark_files.txt'
```

### Test
```
python DQN.py --task play --algo DQN --gpu 0 --load data/models/DQN_multiscale_brain_mri_point_pc_ROI_45_45_45/model-600000 --files './data/filenames/image_files.txt'
```

## Citation

If you use this code in your research, please cite this paper:

```
@article{alansary2019evaluating,
  title={{Evaluating Reinforcement Learning Agents for Anatomical Landmark Detection}},
  author={Alansary, Amir and Oktay, Ozan and Li, Yuanwei and Le Folgoc, Loic and 
          Hou, Benjamin and Vaillant, Ghislain and Kamnitsas, Konstantinos and 
          Vlontzos, Athanasios and Glocker, Ben and Kainz, Bernhard and Rueckert, Daniel},
  journal={Medical Image Analysis},
  year={2019},
  publisher={Elsevier}
}
```
