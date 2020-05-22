## Multitask radiological modality invariant landmark localization using deep reinforcement learning

This work presents multitask modality invariant deep reinforcement learning framework (MIDRL) for landmark localization across multiple different orgrans and modalities using a single reinforcement learning agent. 

---
## Results
Examples of the single 2D agent locating different landmarks in 2D slices. Red is the target bounding box, yellow is the agent bounding box.
<p>
<img src="DQN/images/breast_example.gif" width="250" height="250">
<img src="DQN/images/ADC_Prostate.gif" width="250" height="250">
<img src="DQN/images/WB_W_Heart.gif" width="250" height="250">
<img src="DQN/images/WB_W_Kidney.gif" width="250" height="250">
<img src="DQN/images/WB_W_Trochanter.gif" width="250" height="250">
<img src="DQN/images/WB_W_Knee.gif" width="250" height="250">
</p>


## Installation
Make a virtualenv and set it up by [following the link](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

After setting up a virtual envt, install the dependencies using: 
```bash
pip install -r requirements.txt
```

### Source code
You can clone the latest version of the source code with the command::
```
https://github.com/bocchs/MIDRL-2D.git
```

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


## References

[1] Amir Alansary, Ozan Oktay, Yuanwei Li, Loic Le Folgoc, Benjamin Hou, Ghislain Vaillant, Konstantinos Kamnitsas, Athanasios Vlontzos,  Ben Glocker, Bernhard Kainz, and Daniel Rueckert. Evaluating Reinforcement Learning Agents for Anatomical Landmark Detection. Medical Image Analysis, 2019.</br>
[2] Amir  Alansary,  Loic  Le  Folgoc,  Ghislain  Vaillant,  Ozan  Oktay,  Yuanwei  Li,  Wenjia  Bai,Jonathan  Passerat-Palmbach,  Ricardo  Guerrero,  Konstantinos  Kamnitsas,  BenjaminHou, et al. Automatic view planning with multi-scale deep reinforcement learning agents.InInternational Conference on Medical Image Computing and Computer-Assisted Inter-vention, pages 277–285. Springer, 2018
