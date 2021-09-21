# CLIPort

[**CLIPort: What and Where Pathways for Robotic Manipulation**](https://openreview.net/pdf?id=9uFiX_HRsIL)  
[Mohit Shridhar](https://mohitshridhar.com/), [Lucas Manuelli](http://lucasmanuelli.com/), [Dieter Fox](https://homes.cs.washington.edu/~fox/)  
[CoRL 2021](https://www.robot-learning.org/) 

CLIPort is an end-to-end imitation-learning agent that can learn a single language-conditioned policy for various tabletop tasks. The framework combines the broad semantic understanding (_what_) of [CLIP](https://openai.com/blog/clip/) with the spatial precision (_where_) of [TransporterNets](https://transporternets.github.io/) to learn generalizable skills from limited training demonstrations.

For the latest updates, see: [cliport.github.io](https://cliport.github.io)

![](media/sim_tasks.gif)

## Guides

- Getting Started: [Installation](#installation), [Quick Tutorial](#quickstart)
- Data Generation: [Dataset](#dataset-generation), [Tasks](cliport/tasks)
- Training & Evaluation: [Single Task](#single-task-training--evaluation), [Multi Task](#multi-task-training--evaluation)
- Miscellaneous: [Notebooks](#notebooks), [Docker Guide](#docker-guide), [Disclaimers](#disclaimers), [Real-Robot Training FAQ](#real-robot-training-faq)
- References: [Citations](#citations), [Acknowledgements](#acknowledgements)

## Installation

Clone Repo:
```bash
$ git clone https://github.com/cliport/cliport.git
```

Setup virtualenv and install requirements:
```bash
$ virtualenv -p $(which python3.8) --system-site-packages cliport_env # or whichever package manager you prefer 
$ source cliport_env/bin/activate

$ pip install --upgrade pip
$ pip install -r requirements.txt

$ cd cliport
$ export CLIPORT_ROOT=$(pwd) 
$ python setup.py develop
```

**Note**: You might need versions of `torch==1.7.1` and `torchvision==0.8.2` that are compatible with your CUDA and hardware. 

## Quickstart

A quick tutorial on evaluating a pre-trained multi-task model.

Download pre-trained checkpoints for multi-task models:
```bash
sh TODO
```

Generate 10 `test` instances for `stack-block-pyramid-seq-seen-colors` and save them in `$CLIPORT_ROOT/data`:
```bash
$ python cliport/demos.py n=10 \  
                          task=stack-block-pyramid-seq-seen-colors \ 
                          mode=test \ 
                          disp=True 
```   
This will take a few minutes to finish. If you are on a headless machine turn off the visualization with `disp=False`. 

Evaluate the best validation checkpoint (trained with 1000 demos) on the test set:
```bash
$ python cliport/eval.py model_task=multi-language-conditioned \
                         eval_task=stack-block-pyramid-seq-seen-colors \ 
                         agent=cliport \ 
                         n_demos=10 \ 
                         train_demos=1000 \ 
                         exp_folder=quickstart_exps \ 
                         mode=test \ 
                         checkpoint_type=test_best \
                         disp=True
```

## Training and Evaluation

All tasks follow a 4-phase workflow:
 
1. Generate `train`, `val`, `test` datasets with `demos.py` 
2. Train agents with `train.py` 
3. Run validation with `eval.py` to find the best checkpoint on `val` tasks
4. Evaluate the best checkpoint on `test` tasks with `eval.py`  

### Dataset Generation

#### Single Task

Generate a`train` set of 1000 demonstrations for `stack-block-pyramid-seq-seen-colors` as save them in `$CLIPORT_ROOT/data`:
```bash
$ python cliport/demos.py n=1000 \ 
                          task=stack-block-pyramid-seq-seen-colors \ 
                          mode=train 
```

You can also do a sequential sweep with `-m` and comma-separated params `task=towers-of-hanoi-seq-seen-colors,stack-block-pyramid-seq`. Use `disp=True` to visualize the data generation.  

**Note:** The full dataset requires [~1.6TB of storage](https://i.kym-cdn.com/photos/images/newsfeed/000/515/629/9bd.gif), which includes both language-conditioned and demo-conditioned (original TransporterNets) tasks. It's recommend that you start with single-task training if you don't have enough storage space.

#### Full Dataset

Run [`generate_dataset.sh`](scripts/generate_datasets.sh) to generate the full dataset and save it to `$CLIPORT_ROOT/data`:

```bash
$ sh scripts/generate_dataset.sh data
```
**Note:** This script is not parallelized and will take a long time (maybe days) to finish. 

### Single-Task Training & Evaluation

#### Training

Train a `cliport` agent with `1000` demonstrations on the `stack-block-pyramid-seq-seen-colors` task for 200K iterations:

```bash
$ python cliport/train.py train.task=stack-block-pyramid-seq-seen-colors \
                          train.agent=cliport \
                          train.attn_stream_fusion_type=add \
                          train.trans_stream_fusion_type=conv \
                          train.lang_fusion_type=mult \
                          train.n_demos=1000 \
                          train.n_step=201000 \
                          dataset.cache=False 
```

#### Validation

Iteratively evaluate all the checkpoints on `val` and save the results in `exps/<task>-train/checkpoints/<task>-val-results.json`: 

```bash
python cliport/test.py eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       exp_folder=exps 
```

#### Test

Choose the best checkpoint from validation to run on the `test` set and save the results in `exps/<task>-train/checkpoints/<task>-test-results.json`:

```bash
python cliport/test.py eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=test_best \
                       exp_folder=exps 
```

### Multi-Task Training & Evaluation

#### Training

Train multi-task models by specifying `task=multi-language-conditioned`, `task=multi-loo-packing-box-pairs-unseen-colors` (`loo` stands for leave-one-out or multi-attr tasks) etc.

```bash
$ python cliport/train.py train.task=multi-language-conditioned \
                          train.agent=cliport \
                          train.attn_stream_fusion_type=add \
                          train.trans_stream_fusion_type=conv \
                          train.lang_fusion_type=mult \
                          train.n_demos=1000 \
                          dataset.cache=False \ 
                          dataset.type=multi 
```

**Important**: You need to generate the full dataset of tasks specified in [`dataset.py`](cliport/dataset.py) before multi-task training or modify the list of tasks [here](cliport/dataset.py#L405). 

#### Validation

Run validation with a trained `multi-language-conditioned` multi-task model on `stack-block-pyramid-seq-seen-colors` tasks

```bash
python cliport/test.py model_task=multi-language-conditioned \ 
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       type=single \
                       exp_folder=exps 
```

#### Test

Evaluate the best checkpoint on the `test` set:

```bash
python cliport/test.py model_task=multi-language-conditioned \ 
                       eval_task=stack-block-pyramid-seq-seen-colors \
                       agent=cliport \
                       n_demos=100 \
                       train_demos=1000 \
                       checkpoint_type=test_best \
                       type=single \
                       exp_folder=exps 
```

## Disclaimers

- **Code Quality Level**: Tired grad student. 
- **Scaling**: The code only works for batch size 1. See this issue for reference. In theory, there is nothing preventing larger batch sizes other than GPU memory constraints.
- **Memory and Storage**: There are lots of places where memory usage can be reduced. You don't need 3 copies of the same CLIP ResNet50 and you don't need to save its weights in checkpoints since it's frozen anyway. Dataset sizes could be dramatically reduced with better storage formats and compression. 
- **Frameworks**: There are lots of leftover NumPy bits from when I was trying to reproduce the TransportNets results. I'll try to clean up when I get some time. 
- **Rotation Augmentation Issue**: All tasks use the same distrubution for sampling SE(2) rotation perturbations. This obviously leads to issues with tasks that involve spatial relationships like 'left' or 'forward'.  

## Notebooks

Checkout [Kevin Zakka's](https://kzakka.com/) Colab for [zero-shot detection with CLIP](https://github.com/kevinzakka/clip_playground). This notebook might be a good way of gauging what sort of visual attributes CLIP can ground with language. But note that CLIPort does **NOT** do "object detection", but instead directly "detects actions". 

#### Others Todos
- [ ] Dataset Visualizer
- [ ] Affordance Heatmap Visualizer
- [ ] Evaluation Results Plot 

## Docker Guide

Install [Docker](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker#ubuntu-160418042004-debian-jessiestretchbuster). 

Modify [docker_build.py](scripts/docker_build.py) and [docker_run.py](scripts/docker_run.py) to your needs.

#### Build 

Build the image:

```bash
$ python scripts/docker_build.py 
```

#### Run

Start container:

```bash
$ python scripts/docker_run.py
 
  source ~/cliport_env/bin/activate
  cd ~/cliport
```

Use `scripts/docker_run.py --headless` if you are on a headless machines like a remote server or cloud instance.

## Real-Robot Training FAQ

#### How much training data do I need?

It depends on the complexity of the task. With 5-10 demonstrations the agent should start to do something useful, but it will often make mistakes by picking the wrong object. For robustness you probably need 50-100 demostrations. A good way to gauge how much data you might need is to setup a simulated version of the problem and evaluate agents trained with 1, 10, 100, and 1000 demonstrations.  

#### Why doesn't the agent follow my language instruction?

This means either there is some sort of bias in the dataset that the agent is exploiting, or you don't have enough training data. Also make sure that the task is doable - if a referred attribute is barely legible in the input, then it's going to be hard for agent to figure out what you mean. 

#### Does CLIPort predict height (z-values) of the end-effector? #### 

CLIPort does not predict height values. You can either: (1) come up with a heuristic based on the heightmap to determine the height position, or (2) train a simple MLP like in [TransportNets-6DOF](https://github.com/google-research/ravens/blob/master/ravens/models/transport_6dof.py) to predict z-values.

#### Shouldn't CLIP help in zero-shot detection of things? Why do I need collect more data?

Note that CLIPort is not doing "object detection". CLIPort fine-tunes CLIP's representations to "detect actions" in SE(2). CLIP by itself has no understanding of actions or affordances; recognizing and localizing objects (e.g. detecting hammer) does not tell you anything about how to manipulate them (e.g. grasping hammer by the handle).    

#### What are the best hyperparams for real-robot training?

The [default settings](cliport/cfg/train.yaml) should work well. Although recently, I have been playing around with using FiLM [(Perez et. al, 2017)](https://distill.pub/2021/multimodal-neurons/) to fuse language features inspired by BC-0 [(Jang et. al, 2021)](https://openreview.net/forum?id=8kbp23tSGYv). Qualitatively, it seems like FiLM is better for reading text etc. but I haven't conducted a full quantitative analysis. Try it out yourself with `train.agent=two_stream_clip_film_lingunet_lat_transporter`.      

#### How to pick the best checkpoint for real-robot tasks?

Ideally, you should create a validation set with heldout instances and then choose the checkpoint with the lowest translation and rotation errors. You can also reuse the training instances but swap the language instructions with unseen goals.

#### Why is the agent confusing directions like 'forward' and 'left'?

By default, training samples are augmented with SE(2) rotations sampled from `N(0, 60 deg)`. For tasks with rotational symmetries (like moving pieces on a chessboard) you need to be careful with this [rotation augmentation parameter](cliport/cfg/train.yaml#L15).


## Acknowledgements

This work use code from the following open-source projects and datasets:

#### Google Ravens (TransporterNets)
Original:  [https://github.com/google-research/ravens](https://github.com/google-research/ravens)  
License: [Apache 2.0](https://github.com/google-research/ravens/blob/master/LICENSE)    
Changes: All PyBullet tasks are directly adapted from the Ravens codebase. 

#### OpenAI CLIP

Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)  
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)  
Changes: Minor modifications to CLIP-ResNet50 to save intermediate features for skip connections.

#### Google Scanned Objects

Original: [Dataset](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects)  
License: [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Changes: Fixed center-of-mass (COM) to be geometric-center for selected objects.

#### U-Net 

Original: [https://github.com/milesial/Pytorch-UNet/](https://github.com/milesial/Pytorch-UNet/)  
License: [GPL 3.0](https://github.com/milesial/Pytorch-UNet/)  
Changes: Used as is in [unet.py](cliport/models/core/unet.py). Note: This part of the code is GPL 3.0.  

## Citations

**CLIPort**
```
@inproceedings{shridhar2021cliport,
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  title     = {CLIPort: What and Where Pathways for Robotic Manipulation},
  booktitle = {Proceedings of the 5th Conference on Robot Learning (CoRL)},
  year      = {2021},
}
```

**CLIP**
```
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
```

**TransporterNets**
```
@inproceedings{zeng2020transporter,
  title={Transporter networks: Rearranging the visual world for robotic manipulation},
  author={Zeng, Andy and Florence, Pete and Tompson, Jonathan and Welker, Stefan and Chien, Jonathan and Attarian, Maria and Armstrong, Travis and Krasin, Ivan and Duong, Dan and Sindhwani, Vikas and others},
  booktitle={Proceedings of the 4th Conference on Robot Learning (CoRL)},
  year= {2020},
}
```

## Questions or Issues?

Please file an issue with the issue tracker.  