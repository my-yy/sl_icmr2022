Code for *Self-Lifting: A Novel Framework For Unsupervised Voice-Face Association Learning,ICMR,2022*



## Requirements

```
faiss==1.7.1
pytorch==1.8.1
pytorch-metric-learning==0.9.96
wandb==0.12.10
```



## Dataset

1. Download all files from [Baidu Disk](https://pan.baidu.com/s/1024BMz5fslbQXJ-7iFKXMA) (code:`3d38`) .
2. Unzip `dataset.zip` to the project root.
3. Put `face_input.pkl` and `voice_input.pk` into `dataset/voxceleb/` folder. The  final structure of `dataset` folder is shown below:

```
dataset/
└── voxceleb
    ├── cluster
    │   ├── movie2jpg_path.pkl
    │   ├── movie2wav_path.pkl
    │   └── train_movie_list.pkl
    ├── eval
    │   ├── test_matching_10.pkl
    │   ├── test_matching_g.pkl
    │   ├── test_matching.pkl
    │   ├── test_retrieval.pkl
    │   ├── test_verification.pkl
    │   ├── test_verification_g.pkl
    │   └── valid_verification.pkl
    ├── face_input.pkl
    └── voice_input.pkl
```



# Train

**1. Train Self-Lifting Framework:**

``python sl.py``



**2. Train a baseline:**

``python baseline/1_ccae.py``

``python baseline/2_deepcluster.py``

``python baseline/3_barlow.py``



---

*use [wandb](https://wandb.ai) to view the training process:*

1. Create  `wb_config.json`  file in the  `./configs` folder, using the following content:

   ```
   {
     "WB_KEY": "Your wandb auth key"
   }
   ```

   

2. add `--dryrun=False` to the training command, for example:   `python sl.py --dryrun=False`



## Model Checkpoints 

You can get the final model checkoints at [here](https://pan.baidu.com/s/1Ol0FtaXUm8BticDDNLJaxg) (code:`4ae6`).



