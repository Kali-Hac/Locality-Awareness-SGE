# A Self-Supervised Gait Encoding Approach with Locality-Awareness for 3D Skeleton Based Person Re-Identification
By Haocong Rao, Siqi Wang, Xiping Hu, Mingkui Tan, Huang Da, Jun Cheng, Bin Hu, and Xinwang, Liu.
## Introduction
This is the official implementation of the self-supervised gait encoding approach presented by "A Self-Supervised Gait Encoding Approach with Locality-Awareness for 3D Skeleton Based Person Re-Identification".
The codes are used to reproduce experimental results of the proposed Contrastive Attention-basd Gait Encodings (CAGEs) in the [paper](./).

## Requirements
- Python 3.5
- Tensorflow 1.10.0 (GPU)

## Datasets
We provide three already preprocessed datasets (BIWI, IAS, KGBD) on <br/>
https://share.weiyun.com/5faKfq4 &nbsp; &nbsp; &nbsp; password：&nbsp; &nbsp; ma385h <br/>
Two already trained models (BIWI, IAS) are saved in this repository, and all three models can be acquired on <br/>
https://share.weiyun.com/5EBPkPZ &nbsp; &nbsp; &nbsp; password：&nbsp; &nbsp; 6xpj8r  <br/> 
Please download the preprocessed datasets ``Datasets/`` and the model files ``Models/`` into the current directory. 
<br/>

The original datasets can be downloaded from: http://robotics.dei.unipd.it/reid/index.php/downloads (BIWI and IAS-Lab) <br/>
https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor (KGBD) 
 
## Usage

To (1) train the self-supervised gait encoding model to obtain CAGEs and (2) validate the effectiveness of CAGEs for person Re-ID on a specific dataset with a recognition network,  simply run the following command: 

```bash
# --attention: [LA, BA]  --dataset [BIWI, IAS, KGBD, KS20]  
# --length [4, 6, 8, 10] --t [0.05, 0.1 (for BIWI/IAS/KS20), 0.5 (for KGBD), 0.8, 1.0] 
# --train_flag [1 (for training gait encoding models+RN), 0 (for training RN)] 
# --model [rev_rec, prediction, sorting, rev_rec_plus] --gpu 0

python train.py --dataset BIWI
```
Please see ```train.py``` for more details.

To print evaluation results (Re-ID Confusion Matrix / Rank-n Accuracy / Rank-1 Accuracy / nAUC) of the best model, run:

```bash
# --dataset [BIWI, IAS, KGBD, KS20] --best_model [rev_rec, prediction, sorting, rev_rec_plus] 
python evaluate.py --dataset BIWI --best_model rev_rec

```
To print evaluation results (Re-ID Confusion Matrix / Rank-n Accuracy / Rank-1 Accuracy / nAUC) of the model saved in ```Models/AGEs_RN_models/model_name```, run:

```bash

python evaluate.py --dataset BIWI --RN_dir model_name

```

Please see ```evaluate.py``` for more details.


## License

SGE-LA-LCL is released under the MIT License.
