# A Self-Supervised Gait Encoding Approach with Locality-Awareness for 3D Skeleton Based Person Re-Identification
By Haocong Rao, Siqi Wang, Xiping Hu, Mingkui Tan, Huang Da, Jun Cheng, Bin Hu, and Xinwang, Liu.
## Introduction
This is the official implementation of the self-supervised gait encoding approach presented by "A Self-Supervised Gait Encoding Approach with Locality-Awareness for 3D Skeleton Based Person Re-Identification".
The codes are used to reproduce experimental results of the proposed Contrastive Attention-basd Gait Encodings (CAGEs) in the [paper](https://arxiv.org/abs/2009.03671).

## Requirements
- Python 3.5
- Tensorflow 1.10.0 (GPU)

## Datasets
We provide four already preprocessed datasets (BIWI, IAS, KGBD, KS20) on <br/>
https://pan.baidu.com/s/1FuESlFZkWL6UgARcuMCIVA &nbsp; &nbsp; &nbsp; password：&nbsp; gfij <br/>
All the best models reported in our paper can be acquired on <br/> 
https://pan.baidu.com/s/1sC0mjVTAhA5qq6I73rPA_g &nbsp; &nbsp; &nbsp; password：&nbsp; g3l3  <br/> 
Please download the pre-processed datasets ``Datasets/`` and the model files ``Models/`` into the current directory. <br/><br/>
We also provide the pre-trained gait encoding models on <br/> 
https://pan.baidu.com/s/1aH0dBY5kpTaMVR9XxM89iw &nbsp; &nbsp; &nbsp; password：&nbsp; xkax  <br/> 
Please download the pre-trained gait encoding models into the directory ``Models/``. 
<br/>

The original datasets can be downloaded here: [BIWI and IAS-Lab](http://robotics.dei.unipd.it/reid/index.php/downloads), [KGBD](https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor), [KS20.](http://vislab.isr.ist.utl.pt/datasets/#ks20)
 
## Usage

To (1) train the self-supervised gait encoding model to obtain CAGEs and (2) validate the effectiveness of CAGEs for person Re-ID on a specific dataset with a recognition network, simply run the following command: 

```bash
python train.py --dataset BIWI

# Default options: --attention LA --dataset BIWI --length 6 --t 0.1 --train_flag 1 --model rev_rec --gpu 0
# --attention: [LA, BA]  
# --dataset [BIWI, IAS, KGBD, KS20]  
# --length [4, 6, 8, 10] 
# --t [0.05, 0.1 (for BIWI/IAS/KS20), 0.5 (for KGBD), 0.8, 1.0] 
# --train_flag [1 (for training gait encoding models+RN), 0 (for training RN)] 
# --model [rev_rec, prediction, sorting, rev_rec_plus] Note that "rev_rec_plus" will train three types of models sequentially.
# --gpu [0, 1, ...]

```
Please see ```train.py``` for more details.

To print evaluation results (Re-ID Confusion Matrix / Rank-n Accuracy / Rank-1 Accuracy / nAUC) of the best model, run:

```bash
python evaluate.py --dataset BIWI --best_model rev_rec

# --dataset [BIWI, IAS, KGBD, KS20] 
# --best_model [rev_rec, rev_rec_plus] 
```
To evaluate the already trained model saved in ```Models/CAGEs_RN_models/model_name```, run:

```bash
python evaluate.py --RN_dir model_name

```
 
Please see ```evaluate.py``` for more details.


## License

Locality-Awareness-SGE is released under the MIT License.
