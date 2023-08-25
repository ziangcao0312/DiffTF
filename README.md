# Generalizable 3D Diffusion Model with Transformer


## Abstract
Diffusion-based 3D generative models operating on triplane features have proven the huge potential in both performance and computational efficiency. However, most of them adopt the 2D diffusion, thereby neglecting the 3D information in triplane features. Besides, one model of existing methods works for one category. In this paper, we try to propose a 3D-aware transformer structure to comprehensively introduce 3D awareness (DiffTF). To our best known, it is the first attempt to use one model for multi-categories generation.	Specifically, we build the 3D awareness among different planes via two modules, 3D-aware encoder (decoder) and 3D-aware transformer. The former aims to map the triplane feature to high-level features with 3D awareness in an efficient way while the latter focuses on exploiting the 3D-related information in high-level space. Leveraging the 3D awareness attention, our DiffTF extracts the global interdependencies within individual planes as well as across planes efficiently. It is crucial for generating triplanes with high-quality 3D constraints. To verify the performance in a single category, we compared our method on ShapeNet with other state-of-the-art methods. Furthermore, comprehensive evaluations are conducted on the latest real-scanned dataset, \textit{i.e.}, OmniObject3D, proving the promising performance in complicated multi-categories 3D object conditions (200+ classes with some long-tailed classes).

![Workflow of our tracker](https://github.com/vision4robotics/TCTrack/blob/main/images/workflow.jpg)


The implementation of our online temporally adaptive convolution is based on [TadaConv](https://github.com/alibaba-mmai-research/TAdaConv) (ICLR2022).


## 1. Environment setup
This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## 2. Test

### (a) TCTrack
Download pretrained model by [Baidu](https://pan.baidu.com/s/1jSAcHY9OfarVlxKjOCrVEw) （code: 2u1l) or [Googledrive](https://drive.google.com/file/d/1nWRfvAEcSduR9A4W5MpyZBjp0SCjvmNk/view?usp=sharing) and put it into `tools/snapshot` directory.

Download testing datasets and put them into `test_dataset` directory. 

```bash 
python ./tools/test.py                                
	--dataset OTB100                  
    --tracker_name TCTrack
	--snapshot snapshot/general_model.pth # pre-train model path
```
The testing result will be saved in the `results/dataset_name/tracker_name` directory.

**Note:** The results of TCTrack can be [downloaded](https://pan.baidu.com/s/1-V4JbKvmVPm0aOKWTOQtyQ) (code:kh3e).

### (b) TCTrack++
Download pretrained model by [baidu](https://pan.baidu.com/s/1aggubJ4F-YdMtEo7t0lYtw?pwd=dj2u) (code:dj2u) [Googledrive](https://drive.google.com/file/d/1yHLZTPkU_Mko0OX03fd2HH01g0gflusI/view?usp=sharing) and put it into `tools/snapshot` directory.

Download testing datasets and put them into `test_dataset` directory. 

```bash 
python ./tools/test.py     # offline evaluation                       
	--dataset OTB100                  
    --tracker_name TCTrack++
	--snapshot snapshot/general_model.pth # pre-train model path

```
```bash 
python ./tools/test_rt.py     # online evaluation                       
	--dataset OTB100                  
    --tracker_name TCTrack++
	--snapshot snapshot/general_model.pth # pre-train model path
```

The testing result will be saved in the `results/dataset_name/tracker_name` directory.

**Note:** The results of TCTrack++ can be [downloaded](https://drive.google.com/file/d/1TaolHsyOy_zIkm-MEEkMZuOtbr_NuUYC/view?usp=sharing) or [downloaded](https://pan.baidu.com/s/1v7ie10TmFDiWKoosTESXTw?pwd=3vyx) (code: 3vyx).

## 3. Train

### (a) TCTrack

#### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [Lasot](https://paperswithcode.com/dataset/lasot)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)


**Note:** `train_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.

#### Train a model
To train the TCTrack and TCTrack++ model, run `train.py` with the desired configs:

```bash
cd tools
python train_tctrack.py
```

### (b) TCTrack++

#### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [Lasot](https://paperswithcode.com/dataset/lasot)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)
* [COCO](http://cocodataset.org)

**Note:** `train_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.

### Train a model
To train the TCTrack and TCTrack++ model, run `train.py` with the desired configs:

```bash
cd tools
python train_tctrackpp.py
```

## 4. Offline Evaluation
If you want to evaluate the results of our tracker, please put those results into  `results` directory.
```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset OTB100                  \ # dataset_name
	--tracker_prefix 'general_model'   # tracker_name
```

## 5. Online Evaluation
If you want to evaluate the results of our tracker, please put the pkl files into  `results_rt_raw` directory.


```
#first step

python rt_eva.py 	                          \
	--raw_root ./tools/results_rt_raw/OTB100          \ # pkl path
	--tar_root ./tools/results_rt/OTB100                  \ # output txt files for evaluation
	--gtroot ./test_dataset/OTB100   # groundtruth of dataset
```

```
# second step
python eval.py 	                          \
	--tracker_path ./results_rt          \ # result path
	--dataset OTB100                  \ # dataset_name
	--trackers TCTrack++   # tracker_name
```


**Note:** The code is implemented based on [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit). We would like to express our sincere thanks to the contributors.



## References 

```
@inproceedings{cao2022tctrack,
	title={{TCTrack: Temporal Contexts for Aerial Tracking}},
	author={Cao, Ziang and Huang, Ziyuan and Pan, Liang and Zhang, Shiwei and Liu, Ziwei and Fu, Changhong},
	booktitle={CVPR},
	pages={14798--14808},
	year={2022}
}

@article{cao2023realworld,
      title={Towards Real-World Visual Tracking with Temporal Contexts}, 
      author={Ziang Cao and Ziyuan Huang and Liang Pan and Shiwei Zhang and Ziwei Liu and Changhong Fu},
      journal={arXiv preprint arXiv:2308.10330},
      year={2023},

}

```

## Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot). We would like to express our sincere thanks to the contributors.