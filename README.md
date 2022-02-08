Repository contains 1st place solution for [Clog Loss: Advance Alzheimer’s Research with Stall Catchers](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/) competition organized by DrivenData. 

## Citation

For more details, please refer to the publication: https://doi.org/10.1016/j.compbiomed.2021.105089

If you find this code useful, please cite it as:
```
@article{solovyev20223d,
  title={3D convolutional neural networks for stalled brain capillary detection},
  author={Solovyev, Roman and Kalinin, Alexandr A and Gabruseva, Tatiana},
  journal={Computers in Biology and Medicine},
  volume={141},
  pages={105089},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2021.105089}
}
```

## Software Requirements

* Main requirements: Python 3.5+, keras 2.2+, Tensorflow 1.13+
* Other requirements: numpy, pandas, opencv-python, scipy, sklearn

You need to have CUDA 10.0 installed
Solution was tested on Anaconda3-2019.10-Linux-x86_64.sh: https://www.anaconda.com/distribution/

## Hardware requirements

* All batch sizes for Neural nets are tuned to be used on NVIDIA GTX 1080 Ti 11 GB card. To use code with other GPUs with less memory - decrease batch size accordingly.
* For fast validation 3D volumes during training are read in memory. So training will require ~64GB of RAM.

## How to run

Code expects all input files in "../input/" directory. Fix paths in a00_common_functions.py if needed.
All r*.py files must be run one by one. All intermediate folders will be created automatically.

### Only inference part

To run inference you need the following: 
* Download [Test Set from DrivenData website](https://community.drivendata.org/t/solutions-postings/4852) and put in in `../input/` folder.
* Download [trained weights (~1 GB)](https://github.com/ZFTurbo/DrivenData-Alzheimer-Research-1st-place-solution/releases/download/v1.0.0/net_v20_d121_only_tier1_finetune_r31_train_3D_model_dn121.py_kfold_split_large_v2_5_42.zip) and unzip them in `../models/` folder 

After that you can run following code:
```
python preproc_data/r01_extract_roi_parts.py test
python net_v20_d121_only_tier1_finetune/r42_process_test.py
```

There is also file run_inference.sh - which do all the stuff including pip installation of required modules etc.

### Full pipeline including training of models

To run training you need to download [all data from DrivenData website](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/data/)  and put in in `../input/` folder. 

```
python3 preproc_data/r01_extract_roi_parts.py
# Uncomment if you need to create new KFold split
# python3 preproc_data/r03_gen_kfold_split.py
python3 net_v13_3D_roi_regions_densenet121/r31_train_3D_model_dn121.py
python3 net_v14_d121_auc_large_valid/r31_train_3D_model_dn121.py
python3 net_v20_d121_only_tier1_finetune/r31_train_3D_model_dn121.py
python3 net_v20_d121_only_tier1_finetune/r42_process_test.py
```

There is file run_train.sh - which do all the stuff including pip installation of required modules etc.

You need to change `run_inference.sh` and `run_train.sh` for your environment:

Change this variable to location of your python (Anaconda)
* `export PATH="/var/anaconda3-temp/bin/"`

Change this variable to location of your code
* `export PYTHONPATH="$PYTHONPATH:/var/test_alzheimer/"`

After you run inference or train final submission file will be located in `../subm/submission.csv` file.


## Related repositories

Two useful parts of code, created for this project, were released as separate modules:
* [Classification models 3D](https://github.com/ZFTurbo/classification_models_3D)
* [Volumentations](https://github.com/ZFTurbo/volumentations)

## Visualization

[![Alzheimer’s Research competition (what neural net sees) (Demo)](https://raw.githubusercontent.com/ZFTurbo/DrivenData-Alzheimer-Research-1st-place-solution/master/images/Youtube.jpg)](https://www.youtube.com/watch?v=k7s5DCzvKj8)

## Solution description

* [Small description on DrivenData](https://www.drivendata.co/blog/clog-loss-alzheimers-winners/)
* [Discussion on forum](https://community.drivendata.org/t/solutions-postings/4852)
* [Arxiv paper](https://arxiv.org/abs/2104.01687)
