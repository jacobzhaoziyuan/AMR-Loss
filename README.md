



<div align="center">

# Adaptive Mean-residue Loss for Robust Facial Age Estimation

[![ICME2021](https://img.shields.io/badge/arXiv-2203.12454-blue)](https://arxiv.org/abs/2203.17156)
[![ICME2021](https://img.shields.io/badge/Conference-ICME2022-green)](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_28)



</div>

Pytorch implementation of our method for ICME2022 paper: "Adaptive Mean-residue Loss for Robust Facial Age Estimation". 


## Abstract
Automated facial age estimation has diverse real-world applications in multimedia analysis, e.g., video surveillance, and human-computer interaction. However, due to the randomness and ambiguity of the aging process, age assessment is challenging.  Most research work over the topic regards the task as one of age regression, classification, and ranking problems, and cannot well leverage age distribution in representing labels with age ambiguity. In this work, we propose a simple yet effective loss function for robust facial age estimation via distribution learning, i.e., adaptive mean-residue loss, in which, the mean loss penalizes the difference between the estimated age distribution's mean and the ground-truth age, whereas the residue loss penalizes the entropy of age probability out of dynamic top-K in the distribution. Experimental results in the datasets FG-NET and CLAP2016 have validated the effectiveness of the proposed loss.


<p align="center">
<img src="https://github.com/jacobzhaoziyuan/AMR-Loss/blob/main/assets/archi.png" width="550">
</p>


## Dataset

For [FGNET](https://yanweifu.github.io/FG_NET_data/), check the [README](https://github.com/jacobzhaoziyuan/AMRLoss/blob/main/FGNET/README.md), under `FGNET` folder.

For [CLAP16](https://chalearnlap.cvc.uab.cat/dataset/26/description/), check the [README](https://github.com/jacobzhaoziyuan/AMRLoss/blob/main/CLAP/README.md), under `CLAP` folder.





## Installation

- Install `python` and `pytorch`
```
  Install PyTorch 1.7.1 + CUDA 10.1 
  Clone this repo.
```

- Clone the repository

```
  git clone https://github.com/jacobzhaoziyuan/AMRLoss
  cd AMRLoss
```

- Install dependencies
```
  pip install -r requirements.txt
```


## Running
```
  cd scripts
```
- For FGNET dataset
```
  bash FGNET_experiments.sh 
```
- For CLAP2016 dataset
```
  bash CLAP_experiments.sh 
```

    

    






## Citation
If you find the codebase useful for your research, please cite the paper:
```
@inproceedings{zhao2022adaptive,
  title={Adaptive Mean-Residue Loss for Robust Facial Age Estimation},
  author={Zhao, Ziyuan and Qian, Peisheng and Hou, Yubo and Zeng, Zeng},
  booktitle={2022 IEEE International Conference on Multimedia and Expo (ICME)},
  year={2022}
}

@misc{zhao2022adaptive,
    title={Adaptive Mean-Residue Loss for Robust Facial Age Estimation},
    author={Ziyuan Zhao and Peisheng Qian and Yubo Hou and Zeng Zeng},
    year={2022},
    eprint={2203.17156},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


## Acknowledgement

Part of the code is adapted from open-source codebase and original implementations of algorithms, 
we thank these authors for their fantastic and efficient codebase:
* https://github.com/Herosan163/AgeEstimation
* https://github.com/yu4u/age-gender-estimation
