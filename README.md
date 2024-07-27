# On-the-Fly Guidance (OFG)
> For training medical image registration models

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optron-better-medical-image-registration-via/medical-image-registration-on-ixi)](https://paperswithcode.com/sota/medical-image-registration-on-ixi?p=optron-better-medical-image-registration-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optron-better-medical-image-registration-via/medical-image-registration-on-oasis)](https://paperswithcode.com/sota/medical-image-registration-on-oasis?p=optron-better-medical-image-registration-via)

<!-- OFG is a general training framework that provides an alternative to weakly-supervised and unsupervised training for image registration models. By iteratively optimizing the prediction result of the trained registration model on-the-fly, OFG introduces pseudo ground truth to an unsupervised training process. This supervision provides more direct guidance towards model training compared with unsupervised methods. -->

OFG is a training framework that successfully unites learning-based methods with optimization techniques to enhance the training of learning-based registration models. OFG provides guidance with pseudo ground truth to the model by optimizing the model's output on-the-fly, which allows the model to learn from the optimization process and improve its performance.

## Overall Architecture
<img width="1000" alt="ofg_arch" src="https://github.com/user-attachments/assets/941c01da-c483-44c5-96b1-f5d9614f3100">

OFG is a two stage training method, integrating optimization-based methods with registration models. It optimize the model's output in training time, this process generates a pseudo label on-the-fly, which will provide supervision for the model, yielding a model with better registration performance.

## Performance Benchmark
<img width="1000" alt="benchmark" src="https://github.com/user-attachments/assets/7975ee17-57f9-40e8-9c21-d90addd60870">

OFG consistently improves the registration methods it is used on, and achieves state-of-the-art performance. It has better trainability than unsupervised methods while not using any manually added labels.

## Registration on LPBA40
<img width="1000" alt="ofg_lpba" src="https://github.com/user-attachments/assets/9f651a5d-3dfd-44c7-99a8-f304575bca5f">

OFG provides much smoother deformation while also improving DSC of registration, combining into better overall registration performance across a wide range of modalities and datasets.

## Citation
Cite our work when comparing results:
```
@article{ofg2024,
      title={On-the-Fly Guidance Training for Medical Image Registration}, 
      author={Yuelin Xin and Yicheng Chen and Shengxiang Ji and Kun Han and Xiaohui Xie},
      year={2024},
      eprint={2308.15216},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2308.15216}, 
}
```
