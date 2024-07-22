# On-the-Fly Guidance (OFG)
> For training medical image registration models

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optron-better-medical-image-registration-via/medical-image-registration-on-ixi)](https://paperswithcode.com/sota/medical-image-registration-on-ixi?p=optron-better-medical-image-registration-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optron-better-medical-image-registration-via/medical-image-registration-on-oasis)](https://paperswithcode.com/sota/medical-image-registration-on-oasis?p=optron-better-medical-image-registration-via)

<!-- OFG is a general training framework that provides an alternative to weakly-supervised and unsupervised training for image registration models. By iteratively optimizing the prediction result of the trained registration model on-the-fly, OFG introduces pseudo ground truth to an unsupervised training process. This supervision provides more direct guidance towards model training compared with unsupervised methods. -->

OFG is a training framework that successfully unites learning-based methods with optimization techniques to enhance the training of learning-based registration models. OFG provides guidance with pseudo ground truth to the model by optimizing the model's output on-the-fly, which allows the model to learn from the optimization process and improve its performance.

## Overall Architecture
<img width="932" alt="ofg" src="https://github.com/miraclefactory/on-the-fly-guidance/assets/89094576/409cde65-02bc-4332-b5eb-24ac5099ce4f">

OFG is a two stage training method, integrating optimization-based methods with registration models. It optimize the model's output in training time, this process generates a pseudo label on-the-fly, which will provide supervision for the model, yielding a model with better registration performance.

## Performance Benchmark
<img width="792" alt="hero" src="https://github.com/miraclefactory/on-the-fly-guidance/assets/89094576/9fe81ae2-a03d-4036-bbb7-af0b32595dab">

OFG consistently improves the registration methods it is used on, and achieves state-of-the-art performance. It has better trainability than unsupervised methods while not using any manually added labels.

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
