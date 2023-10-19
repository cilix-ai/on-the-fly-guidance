# Optron
> Better image registration via optimizing in the loop.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optron-better-medical-image-registration-via/medical-image-registration-on-ixi)](https://paperswithcode.com/sota/medical-image-registration-on-ixi?p=optron-better-medical-image-registration-via)

Optron is a general training archiecture incorporating the idea of optimizing in the loop. By iteratively optimizing the prediction result of a deep learning model through a plug-and-play optimizer module in the training loop, Optron introduces pseudo ground truth to an unsupervised training process. This pseudo supervision provides more direct guidance towards model training compared with unsupervised methods

## Overall Architecture
<img width="1020" alt="optron-simple-trans" src="https://github.com/miraclefactory/optron/assets/89094576/fb6f9dd1-4fe2-42a8-969a-64e04d0ffe75">
Optron is a two stage method, integrating the advantages of optimization-based methods with deep learning models. It optimize the deep learning model's output in training time, which will provide pseudo supervision for the model, yielding a model with better registration performance.

## Performance Benchmark
<img width="564" alt="optron-bench" src="https://github.com/miraclefactory/optron/assets/89094576/21ac1af3-24e6-4763-89a2-86744c021ac5">

Optron consistently improves the deep learning methods it is used on, it achieves state-of-the-art performance on IXI with TransMorph.

## Citation
Cite our work when comparing results:
```
@misc{optron2023,
    title={Optron: Better Medical Image Registration via Optimizing in the Loop}, 
    author={Yicheng Chen and Shengxiang Ji and Yuelin Xin and Kun Han and Xiaohui Xie},
    year={2023},
    eprint={2308.15216},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
