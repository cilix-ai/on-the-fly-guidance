# Optron
> Better image registration via optimizing in the loop.

Optron is a general training archiecture incorporating the idea of optimizing in the loop. By iteratively optimizing the prediction result of a deep learning model through a plug-and-play optimizer module in the training loop, Optron introduces pseudo ground truth to an unsupervised training process. This pseudo supervision provides more direct guidance towards model training compared with unsupervised methods

## Overall Architecture
<img width="1020" alt="optron-simple-trans" src="https://github.com/miraclefactory/optron/assets/89094576/fb6f9dd1-4fe2-42a8-969a-64e04d0ffe75">
Optron is a two stage method, integrating the advantages of optimization-based methods with deep learning models. It optimize the deep learning model's output in training time, which will provide pseudo supervision for the model, yielding a model with better registration performance.

## Performance Benchmark
<img width="564" alt="optron-bench" src="https://github.com/miraclefactory/optron/assets/89094576/21ac1af3-24e6-4763-89a2-86744c021ac5">

Optron consistently improves the deep learning methods it is used on, it achieves state-of-the-art performance on IXI with TransMorph.
