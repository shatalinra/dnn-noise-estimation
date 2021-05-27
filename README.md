# deep-noise-estimation

Few deep convolutional neural networks for noise estimation based only on the noisy image without any reference.

## Networks

**Chuah et al**: network from paper by Chuah et al [1]. Several training parameters were changed in order to fit more varied dataset and achieve higher performance than reported in the paper.

**Simple**: my proposed network with far less amount of parameters than Chuah et al network.

**Efficient**: my proposed network based on EfficientNet B0 network [2]. The backbone weights were taken from ImageNet. While training model, the backbone was frozen and ran in inference mode with both 'trainable' and 'training' flags set to False.

## Results

Experimental procedure is loosely based on Chuah et al [1] and consists from several steps:
1. Several images from MS COCO [3] training dataset were taken to generate training and training dataset. Each image had noise applied at ten different levels: 0 means an image without any additional noise while 9 means an image is close to noise itself. Each dataset consists from patches extracted from generated images and labels corresponding to level of noise applied. This turns noise estimation task into image classification with 10 output classes representing different amount of noise.
2. Networks were trained on training dataset while their architecture and learning parameters were tuned based on testing accuracy. 
3. After fiding optimal parameters, each network was validated by estimating noise level on yet another noisy MS COCO training image.

Results for networks are presented in this table. Training loss is value of categorical crossentropy reported in the end of network training. Testing accuracy is categorical accuracy over testing dataset. Validation accuracy is confidence in correct noise level of validation image.

| Network       | Input shape |Total number of parameters | Training loss | Testing accuracy | Validation accuracy | 
| ------------- | ----------- | ------------------------- | ------------- | ---------------- | ------------------- |
| Chuah et al   | 32x32x3     | 447 080                   | 0.002288      | 97.7%            | 93.0%               |
| Simple        | 32x32x3     | 1 787                     | 0.034739      | 98.3%            | 98.5%               | 
| Efficient     | 224x224x3   | 4 062 381                 | 0.112720      | 43.1%            | 50.0%               |

Several notes on the results:
1. While for Chuah et al and simple network patches of 32x32x3 shape were sufficient, the efficent net required input of at least 224x224x3 shape. Although their are CIFAR versions of the Efficient net, using them does not really make sense as 32x32 patch from large image is nothing like image compressed to 32x32 resolution. 
2. Task of classifying image into ten classes of noises is so simple that even custom network with few thousands parameters achieve top results. In theory, similar results could be achieved with efficient network but it would require a lot of data and regularization to prevent overfitting. I have tried increasing size of datasets, using dropout, layer regularizers and several other things with it but testing and validation accuracy have not increased past 50%.
3. In the end, it is more practical to just use simple network instead of trying to heavily regularize more complex one.

## Practical notes

If you are interested in replicating results or it did not work for you, there are few important details:
1. I used Tensorflow 2.5, CUDA 11.3 with cuDNN and Python 3.7.8.
2. The code is heavily geared towards my own machine with 16 GB of RAM and NVIDIA GeForce RTX 2070 SUPER 8 GB. For example, I managed to fit all training data for smaller networks on my GPU in order to have faster training times while I was forced to using tf.data.Dataset for efficient net.
3. Code relies on having folder coco/2017/train with MS COCO training images on the same level as cloned repository folder. For now, path is hardcoded in dnn_noise_estimation.py.

## References
1. J. H. Chuah, H. Y. Khaw, F. C. Soon and C. Chow, "Detection of Gaussian noise and its level using deep convolutional neural network", TENCON 2017 - 2017 IEEE Region 10 Conference, 2017, pp. 2447-2450, doi: 10.1109/TENCON.2017.8228272.
2. Mingxing Tan, Quoc V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", International Conference on Machine Learning, 2019.
3. Lin TY. et al. (2014) Microsoft COCO: Common Objects in Context. In: Fleet D., Pajdla T., Schiele B., Tuytelaars T. (eds) Computer Vision – ECCV 2014. ECCV 2014. Lecture Notes in Computer Science, vol 8693. Springer, Cham. https://doi.org/10.1007/978-3-319-10602-1_48

