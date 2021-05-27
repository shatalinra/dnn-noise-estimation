# deep-noise-estimation

Few deep convolutional neural networks for estimating levels of noise based only on the noisy image without any reference.

## Networks

**Chuah et al**: network from paper by Chuah et al [1]. Several training parameters were changed in order to fit more varied dataset and achieve higher performance than reported in the paper.

**Simple**: my proposed network with far less amount of parameters than Chuah et al network.

**Efficient net**: my proposed network based on Efficient B0 network. The backbone weights were taken from ImageNet. While training model, the backbone was frozen and ran in inference mode with both 'trainable' and 'training' flags set to False.

## Results

Experimental procedure is loosely based on Chuah et al [1]. First, several images from MS COCO training dataset were taken to generate training and training dataset. Each image had noise applied at ten different levels: 0 means an image without any added noise while 9 means an image is close to noise itself. Each dataset consists from patches extracted from generated images and labels corresponding to level of noise applied. This turns noise estimation task into image classification with 10 output classes representing different amount of noise. Second, networks were trained on training dataset while their architecture and learning parameters were tuned based on testing accuracy. Third, after fiding optimal parameters, each network was validated by estimating noise level on yet another noisy MS COCO training image.

Results for networks are presented in this table. Training loss is value of sparse categorical crossentropy reported in the end of network training. Testing accuracy is categorical accuracy over testing dataset. Validation accuracy is confidence in correct noise level of validation image.

| Network       | Input shape |Total number of parameters | Training loss | Testing accuracy | Validation accuracy | 
| ------------- | ----------- | ------------------------- | ------------- | ---------------- | ------------------- |
| Chuah et al   | 32x32x3     | 447 080                   | 0.002288      | 97.7%            | 93.0%               |
| Simple        | 32x32x3     | 1 787                     | 0.034739      | 98.3%            | 98.5%               | 
| Efficient     | 224x224x3   | 4 062 381                 | 0.112720      | 43.1%            | 50.0%               |

First, while for Chuah et al and simple network patches of 32x32x3 shape were sufficient, the efficent net required input of at least 224x224x3 shape. Although their are CIFAR versions of the Efficient net, using them does not really make sense as 32x32 patch from large image is nothing like image compressed to 32x32 resolution. Second, the task is so simple that even custom network with few thousands parameters achieve top results. In order to achieve similar results with efficient net a lot of data and regularization is needed while simple networks work just fine. I have tried enlarging datasets, using dropout, layer regularizers and several other things with efficient net but testing and validation accuracy have not increased past 50%. It makes more practical sense to use simple networks instead of trying heavily regulaze more complex one.


## References
1. J. H. Chuah, H. Y. Khaw, F. C. Soon and C. Chow, "Detection of Gaussian noise and its level using deep convolutional neural network," TENCON 2017 - 2017 IEEE Region 10 Conference, 2017, pp. 2447-2450, doi: 10.1109/TENCON.2017.8228272.

