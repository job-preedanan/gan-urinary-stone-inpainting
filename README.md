# GAN-Based Urinary Stones Inpainting Augmentation

We provided two main files:
- cascaded_dilatedNet_gam.py : training GAN for inpainting the stone in the missing part
- generate_test.py : testing stone inpainting

## DATASET
<a href="https://ibb.co/rfvRwvj"><img src="https://i.ibb.co/5x2gR20/inpaint-dataset-example.png" alt="inpaint-dataset-example" border="0"></a>

Fig1. Illustration of cropped urinary stone images and theircorresponding images with stone masks Ms in the image’s center (columns 1-3), as well as cropped non-stone images and thecorresponding images with non-stone masks Mns in the image’s center (columns 4-6).

## GAN ARCHITECTURE FOR STONE INPAINTING
<a href="https://ibb.co/8dWdqrz"><img src="https://i.ibb.co/pRtRcfW/GAN-achitecture.png" alt="GAN-achitecture" border="0"></a>
Fig2. Overview of our framework for generative stone inpainting. A cascaded U-Net generator using dilated convolution is trained with
reconstruction loss, content loss from the pre-trained VGG19, global adversarial loss, and local adversarial loss.

The first part of the cascaded U-Net model was trained with only $\mathcal{L}_{L1}$ loss to generate the coarse result, while the second network was optimized by using the combined objective functions of adversarial loss, L1 reconstruction loss, and content loss expressed as:

$$ {\mathcal{L}_{total} = \lambda}_{1}  \mathcal{L}_{adv} + {\lambda}_{2} \mathcal{L}_{L1} + {\lambda}_{3} \mathcal{L}_{content} $$

where $\lambda$1, $\lambda$2, and $\lambda$3 represent the contributions of adversarial loss, L1 loss, and content loss, respectively. 

This code is the part of the following paper:
> W. Preedanan et al., "Improvement of Urinary Stone Segmentation Using GAN-Based Urinary Stones Inpainting Augmentation," in IEEE Access, vol. 10, pp. 115131-115142, 2022

The paper can be accessed at https://ieeexplore.ieee.org/abstract/document/9933448
