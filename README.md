# GAN-Based Urinary Stones Inpainting Augmentation

We provided two main files:
- cascaded_dilatedNet_gam.py : training GAN for inpainting the stone in the missing part
- generate_test.py : testing stone inpainting

## DATASET
<a href="https://ibb.co/rfvRwvj"><img src="https://i.ibb.co/5x2gR20/inpaint-dataset-example.png" alt="inpaint-dataset-example" border="0"></a>

Fig. 1 Illustration of cropped urinary stone images and theircorresponding images with stone masks Ms in the image’s center (columns 1-3), as well as cropped non-stone images and thecorresponding images with non-stone masks Mns in the image’s center (columns 4-6).

## GAN ARCHITECTURE FOR STONE INPAINTING
<a href="https://ibb.co/8dWdqrz"><img src="https://i.ibb.co/pRtRcfW/GAN-achitecture.png" alt="GAN-achitecture" border="0"></a>

Fig. 2 Overview of our framework for generative stone inpainting. A cascaded U-Net generator using dilated convolution is trained with
reconstruction loss, content loss from the pre-trained VGG19, global adversarial loss, and local adversarial loss.

The first part of the cascaded U-Net model was trained with only $\mathcal{L}_{L1}$ loss to generate the coarse result, while the second network was optimized by using the combined objective functions of adversarial loss, L1 reconstruction loss, and content loss expressed as:

$$ {\mathcal{L}_{total} = \lambda}_{1}  \mathcal{L}_{adv} + {\lambda}_{2} \mathcal{L}_{L1} + {\lambda}_{3} \mathcal{L}_{content} $$

where $\lambda$1, $\lambda$2, and $\lambda$3 represent the contributions of adversarial loss, L1 loss, and content loss, respectively. 


## APPLICATION : STON-SYNTHESIS AUGMENTATION IN ABDOMINAL-XRAY IMAGES

<a href="https://ibb.co/64J8Twt"><img src="https://i.ibb.co/zSHFBx2/datagen-pipeline4.png" alt="datagen-pipeline4" border="0"></a>

Fig. 3 Proposed framework for image augmentation including GAN-based augmentation and classic augmentation techniques for urinary stone
segmentation.

## EXAMPLE OF STONE INPAINTING RESULTS
<a href="https://ibb.co/z6GzvXJ"><img src="https://i.ibb.co/xFJNT1H/g-sc-1-g-sc-2-g-sf.png" alt="g-sc-1-g-sc-2-g-sf" border="0"></a>

Fig. 4 Illustrations in columns 1-3 show original cropped Isf images, cropped $I_{sf}$  with random stone masks, and $G(I_{sf})$ results from the stone-free
augmentation. Illustrations of original cropped $I_{sc}$ images, cropped $I_{sc}$ with masks, and $I_{sc}$ results from the stone-synthesized and stone-removed
augmentation are shown in columns 4-6, and 7-9, respectively.

This code is the part of the following paper:
> W. Preedanan et al., "Improvement of Urinary Stone Segmentation Using GAN-Based Urinary Stones Inpainting Augmentation," in IEEE Access, vol. 10, pp. 115131-115142, 2022

The paper can be accessed at https://ieeexplore.ieee.org/abstract/document/9933448
