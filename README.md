# Conditional Image Generation
We propose multiple models for conditional image generation on the COCO 2014 
dataset http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/.


### Setup

- Download the [COCO 2014 dataset](http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/) 
- Change the path to the dataset in data_utils.py
- Run create_embedding_dict function to create a pickled version of the embedded captions
- Run the model you want by calling main.py in the terminal


### Some results using the Boundary Equilibrium GAN (BEGAN) model...

![](began_examples.png)

