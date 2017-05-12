# Conditional Image Generation
We propose multiple models for conditional image generation on the COCO 2014 
dataset http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/.


### Setup

- Download the [COCO 2014 dataset](http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/) 
- Fork the [Skip-Thought Vector repo](https://github.com/ryankiros/skip-thoughts) and download the necessary dependencies
- Change the path to the dataset in [data_utils.py](https://github.com/charlesashby/conditional-image-generation/blob/master/lib/data_utils.py)
- Run create_embedding_dict function to create a pickled version of the embedded captions
- Run the model you want by calling main.py from the terminal


```
# e.g. to train the BEGAN model...

python main.py BEGAN
```


### Some results using the Boundary Equilibrium GAN (BEGAN) model...

![](began_examples.png)

You can read the full blog post [here](https://charlesashby.github.io/gan/).
