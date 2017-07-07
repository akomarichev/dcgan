# DCGAN on MNIST

Concise implementation of DCGAN model, [introduced by Alec Radford, et al.](https://arxiv.org/abs/1511.06434), on MNIST dataset.

Requirements
------------

The following luarocks package is required:

- mnist

Also *imageio* python package is required to create the gif file:

How to run
----------

- th main.lua
- python make_gif.py

Generated images
-----

![mnist_generated](https://user-images.githubusercontent.com/7283046/27980277-df861a48-634a-11e7-828e-5982b58f9050.gif)

Plots
-----

![loss_fake](https://user-images.githubusercontent.com/7283046/27980297-1adab70c-634b-11e7-8908-f7431f04c06b.png)
![loss_gen](https://user-images.githubusercontent.com/7283046/27980298-1adbd16e-634b-11e7-9c03-a461c7e04770.png)
![loss_real](https://user-images.githubusercontent.com/7283046/27980296-1ad69528-634b-11e7-8f2a-99c7cd16ea1b.png)

