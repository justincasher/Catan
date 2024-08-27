# Catan

This repository contains the code behind my [Catan project](https://justinasher.me/catan_rl). It is currently under construction...come back soon!

The folders are as follows:

- ```fixed_training``` contains the training procedure for a fixed model
- ```general_training``` contains the training procedure for arbitrary board positions.

*Note:* ```general_training``` is nearly identical to ```fixed_training```. Obviously, the board is not shuffled in ```fixed_training```. Furthermore, what is called a ```BasicBlock``` in ```fixed_training``` is called a ```ResidueBlock``` in ```general_training```. This difference is important when importing models.

Here are links to download trained models:
- [General model](https://drive.google.com/uc?export=download&id=1b47cCXPFYiMhxH-_S5Nqxfy4LX3VMXG-)
