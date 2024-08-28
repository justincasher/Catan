# Catan

This repository contains the code behind my [Catan project](https://justinasher.me/catan_rl).

The folders are as follows:

- ```fixed_training``` contains the training procedure for a fixed model.
- ```general_training``` contains the training procedure for arbitrary board positions.
- ```fixed_human_vs_bot``` allows you to play against the model as a text game (y.
- ```general_human_vs_bot``` allows you to play against the model as a text game.

To play against the bot, you will net to set up a physical board. (I did not want to create my own version of the game and infringe upon Kosmos' copyright.)

*Note:* ```general_training``` is nearly identical to ```fixed_training```. Obviously, the board is not shuffled in ```fixed_training```. Furthermore, what is called a ```BasicBlock``` in ```fixed_training``` is called a ```ResidueBlock``` in ```general_training```, by mistake. This difference is important when importing models.

Here are links to download trained models:
- [Fixed board model](https://drive.google.com/uc?export=download&id=1xNLq4p44D234_PxZcSEPZT2agR9btUou)
- [General model](https://drive.google.com/uc?export=download&id=1b47cCXPFYiMhxH-_S5Nqxfy4LX3VMXG-)
