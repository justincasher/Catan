######################
## # # # # # # # # # #
# # # CATAN BOT # # ##
## # # # # # # # # # #
######################

import torch
from network import CatanNetwork
from game import CatanGame
    
### Run simulation ###
net = torch.load("fixed_net")
game = CatanGame(3, net, 0)
game.runGame()