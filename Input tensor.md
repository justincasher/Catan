# Output tensor

*Note*: When discussing 1-hot vectors, we start at 0, like Python lists, i.e., the coordinates are [0, 1,..., d]. In general, positions are indexed by going left to right like reading. 

## Overview

The input tensor for our neural network is structured as a 395 x 12 tensor as follows:

<center> 

|   Row     |       Value       |
| :-------: | :---------------: |
|     0     | Player number     |
|    1–29   | Tile resources    |
|   30–48   | Tile numbers      |
|   49–57   | Ship resources    |
|   58–111  | Settlement positions |
|  112–182  | Road positions    |
|    183    | Development cards deck |
|    184    | Development cards hands |
|    185    | Players scores    |
|    186    | Has largest army  |
|    187    | Army sizes        |
|    188    | Has longest raod  |
|    189    | Road lengths      |
|  190–194  | Trade ratios      |
|  195–249  | Resources per roll | 
|    250    | Robber per roll   |
|    251    | Number of resources |
|    252    | Total resources   |
|  253–255  | Buildings         |
|    256    | Roll dice         |
|    257    | Skip turn         |
|  258–312  | Build settlement/city |
|  313–383  | Build road        |
|    384    | Draw card         |
|    385    | Play development card |
|    386    | Monopoly rob      |
|    387    | Trade             |
|    388    | Discard           |
|    389    | Move robber       |
|    390    | Steal card        |

</center>

## Game details

### Player number (0)

This is a 1-hot vector with a 1 in the $ n $-th coordinate, where $ n $ is the player's number.

### Tile resources (1–29)

This is a collection of 19 1-hot vectors with a 1 in the coordinate ascribed by the following list:

1. Wheat
2. Wood
3. Sheep
4. Brick
5. Stone
6. Desert

### Tile numbers (30–48)

This is collection of 19 vectors with the tile number in the first coordinate. Additionally, we put a 1 in the 2nd coordinate if the robber is there.


### Ship resources (49–57)

This is a collection of 9 1-hot vectors with a 1 in the coordinate ascribed by the following list:

1. Wheat
2. Wood
3. Sheep
4. Brick
5. Stone
6. Anything

### Settlement positions (58–111)

This is a collection of 54 1-hot vectors with a 1 in the coordinate ascribed by the following list:

1. No one can settle here
2. Player 1 has settled here 
3. Player 2 has settled here
4. Player 3 has settled here
5. etc.

Furthermore, we change the 1 to a 2 if they have a city there.

### Road positions (112–182)

This is a collection of 71 1-hot vectors with a 1 in the coordiante ascribed by the following list:

1. No on has built a road here 
2. Player 1 has built a road here
3. Player 2 has built a road here
4. Player 3 has built a road here
5. etc.

### Development cards deck (183)

This is a vector containing the following information about development cards in each coordinate:

1. Number of development cards remaining in deck
2. Number of knight cards played
3. Number of victory point cards played
4. Number of road building cards played
5. Number of year of plenty cards played
6. Number of monopoly cards played

### Development cards hands (184)

This is a vector containing the following information in each coordiante:

1. Player 1 number of development cards
2. Player 2 number of development cards
3. Player 3 number of development cards
4. etc.

### Players scores (185)

This is a vector containing the following information in each coordiante:

1. Player 1 points
2. Player 2 points
3. Player 3 points
4. etc.

### Has largest army (186)

This is a 1-hot vector whose $ n $-th coordinate is 1 if player $ n $ has the largest army.

### Army sizes (187)

This is a vector containing the following information in each coordiante:

1. Player 1 army size 
2. Player 2 army size 
3. Player 3 army size
4. etc.

### Has longest road (188)

This is a 1-hot vector whose $ n $-th coordinate is the player number of whoever has the longest road.

### Road lengths (189)

This is a vector containing the following information in each coordiante:

1. Player 1 road length 
2. Player 2 road length 
3. Player 3 road length
4. etc.

### Trade ratios (190–194)

This is a collection of vectors describing the following resources:

1. Wheat
2. Wood
3. Sheep
4. Brick
5. Stone

Each vector is formatting as follows:

1. Player 1 [resource] trade ratio
2. Player 2 [resource] trade ratio
3. Player 3 [resource] trade ratio
4. etc.

### Resources per roll (195–249)

This is a collection of 55 vectors describing how many of each resource per roll each player gets. Here, we have $ 55 = 11 \cdot 5 $ since there are 11 possible numbers and 5 resource types. 

Each vector is formatted as follows:

1. Number of [resource] for player 1 per roll [x]
2. Number of [resource] for player 2 per roll [x]
3. Number of [resource] for player 3 per roll [x]
4. etc. 

We have our vectors ordered as follows:

1. Wheat, roll 2
2. Wood, roll 2
3. Sheep, roll 2
4. Brick, roll 2
5. Stone, roll 2
6. Wheat, roll 3
7. Wood, roll 3
8. etc.

And then, to finish: 53. Sheep, roll 12; 54. Brick, roll 12; 55. Stone, roll 12.

### Robber per roll (250)

This is a vector containing the following information in each coordiante:

1. Player 1 number of cards currently lost to robber on roll
2. Player 2 number of cards currently lost to robber on roll
3. Player 3 number of cards currently lost to robber on roll
4. etc.

### Number of resources (251)

This is a collection of 6 vectors:

1. Number of wheat
2. Number of wood
3. Number of sheep
4. Number of brick
5. Number of stone
6. Number of unknown resources

Each vector is formatted as follows:

1. Number of [resource] in player 1's hand
2. Number of [resource] in player 2's hand
3. Number of [resource] in player 3's hand 
4. etc.

Unknown is in reference that the robber means players will not always know what eachother have.

### Total resources (252)

This is a vector containing the following information in each coordinate:

1. Player 1 total number of resource cards
2. Player 2 total number of resource cards
3. Player 3 total number of resource cards
4. etc.

### Buildings (253–255)

This is a collection of 3 vectors: 

1. Number of settlements
2. Number of cities 
3. Number of roads

Each vector is formatted as follows:

1. Number of [building] in player 1's hand
2. Number of [building] in player 2's hand
3. Number of [building] in player 3's hand
4. etc.

## Moves

### Roll dice (256)

This is a vector with a 1 in the first coordinate if the player can roll the dice.

### Skip turn (257)

TThis is a vector with a 1 in the first coordinate if the player can skip their turn.

### Build settlement (258–311)

This is a collection of 54 vectors with a 1 in the first coordinate if the current player can settle here.

### Build city (312–365)

This is a collection of 54 vectors with a 1 in the first coordinate if the current player can build a city here.

### Build road (366–436)

This is a collection of 71 vectors with a 1 in the first coordinate if the player can build a road there.

### Draw card (437–442)

This is a collection of 6 vectors which contain a 1 in the first coordinate if the player can:

1. Draw a wheat card
2. Draw a wood card
3. Draw a sheep card
4. Draw a brick card
5. Draw a stone card
6. Draw a development card

### Play development card (443–446)

This is a collection of 4 vectors which contain a 1 in the first coordinate if the player can:

1. Play a knight card
2. Play a road building card
3. Play a year of plenty card
4. Play a monopoly card 

### Monopoly rob (447–451)

This is a collection of 5 vectors which contain a 1 in the first coordinate if the player can:

1. Monopoly rob wheat
2. Monopoly rob wood
3. Monopoly rob sheep
4. Monopoly rob brick
5. Monopoly rob stone

### Trade (452-471)

This is a collection of 20 vectors whose first coordinate indicates the player can do the following:

1. Trade wheat for wood
2. Trade wheat for sheep
3. Trade wheat for brick
4. Trade wheat for stone
5. Trade wood for wheat
6. Trade wood for sheep
7. Trade wood for brick
8. Trade wood for stone
9. Trade sheep for wheat
10. Trade sheep for wood
11. Trade sheep for brick
12. Trade sheep for stone
13. Trade brick for wheat
14. Trade brick for wood
15. Trade brick for sheep
16. Trade brick for stone
17. Trade stone for wheat
18. Trade stone for wood
19. Trade stone for sheep
20. Trade stone for brick

### Discard (472–476)

This is a collection of 5 vectors whose first coordinate indicates the player can do the following:

1. Discard wheat
2. Discard wood
3. Discard sheep
4. Discard brick
5. Discard stone 

### Move robber (477–495)

This is a collection of 19 vectors whose first coordinate indicates the player can move the tile to the position.

### Steal card (496-496+[number of players])

This is a collection of [number of players] vectors whose first coordinate indicates the player can do the following:

1. Steal a card from player 1 
2. Steal a card from player 2
3. Steal a card from player 3 
1. etc.
