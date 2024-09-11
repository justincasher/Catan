# Moves tensor

*Note*: When discussing 1-hot vectors, we start at 0, like Python lists, i.e., the coordinates are [0, 1,..., d]. In general, positions are indexed by going left to right like reading.

## Overview

The moves tensor for our neural network is structured as a tensor of size 246 x 1 as follows:

<center> 

|   Row     |       Value       |
| :-------: | :---------------: |
|     0     | Roll dice         |
|     1     | Skip turn         |
|    2–55   | Build settlement  |
|   56–109  | Build city        |
|  110-181  | Build road        |
|  182-187  | Draw card         |
|  188-191  | Play development card |
|  192-196  | Monopoly rob      |
|  197-216  | Trade             |
|  217-221  | Discard           |
|  222-240  | Move robber       |
|  241-246  | Steal card        |

</center>


### Roll dice (0)

This is a vector with a 1 in the first coordinate if the player can roll the dice.

### Skip turn (1)

TThis is a vector with a 1 in the first coordinate if the player can skip their turn.

### Build settlement (2-55)

This is a collection of 54 vectors with a 1 in the first coordinate if the current player can settle here.

### Build city (56–109)

This is a collection of 54 vectors with a 1 in the first coordinate if the current player can build a city here.

### Build road (110–181)

This is a collection of 71 vectors with a 1 in the first coordinate if the player can build a road there.

### Draw card (182–187)

This is a collection of 6 vectors which contain a 1 in the first coordinate if the player can:

1. Draw a wheat card
2. Draw a wood card
3. Draw a sheep card
4. Draw a brick card
5. Draw a stone card
6. Draw a development card

### Play development card (188-191)

This is a collection of 4 vectors which contain a 1 in the first coordinate if the player can:

1. Play a knight card
2. Play a road building card
3. Play a year of plenty card
4. Play a monopoly card 

### Monopoly rob (192–196)

This is a collection of 5 vectors which contain a 1 in the first coordinate if the player can:

1. Monopoly rob wheat
2. Monopoly rob wood
3. Monopoly rob sheep
4. Monopoly rob brick
5. Monopoly rob stone

### Trade (197-216)

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

### Discard (217–221)

This is a collection of 5 vectors whose first coordinate indicates the player can do the following:

1. Discard wheat
2. Discard wood
3. Discard sheep
4. Discard brick
5. Discard stone

### Move robber (222–240)

This is a collection of 19 vectors whose first coordinate indicates the player can move the tile to the position.

### Steal card (241-241+[number of players])

This is a collection of [number of players] vectors whose first coordinate indicates the player can do the following:

1. Steal a card from player 1 
2. Steal a card from player 2
3. Steal a card from player 3 
1. etc.
