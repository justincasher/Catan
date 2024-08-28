######################
## # # # # # # # # # #
# # # CATAN BOT # # ##
## # # # # # # # # # #
######################

from random import shuffle

class Board :
    """
    Represents the current state of the Catan board.

    Attributes
    ----------
    :type development_cards_played: Dict
        Keys are possible development cards; values are how many
        times they have been played
    :type nearby_settlements: Dict
        Keys are the settlement positions; values are the nearby 
        settlement positions
    :type roads: Dict 
        Keys are the road positions as sets of settlement positions;
        values are the player number that has built there, -1 if no one 
        has built there
    :type road_map: Dict
        Key are the road positions as sets of settlement positions;
        values are the relative positions of the roads in the output
        tensor
    :type settlement_positions: Dict 
        Contains all possible settlement locations still available
    :type settlements: Dict 
        Keys are settlements positions; values are the player number 
        that has settled there, -1 if no one has settled there, and -2 
        if no one can settle there
    :type settlement_map: Dict 
        Keys are settlements positions; values are the relative positions
        of the settlements in the output tensor
    :type ships: Dict
        Keys are the ship positions as sets of settlements positions;
        values are the resources being traded there
    :type ships_map: Dict
        Keys are the ship positions as sets of settlements positions;
        values are the relative positions of the ships in the output
        tensor 
    :type tiles: Dict
        Keys are the tile positions as sets of settlement positions;
        values are tuples (Resource, Dice number)
    :type tiles_map: Dict
        Keys are the tile positions as sets of settlement positions;
        values are the relative positions of the tiles in the output
        tensor 

    :type development_cards: List 
        Acts as the deck of cards
    :type possible_settlement_positions: List
        Contains all of the remaining settlement positions;
        used in the setup

    :type robber: frozen Set 
        Tile that the robber is currently on
        
    Methods
    -------
    __init__(self)
        Construct all of the board variables, and sets of the tiles,
        ships, and development cards
    genResources(self)
        Randomly assigns a (Resource, Dice number) to reach tile
    genShips(self)
        Randomly assigns a resource to each ship
    drawDevelopmentCard(self)
        Removes the top development card and returns it
    """


    def __init__(self) :
        """
        Creates all the variables. Then generates the tile resource 
        and dice numbers, ship values, and shuffles the development 
        cards.
        """
        
        self.settlements = {
            (0, -2) : -1, (0, 0) : -1, (0, 2) : -1,
            (1, -3) : -1, (1, -1) : -1, (1, 1) : -1, (1, 3) : -1,
            (2, -3) : -1, (2, -1) : -1, (2, 1) : -1, (2, 3) : -1,
            (3, -4) : -1, (3, -2) : -1, (3, 0) : -1, (3, 2) : -1, (3, 4) : -1,
            (4, -4) : -1, (4, -2) : -1, (4, 0) : -1, (4, 2) : -1, (4, 4) : -1,
            (5, -5) : -1, (5, -3) : -1, (5, -1) : -1, (5, 1) : -1, (5, 3) : -1, (5, 5) : -1,
            (6, -5) : -1, (6, -3) : -1, (6, -1) : -1, (6, 1) : -1, (6, 3) : -1, (6, 5) : -1,
            (7, -4) : -1, (7, -2) : -1, (7, 0) : -1, (7, 2) : -1, (7, 4) : -1,
            (8, -4) : -1, (8, -2) : -1, (8, 0) : -1, (8, 2) : -1, (8, 4) : -1,
            (9, -3) : -1, (9, -1) : -1, (9, 1) : -1, (9, 3) : -1,
            (10, -3) : -1, (10, -1) : -1, (10, 1) : -1, (10, 3) : -1,
            (11, -2) : -1, (11, 0) : -1, (11, 2) : -1
        }

        self.settlements_map = {
            (0, -2) : 0, (0, 0) : 1, (0, 2) : 2,
            (1, -3) : 3, (1, -1) : 4, (1, 1) : 5, (1, 3) : 6,
            (2, -3) : 7, (2, -1) : 8, (2, 1) : 9, (2, 3) : 10,
            (3, -4) : 11, (3, -2) : 12, (3, 0) : 13, (3, 2) : 14, (3, 4) : 15,
            (4, -4) : 16, (4, -2) : 17, (4, 0) : 18, (4, 2) : 19, (4, 4) : 20,
            (5, -5) : 21, (5, -3) : 22, (5, -1) : 23, (5, 1) : 24, (5, 3) : 25, (5, 5) : 26,
            (6, -5) : 27, (6, -3) : 28, (6, -1) : 29, (6, 1) : 30, (6, 3) : 31, (6, 5) : 32,
            (7, -4) : 33, (7, -2) : 34, (7, 0) : 35, (7, 2) : 36, (7, 4) : 37,
            (8, -4) : 38, (8, -2) : 39, (8, 0) : 40, (8, 2) : 41, (8, 4) : 42,
            (9, -3) : 43, (9, -1) : 44, (9, 1) : 45, (9, 3) : 46,
            (10, -3) : 47, (10, -1) : 48, (10, 1) : 49, (10, 3) : 50,
            (11, -2) : 51, (11, 0) : 52, (11, 2) : 53
        }

        self.roads = {
            frozenset([(1, -3), (0, -2)]) : -1,
            frozenset([(0, -2), (1, -1)]) : -1,
            frozenset([(1, -1), (0, 0)]) : -1,
            frozenset([(0, 0), (1, 1)]) : -1,
            frozenset([(1, 1), (0, 2)]) : -1,
            frozenset([(0, 2), (1, 3)]) : -1,
            frozenset([(1, -3), (2, -3)]) : -1,
            frozenset([(1, -1), (2, -1)]) : -1,
            frozenset([(1, 1), (2, 1)]) : -1,
            frozenset([(1, 3), (2, 3)]) : -1,
            frozenset([(3, -4), (2, -3)]) : -1,
            frozenset([(2, -3), (3, -2)]) : -1,
            frozenset([(3, -2), (2, -1)]) : -1,
            frozenset([(2, -1), (3, 0)]) : -1,
            frozenset([(3, 0), (2, 1)]) : -1,
            frozenset([(2, 1), (3, 2)]) : -1,
            frozenset([(3, 2), (2, 3)]) : -1,
            frozenset([(2, 3), (3, 4)]) : -1,
            frozenset([(3, -4), (4, -4)]) : -1,
            frozenset([(3, -2), (4, -2)]) : -1,
            frozenset([(3, 0), (4, 0)]) : -1,
            frozenset([(3, 2), (4, 2)]) : -1,
            frozenset([(3, 4), (4, 4)]) : -1,
            frozenset([(5, -5), (4, -4)]) : -1,
            frozenset([(4, -4), (5, -3)]) : -1,
            frozenset([(5, -3), (4, -2)]) : -1,
            frozenset([(4, -2), (5, -1)]) : -1,
            frozenset([(5, -1), (4, 0)]) : -1,
            frozenset([(4, 0), (5, 1)]) : -1,
            frozenset([(5, 1), (4, 2)]) : -1,
            frozenset([(4, 2), (5, 3)]) : -1,
            frozenset([(5, 3), (4, 4)]) : -1,
            frozenset([(4, 4), (5, 5)]) : -1,
            frozenset([(5, -5), (6, -5)]) : -1,
            frozenset([(5, -3), (6, -3)]) : -1,
            frozenset([(5, -1), (6, -1)]) : -1,
            frozenset([(5, 1), (6, 1)]) : -1,
            frozenset([(5, 3), (6, 3)]) : -1,
            frozenset([(5, 5), (6, 5)]) : -1,
            frozenset([(6, -5), (7, -4)]) : -1,
            frozenset([(7, -4), (6, -3)]) : -1,
            frozenset([(6, -3), (7, -2)]) : -1,
            frozenset([(7, -2), (6, -1)]) : -1,
            frozenset([(6, -1), (7, 0)]) : -1,
            frozenset([(7, 0), (6, 1)]) : -1,
            frozenset([(6, 1), (7, 2)]) : -1,
            frozenset([(7, 2), (6, 3)]) : -1,
            frozenset([(6, 3), (7, 4)]) : -1,
            frozenset([(7, 4), (6, 5)]) : -1,
            frozenset([(7, -4), (8, -4)]) : -1,
            frozenset([(7, -2), (8, -2)]) : -1,
            frozenset([(7, 0), (8, 0)]) : -1,
            frozenset([(7, 2), (8, 2)]) : -1,
            frozenset([(7, 4), (8, 4)]) : -1,
            frozenset([(8, -4), (9, -3)]) : -1,
            frozenset([(9, -3), (8, -2)]) : -1,
            frozenset([(8, -2), (9, -1)]) : -1,
            frozenset([(9, -1), (8, 0)]) : -1,
            frozenset([(8, 0), (9, 1)]) : -1,
            frozenset([(9, 1), (8, 2)]) : -1,
            frozenset([(8, 2), (9, 3)]) : -1,
            frozenset([(9, 3), (8, 4)]) : -1, 
            frozenset([(9, -3), (10, -3)]) : -1,
            frozenset([(9, -1), (10, -1)]) : -1,
            frozenset([(9, 1), (10, 1)]) : -1,
            frozenset([(9, 3), (10, 3)]) : -1,
            frozenset([(10, -3), (11, -2)]) : -1,
            frozenset([(11, -2), (10, -1)]) : -1,
            frozenset([(10, -1), (11, 0)]) : -1,
            frozenset([(11, 0), (10, 1)]) : -1,
            frozenset([(10, 1), (11, 2)]) : -1,
            frozenset([(11, 2), (10, 3)]) : -1
        }

        self.roads_map = {
            frozenset([(1, -3), (0, -2)]) : 0,
            frozenset([(0, -2), (1, -1)]) : 1,
            frozenset([(1, -1), (0, 0)]) : 2,
            frozenset([(0, 0), (1, 1)]) : 3,
            frozenset([(1, 1), (0, 2)]) : 4,
            frozenset([(0, 2), (1, 3)]) : 5,
            frozenset([(1, -3), (2, -3)]) : 6,
            frozenset([(1, -1), (2, -1)]) : 7,
            frozenset([(1, 1), (2, 1)]) : 8,
            frozenset([(1, 3), (2, 3)]) : 9,
            frozenset([(3, -4), (2, -3)]) : 10,
            frozenset([(2, -3), (3, -2)]) : 11,
            frozenset([(3, -2), (2, -1)]) : 12,
            frozenset([(2, -1), (3, 0)]) : 13,
            frozenset([(3, 0), (2, 1)]) : 14,
            frozenset([(2, 1), (3, 2)]) : 15,
            frozenset([(3, 2), (2, 3)]) : 16,
            frozenset([(2, 3), (3, 4)]) : 17,
            frozenset([(3, -4), (4, -4)]) : 18,
            frozenset([(3, -2), (4, -2)]) : 19,
            frozenset([(3, 0), (4, 0)]) : 20,
            frozenset([(3, 2), (4, 2)]) : 21,
            frozenset([(3, 4), (4, 4)]) : 22,
            frozenset([(5, -5), (4, -4)]) : 23,
            frozenset([(4, -4), (5, -3)]) : 24,
            frozenset([(5, -3), (4, -2)]) : 25,
            frozenset([(4, -2), (5, -1)]) : 26,
            frozenset([(5, -1), (4, 0)]) : 27,
            frozenset([(4, 0), (5, 1)]) : 28,
            frozenset([(5, 1), (4, 2)]) : 29,
            frozenset([(4, 2), (5, 3)]) : 30,
            frozenset([(5, 3), (4, 4)]) : 31,
            frozenset([(4, 4), (5, 5)]) : 32,
            frozenset([(5, -5), (6, -5)]) : 33,
            frozenset([(5, -3), (6, -3)]) : 34,
            frozenset([(5, -1), (6, -1)]) : 35,
            frozenset([(5, 1), (6, 1)]) : 36,
            frozenset([(5, 3), (6, 3)]) : 37,
            frozenset([(5, 5), (6, 5)]) : 38,
            frozenset([(6, -5), (7, -4)]) : 39,
            frozenset([(7, -4), (6, -3)]) : 40,
            frozenset([(6, -3), (7, -2)]) : 41,
            frozenset([(7, -2), (6, -1)]) : 42,
            frozenset([(6, -1), (7, 0)]) : 43,
            frozenset([(7, 0), (6, 1)]) : 44,
            frozenset([(6, 1), (7, 2)]) : 45,
            frozenset([(7, 2), (6, 3)]) : 46,
            frozenset([(6, 3), (7, 4)]) : 47,
            frozenset([(7, 4), (6, 5)]) : 48,
            frozenset([(7, -4), (8, -4)]) : 49,
            frozenset([(7, -2), (8, -2)]) : 50,
            frozenset([(7, 0), (8, 0)]) : 51,
            frozenset([(7, 2), (8, 2)]) : 52,
            frozenset([(7, 4), (8, 4)]) : 53,
            frozenset([(8, -4), (9, -3)]) : 54,
            frozenset([(9, -3), (8, -2)]) : 55,
            frozenset([(8, -2), (9, -1)]) : 56,
            frozenset([(9, -1), (8, 0)]) : 57,
            frozenset([(8, 0), (9, 1)]) : 58,
            frozenset([(9, 1), (8, 2)]) : 59,
            frozenset([(8, 2), (9, 3)]) : 60,
            frozenset([(9, 3), (8, 4)]) : 61,
            frozenset([(9, -3), (10, -3)]) : 62,
            frozenset([(9, -1), (10, -1)]) : 63,
            frozenset([(9, 1), (10, 1)]) : 64,
            frozenset([(9, 3), (10, 3)]) : 65,
            frozenset([(10, -3), (11, -2)]) : 66,
            frozenset([(11, -2), (10, -1)]) : 67,
            frozenset([(10, -1), (11, 0)]) : 68,
            frozenset([(11, 0), (10, 1)]) : 69,
            frozenset([(10, 1), (11, 2)]) : 70,
            frozenset([(11, 2), (10, 3)]) : 71 
        }

        self.tiles = {
            frozenset([(0, -2), (1, -1), (2, -1), (3, -2), (2, -3), (1, -3)]) : "0",
            frozenset([(0, 0), (1, 1), (2, 1), (3, 0), (2, -1), (1, -1)]) : "0",
            frozenset([(0, 2), (1, 3), (2, 3), (3, 2), (2, 1), (1, 1)]) : "0",
            frozenset([(2, -3), (3, -2), (4, -2), (5, -3), (4, -4), (3, -4)]) : "0",
            frozenset([(2, -1), (3, 0), (4, 0), (5, -1), (4, -2), (3, -2)]) : "0",
            frozenset([(2, 1), (3, 2), (4, 2), (5, 1), (4, 0), (3, 0)]) : "0",
            frozenset([(2, 3), (3, 4), (4, 4), (5, 3), (4, 2), (3, 2)]) : "0",
            frozenset([(4, -4), (5, -3), (6, -3), (7, -4), (6, -5), (5, -5)]) : "0",
            frozenset([(4, -2), (5, -1), (6, -1), (7, -2), (6, -3), (5, -3)]) : "0",
            frozenset([(4, 0), (5, 1), (6, 1), (7, 0), (6, -1), (5, -1)]) : "0",
            frozenset([(4, 2), (5, 3), (6, 3), (7, 2), (6, 1), (5, 1)]) : "0",
            frozenset([(4, 4), (5, 5), (6, 5), (7, 4), (6, 3), (5, 3)]) : "0",
            frozenset([(6, -3), (7, -2), (8, 2), (9, -3), (8, -4), (7, -4)]) : "0",
            frozenset([(6, -1), (7, 0), (8, 0), (9, -1), (8, -2), (7, -2)]) : "0",
            frozenset([(6, 1), (7, 2), (8, 2), (9, 1), (8, 0), (7, 0)]) : "0",
            frozenset([(6, 3), (7, 4), (8, 4), (9, 3), (8, 2), (7, 2)]) : "0",
            frozenset([(8, -2), (9, -1), (10, -1), (11, -2), (10, -3), (9, -3)]) : "0",
            frozenset([(8, 0), (9, 1), (10, 1), (11, 0), (10, -1), (9, -1)]) : "0",
            frozenset([(8, 2), (9, 3), (10, 3), (11, 2), (10, 1), (9, 1)]) : "0"
        }

        self.tiles_map = {
            frozenset([(0, -2), (1, -1), (2, -1), (3, -2), (2, -3), (1, -3)]) : 0,
            frozenset([(0, 0), (1, 1), (2, 1), (3, 0), (2, -1), (1, -1)]) : 1,
            frozenset([(0, 2), (1, 3), (2, 3), (3, 2), (2, 1), (1, 1)]) : 2,
            frozenset([(2, -3), (3, -2), (4, -2), (5, -3), (4, -4), (3, -4)]) : 3,
            frozenset([(2, -1), (3, 0), (4, 0), (5, -1), (4, -2), (3, -2)]) : 4,
            frozenset([(2, 1), (3, 2), (4, 2), (5, 1), (4, 0), (3, 0)]) : 5,
            frozenset([(2, 3), (3, 4), (4, 4), (5, 3), (4, 2), (3, 2)]) : 6,
            frozenset([(4, -4), (5, -3), (6, -3), (7, -4), (6, -5), (5, -5)]) : 7,
            frozenset([(4, -2), (5, -1), (6, -1), (7, -2), (6, -3), (5, -3)]) : 8,
            frozenset([(4, 0), (5, 1), (6, 1), (7, 0), (6, -1), (5, -1)]) : 9,
            frozenset([(4, 2), (5, 3), (6, 3), (7, 2), (6, 1), (5, 1)]) : 10,
            frozenset([(4, 4), (5, 5), (6, 5), (7, 4), (6, 3), (5, 3)]) : 11,
            frozenset([(6, -3), (7, -2), (8, 2), (9, -3), (8, -4), (7, -4)]) : 12,
            frozenset([(6, -1), (7, 0), (8, 0), (9, -1), (8, -2), (7, -2)]) : 13,
            frozenset([(6, 1), (7, 2), (8, 2), (9, 1), (8, 0), (7, 0)]) : 14,
            frozenset([(6, 3), (7, 4), (8, 4), (9, 3), (8, 2), (7, 2)]) : 15,
            frozenset([(8, -2), (9, -1), (10, -1), (11, -2), (10, -3), (9, -3)]) : 16,
            frozenset([(8, 0), (9, 1), (10, 1), (11, 0), (10, -1), (9, -1)]) : 17,
            frozenset([(8, 2), (9, 3), (10, 3), (11, 2), (10, 1), (9, 1)]) : 18
        }

        self.ships = {
            frozenset([(0, -2), (1, -3)]) : "0",
            frozenset([(0, 0), (1, 1)]) : "0",
            frozenset([(2, 3), (3, 4)]) : "0",
            frozenset([(3, -4), (4, -4)]) : "0",
            frozenset([(5, 5), (6, 5)]) : "0",
            frozenset([(7, -4), (8, -4)]) : "0",
            frozenset([(8, 4), (9, 3)]) : "0",
            frozenset([(10, -3), (11, -2)]) : "0",
            frozenset([(11, 0), (10, 1)]) : "0"
        }

        self.ships_map = {
            frozenset([(0, -2), (1, -3)]) : 0,
            frozenset([(0, 0), (1, 1)]) : 1,
            frozenset([(2, 3), (3, 4)]) : 2,
            frozenset([(3, -4), (4, -4)]) : 3,
            frozenset([(5, 5), (6, 5)]) : 4,
            frozenset([(7, -4), (8, -4)]) : 5,
            frozenset([(8, 4), (9, 3)]) : 6,
            frozenset([(10, -3), (11, -2)]) : 7,
            frozenset([(11, 0), (10, 1)]) : 8
        }
        
        self.nearby_settlements = {
            (0, -2) : [(1, -3), (1, -1)],
            (0, 0) : [(1, -1), (1, 1)],
            (0, 2) : [(1, 1), (1, 3)],
            (1, -3) : [(0, -2), (2, -3)],
            (1, -1) : [(0, -2), (0, 0), (2, -1)],
            (1, 1) : [(0, 0), (0, 2), (2, 1)],
            (1, 3) : [(0, 2), (2, 3)],
            (2, -3) : [(1, -3), (3, -4), (3, -2)],
            (2, -1) : [(1, -1), (3, -2), (3, 0)],
            (2, 1) : [(1, 1), (3, 0), (3, 2)],
            (2, 3) : [(1, 3), (3, 2), (3, 4)],
            (3, -4) : [(2, -3), (4, -4)],
            (3, -2) : [(2, -3), (2, -1), (4, -2)],
            (3, 0) : [(2, -1), (2, 1), (4, 0)],
            (3, 2) : [(2, 1), (2, 3), (4, 2)],
            (3, 4) : [(2, 3), (4, 4)],
            (4, -4) : [(3, -4), (5, -5), (5, -3)],
            (4, -2) : [(3, -2), (5, -3), (5, -1)],
            (4, 0) : [(3, 0), (5, -1), (5, 1)],
            (4, 2) : [(3, 2), (5, 1), (5, 3)],
            (4, 4) : [(3, 4), (5, 3), (5, 5)],
            (5, -5) : [(4, -4), (6, -5)],
            (5, -3) : [(4, -4), (4, -2), (6, -3)],
            (5, -1) : [(4, -2), (4, 0), (6, -1)],
            (5, 1) : [(4, 0), (4, 2), (6, 1)],
            (5, 3) : [(4, 2), (4, 4), (6, 3)],
            (5, 5) : [(4, 4), (6, 5)],
            (6, -5) : [(5, -5), (7, -4)],
            (6, -3) : [(5, -3), (7, -4), (7, -2)],
            (6, -1) : [(5, -1), (7, -2), (7, 0)],
            (6, 1) : [(5, 1), (7, 0), (7, 2)],
            (6, 3) : [(5, 3), (7, 2), (7, 4)],
            (6, 5) : [(5, 5), (7, 4)],
            (7, -4) : [(6, -5), (6, -3), (8, -4)],
            (7, -2) : [(6, -3), (6, -1), (8, -2)],
            (7, 0) : [(6, -1), (6, 1), (8, 0)],
            (7, 2) : [(6, 1), (6, 3), (8, 2)],
            (7, 4) : [(6, 3), (6, 5), (8, 4)],
            (8, -4) : [(7, -4), (9, -3)],
            (8, -2) : [(7, -2), (9, -3), (9, -1)],
            (8, 0) : [(7, 0), (9, -1), (9, 1)],
            (8, 2) : [(7, 2), (9, 1), (9, 3)],
            (8, 4) : [(7, 4), (9, 3)],
            (9, -3) : [(8, -4), (8, -2), (10, -3)],
            (9, -1) : [(8, -2), (8, 0), (10, -1)],
            (9, 1) : [(8, 0), (8, 2), (10, 1)],
            (9, 3) : [(8, 2), (8, 4), (10, 3)],
            (10, -3) : [(9, -3), (11, -2)],
            (10, -1) : [(9, -1), (11, -2), (11, 0)],
            (10, 1) : [(9, 1), (11, 0), (11, 2)],
            (10, 3) : [(9, 3), (11, 2)],
            (11, -2) : [(10, -3), (10, -1)],
            (11, 0) : [(10, -1), (10, 1)],
            (11, 2) : [(10, 1), (10, 3)]
        }

        self.possible_settlement_positions = list(self.settlements.keys())
        self.possible_settlement_positions.sort()

        self.robber = None
        
        self.development_cards = [
            "Knight", "Knight", "Knight", "Knight", "Knight",
            "Knight", "Knight", "Knight", "Knight", "Knight",
            "Knight", "Knight", "Knight", "Knight",
            "Victory point", "Victory point", "Victory point", "Victory point", "Victory point",
            "Road building", "Road building",
            "Year of plenty", "Year of plenty",
            "Monopoly", "Monopoly"
        ]

        shuffle(self.development_cards)

        self.development_cards_played = {
            "Knight" : 0,
            "Victory point" : 0,
            "Road building" : 0,
            "Year of plenty" : 0,
            "Monopoly" : 0
        }

        # Generate resources and print them
        width = 80
        print("\n" + "─" * width + "\n")
        print("GAME RESOURCES".center(width))

        self.genResources()
        self.genShips()

    def genResources(self) :
        """
        Randomly assigns a resouce and a dice numbers to 
        each tile.
        """
        
        possible_resources = [
            "Wood", "Sheep", "Wheat", 
            "Brick", "Stone", "Brick", "Sheep", 
            "Desert", "Wood", "Wheat", "Wood", "Wheat",
            "Brick", "Sheep", "Sheep", "Stone",
            "Stone", "Wheat", "Wood", 
        ]

        numbers = [
            11, 12, 9, 
            4, 6, 5, 10,
            3, 11, 4, 8, 
            8, 10, 9, 3, 
            5, 2, 6    
        ]

        # shuffle(possible_resources)
        # shuffle(numbers)

        for tile in self.tiles :
            resource = possible_resources.pop()              
            number = 0

            if resource != "Desert" : number = numbers.pop()   
            else : self.robber = tile    

            self.tiles[tile] = (resource, number)

            tile_string = ", ".join([str(x) for x in tile])
            print("{" + tile_string + "}:\t" + resource + ", " + str(number))

    def genShips(self) :
        """
        Randomly assigns a resources to each ship.
        """

        values = [ 
            "Any", "Sheep", "Any", 
            "Stone", "Any", "Wheat", 
            "Brick", "Any", "Wood"
        ]

        shuffle(values) 

        for ship in self.ships:
            value = values.pop()        
            self.ships[ship] = value 

            ship_string = ", ".join([str(x) for x in ship])
            print(ship_string + ": \t" + value)

    def drawDevelopmentCard(self) :
        """
        Draws the top development card.

        :rtype: str
            Development card type
        """

        return self.development_cards.pop()