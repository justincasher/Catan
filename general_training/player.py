######################
## # # # # # # # # # #
# # # CATAN BOT # # ##
## # # # # # # # # # #
######################

from random import shuffle

import torch

class Player :
    """
    Represents a single Catan player.

    Attributes
    ----------
    :type board: Board
        Points to the current board

    :type game: CatanGame
        Points to the current game

    :type army_sze: int 
        Number of largest army cards they have plyed
    :type number: int 
        Player's number in the game
    :type victory_card_points: int
        Number of victory cards they have played

    :type has_largest_army: bool
        Whether they have the largest army
    :type has_longest_road: bool
        Whether they have the longest road
    :type has_played_development_card: bool
        Whether they have played a development card on a given turn
    :type has_rolled: bool
        Whether they have rolled their dice on a given turn

    :type buildings: Dict
        Number of each building available
    :type cards: Dict
        Number of each card in their hand
    :type resources_per_roll: Dict
        Resources to be received for each dice roll
    :type roads_adjacency: Dict
        Adjacency matrix of roads used to find longest road
    :type robber_per_roll: Dict
        Resources currently being robbed each dice roll
    :type trade_ratios: Dict
        Ratio that they can trade each given resource

    :type moves: List
        Number of moves available next turn
    :type predictions: List
        The predictions the player has made throughout 
        the game, used for computing the error
    :type possible_roads: List
        Positions of future possible roads
    :type possible_settlements: List 
        Positions of future possible settlements

    :type cities: Set
        Positions of current cities
    :type roads: Set 
        Positions of current roads
    :type settlements: Set
        Positions of current settlements
    
    :type input_tensor: Torch Tensor
        The board position seen from the current player's
        perspective
    :type moves_tensor: Torch Tensor
        The moves that the player can make
        
    Methods
    -------
    __init__(self, board)
        Constructs all of the player variables
    numberOfResources(self)
        Returns the number of resource cards the player has
    score(self)
        Returns the number of points the player has
    randomDiscard(self)
        Makes the player discard a random card and returns it
    roadLength(self)
        Returns the length of the players longest road
    """


    ####################
    ## Initialization ##
    ####################

    def __init__(self, load_board, load_game, player_number) :
        """
        Initializes the board by setting the board
        and player number.

        :type load_board: Board
            Board pointer
        :type load_game: CatanGame
            Game pointer
        :type player_number: int
            Number of this player
        """

        self.predictions = []

        self.board = load_board  

        self.game = load_game

        self.number = player_number

        self.victory_card_points = 0

        self.army_size, self.has_largest_army = 0, False

        self.has_longest_road = False

        self.has_rolled = False # Resets every turn

        self.has_played_development_card = False # Resets every turn

        self.cards = {
            "Wheat" : 0,
            "Wood" : 0,
            "Sheep" : 0,
            "Brick" : 0,
            "Stone" : 0,
            "Knight" : 0,
            "Victory point" : 0,
            "Road building" : 0,
            "Year of plenty" : 0,
            "Monopoly" : 0
        }

        self.new_development_cards = {
            "Knight" : 0,
            "Road building" : 0,
            "Year of plenty" : 0,
            "Monopoly" : 0,
            "Victory point" : 0
        }

        self.buildings = {
            "Settlements" : 5,
            "Cities" : 4,
            "Roads" : 15
        }

        self.moves = [] 
        self.moves_tensor = torch.zeros(246)
        
        self.input_tensor = torch.zeros([256, 6])

        self.settlements = set()
        self.possible_settlements = set()

        self.cities = set()

        self.roads, self.possible_roads = set(), list()
        self.roads_adjacency = {}

        self.trade_ratios = {
            "Wheat" : 4,
            "Wood" : 4,
            "Sheep" : 4,
            "Brick" : 4,
            "Stone" : 4
        }

        self.resources_per_roll = {
            2 : [],
            3 : [],
            4 : [],
            5 : [],
            6 : [],
            7 : [],
            8 : [],
            9 : [],
            10 : [],
            11 : [],
            12 : [],
        }

        self.robber_per_roll = {
            2 : [],
            3 : [],
            4 : [],
            5 : [],
            6 : [],
            7 : [],
            8 : [],
            9 : [],
            10 : [],
            11 : [],
            12 : [],
        }


    ####################
    ## Get player data #
    ####################

    def numberOfResources(self) : 
        """
        Returns the number of resource cards current in the 
        player's hand.

        :rtype total_resource_cards: int
        """

        total_resource_cards = self.cards["Wheat"] + self.cards["Wood"] + self.cards["Sheep"] + self.cards["Brick"] + self.cards["Stone"]

        return total_resource_cards

    def numberOfDevelopmentCards(self) : 
        """
        Returns the number of development cards the player has.

        :rtype: int
            Sum of development cards
        """

        number_of_development_cards = self.cards["Knight"]
        number_of_development_cards += self.cards["Victory point"]
        number_of_development_cards += self.cards["Road building"]
        number_of_development_cards += self.cards["Year of plenty"]
        number_of_development_cards += self.cards["Monopoly"]

        return number_of_development_cards

    def score(self) : 
        """
        Returns the number of points the player currently has.

        :rtype points: int
        """

        points = 0 

        points += 5 - self.buildings["Settlements"]
        points += 2 * (4 - self.buildings["Cities"])

        if self.has_largest_army :
            points += 2
        
        if self.has_longest_road : 
            points += 2

        points += self.victory_card_points

        return points

    def roadLength(self) : 
        """
        Returns the length of the players longest road.

        :rtype: int
            Longest road length
        """

        output = 0 

        for pos in self.roads_adjacency : 
            output = max(output, self.roadLengthDFS(pos, 0))

        return output

    def roadLengthDFS(self, pos, counter) : 
        """
        Performs a depth-first search to find the longest
        road for the player.

        :type pos: str
            Settlement to perform DFS starting at 
        :type counter: int
            Length of the current longest road
        """

        output = counter 

        for neighbor in self.roads_adjacency[pos] : 
            # Stop if the neighbor is an opponent's settlement 
            # (you cannot pass through)
            for player in self.game.players : 
                if player != self : 
                    if neighbor in player.settlements.union(player.cities) : 
                        continue

            self.roads_adjacency[pos].remove(neighbor)
            self.roads_adjacency[neighbor].remove(pos)

            output = max(output, self.roadLengthDFS(neighbor, counter+1))

            self.roads_adjacency[pos].append(neighbor)
            self.roads_adjacency[neighbor].append(pos)

        return output 
    
    def printCards(self) : 
        """
        Prints the player's current cards.
        """
        total_counted_resources = [str(self.cards[key]) + " " + key.lower() for key in self.cards if self.cards[key] > 0]
            
        print("\nPlayer " + str(self.number) + ", you have " + ", ".join(total_counted_resources) + ".")

    #######################
    ## Get player action ##
    #######################

    def randomDiscard(self) : 
        """
        Returns a random "stolen" card, which is used when
        the robber is played.

        rtype stolen_card: str
        """

        possibilities = []

        possibilities.extend(["Wheat" for x in range(self.cards["Wheat"])])
        possibilities.extend(["Wood" for x in range(self.cards["Wood"])])
        possibilities.extend(["Sheep" for x in range(self.cards["Sheep"])])
        possibilities.extend(["Brick" for x in range(self.cards["Brick"])])
        possibilities.extend(["Stone" for x in range(self.cards["Stone"])])
        
        shuffle(possibilities)

        stolen_card = possibilities.pop()

        self.cards[stolen_card] += -1

        return stolen_card
    