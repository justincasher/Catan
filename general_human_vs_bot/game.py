######################
## # # # # # # # # # #
# # # CATAN BOT # # ##
## # # # # # # # # # #
######################

from random import randint

from collections import Counter

import torch 

from board import Board
from player import Player

class CatanGame :
    """
    Acts as the underlying game play for a Catan game.

    Parameters
    ----------
    :type board: Board
        Pointer to the board

    :type net: CatanNetwork
        The model being trained

    :type moves_dict: Dict
        Keys are the integers for moves in the players 
        moves_tensor; values are the associated move 
        return type

    :type turn_number: int 
        The current turn we are on, i.e., the number of times the
        dice have been rolled
        
    :type players: List
        Pointer to each player in the game

    :type device: torch.device
        The device (CPU or CUDA) that the network will be trained on
    """

    ####################
    ## Initialization ##
    ####################

    def __init__(self, number_of_players, net, human_number) :
        """
        Initializes the game by creating the board and players.

        :type player_nets: List
            Neural networks used to make players decisions
        """

        self.number_of_rolls = 0

        self.human_number = human_number

        self.net = net

        self.board = Board()

        self.players = []

        for i in range(number_of_players) : 
            if i == human_number : 
                self.players.append(Player(
                    load_board=self.board, 
                    load_game=self, 
                    player_number=i, 
                    is_human=True,
                ))
            else : 
                self.players.append(Player(
                    load_board=self.board, 
                    load_game=self, 
                    player_number=i, 
                    is_human=False,
                ))

        self.moves_dict = {
            0 : "Roll dice",
            1 : "Skip turn",
            2 : ("Build settlement", (0, -2)),
            3 : ("Build settlement", (0, 0)),
            4 : ("Build settlement", (0, 2)),
            5 : ("Build settlement", (1, -3)),
            6 : ("Build settlement", (1, -1)),
            7 : ("Build settlement", (1, 1)),
            8 : ("Build settlement", (1, 3)),
            9 : ("Build settlement", (2, -3)),
            10 : ("Build settlement", (2, -1)),
            11 : ("Build settlement", (2, 1)),
            12 : ("Build settlement", (2, 3)),
            13 : ("Build settlement", (3, -4)),
            14 : ("Build settlement", (3, -2)),
            15 : ("Build settlement", (3, 0)),
            16 : ("Build settlement", (3, 2)),
            17 : ("Build settlement", (3, 4)),
            18 : ("Build settlement", (4, -4)),
            19 : ("Build settlement", (4, -2)),
            20 : ("Build settlement", (4, 0)),
            21 : ("Build settlement", (4, 2)),
            22 : ("Build settlement", (4, 4)),
            23 : ("Build settlement", (5, -5)),
            24 : ("Build settlement", (5, -3)),
            25 : ("Build settlement", (5, -1)),
            26 : ("Build settlement", (5, 1)),
            27 : ("Build settlement", (5, 3)),
            28 : ("Build settlement", (5, 5)),
            29 : ("Build settlement", (6, -5)),
            30 : ("Build settlement", (6, -3)),
            31 : ("Build settlement", (6, -1)),
            32 : ("Build settlement", (6, 1)),
            33 : ("Build settlement", (6, 3)),
            34 : ("Build settlement", (6, 5)),
            35 : ("Build settlement", (7, -4)),
            36 : ("Build settlement", (7, -2)),
            37 : ("Build settlement", (7, 0)),
            38 : ("Build settlement", (7, 2)),
            39 : ("Build settlement", (7, 4)),
            40 : ("Build settlement", (8, -4)),
            41 : ("Build settlement", (8, -2)),
            42 : ("Build settlement", (8, 0)),
            43 : ("Build settlement", (8, 2)),
            44 : ("Build settlement", (8, 4)),
            45 : ("Build settlement", (9, -3)),
            46 : ("Build settlement", (9, -1)),
            47 : ("Build settlement", (9, 1)),
            48 : ("Build settlement", (9, 3)),
            49 : ("Build settlement", (10, -3)),
            50 : ("Build settlement", (10, -1)),
            51 : ("Build settlement", (10, 1)),
            52 : ("Build settlement", (10, 3)),
            53 : ("Build settlement", (11, -2)),
            54 : ("Build settlement", (11, 0)),
            55 : ("Build settlement", (11, 2)),
            56 : ("Build city", (0, -2)),
            57 : ("Build city", (0, 0)),
            58 : ("Build city", (0, 2)),
            59 : ("Build city", (1, -3)),
            60 : ("Build city", (1, -1)),
            61 : ("Build city", (1, 1)),
            62 : ("Build city", (1, 3)),
            63 : ("Build city", (2, -3)),
            64 : ("Build city", (2, -1)),
            65 : ("Build city", (2, 1)),
            66 : ("Build city", (2, 3)),
            67 : ("Build city", (3, -4)),
            68 : ("Build city", (3, -2)),
            69 : ("Build city", (3, 0)),
            70 : ("Build city", (3, 2)),
            71 : ("Build city", (3, 4)),
            72 : ("Build city", (4, -4)),
            73 : ("Build city", (4, -2)),
            74 : ("Build city", (4, 0)),
            75 : ("Build city", (4, 2)),
            76 : ("Build city", (4, 4)),
            77 : ("Build city", (5, -5)),
            78 : ("Build city", (5, -3)),
            79 : ("Build city", (5, -1)),
            80 : ("Build city", (5, 1)),
            81 : ("Build city", (5, 3)),
            82 : ("Build city", (5, 5)),
            83 : ("Build city", (6, -5)),
            84 : ("Build city", (6, -3)),
            85 : ("Build city", (6, -1)),
            86 : ("Build city", (6, 1)),
            87 : ("Build city", (6, 3)),
            88 : ("Build city", (6, 5)),
            89 : ("Build city", (7, -4)),
            90 : ("Build city", (7, -2)),
            91 : ("Build city", (7, 0)),
            92 : ("Build city", (7, 2)),
            93 : ("Build city", (7, 4)),
            94 : ("Build city", (8, -4)),
            95 : ("Build city", (8, -2)),
            96 : ("Build city", (8, 0)),
            97 : ("Build city", (8, 2)),
            98 : ("Build city", (8, 4)),
            99 : ("Build city", (9, -3)),
            100 : ("Build city", (9, -1)),
            101 : ("Build city", (9, 1)),
            102 : ("Build city", (9, 3)),
            103 : ("Build city", (10, -3)),
            104 : ("Build city", (10, -1)),
            105 : ("Build city", (10, 1)),
            106 : ("Build city", (10, 3)),
            107 : ("Build city", (11, -2)),
            108 : ("Build city", (11, 0)),
            109 : ("Build city", (11, 2)), 
            110 : ("Build road", frozenset({(1, -3), (0, -2)})),
            111 : ("Build road", frozenset({(0, -2), (1, -1)})),
            112 : ("Build road", frozenset({(1, -1), (0, 0)})),
            113 : ("Build road", frozenset({(1, 1), (0, 0)})),
            114 : ("Build road", frozenset({(1, 1), (0, 2)})),
            115 : ("Build road", frozenset({(0, 2), (1, 3)})),
            116 : ("Build road", frozenset({(2, -3), (1, -3)})),
            117 : ("Build road", frozenset({(2, -1), (1, -1)})),
            118 : ("Build road", frozenset({(1, 1), (2, 1)})),
            119 : ("Build road", frozenset({(2, 3), (1, 3)})),
            120 : ("Build road", frozenset({(2, -3), (3, -4)})),
            121 : ("Build road", frozenset({(2, -3), (3, -2)})),
            122 : ("Build road", frozenset({(2, -1), (3, -2)})),
            123 : ("Build road", frozenset({(2, -1), (3, 0)})),
            124 : ("Build road", frozenset({(2, 1), (3, 0)})),
            125 : ("Build road", frozenset({(3, 2), (2, 1)})),
            126 : ("Build road", frozenset({(2, 3), (3, 2)})),
            127 : ("Build road", frozenset({(2, 3), (3, 4)})),
            128 : ("Build road", frozenset({(3, -4), (4, -4)})),
            129 : ("Build road", frozenset({(4, -2), (3, -2)})),
            130 : ("Build road", frozenset({(4, 0), (3, 0)})),
            131 : ("Build road", frozenset({(3, 2), (4, 2)})),
            132 : ("Build road", frozenset({(4, 4), (3, 4)})),
            133 : ("Build road", frozenset({(5, -5), (4, -4)})),
            134 : ("Build road", frozenset({(5, -3), (4, -4)})),
            135 : ("Build road", frozenset({(5, -3), (4, -2)})),
            136 : ("Build road", frozenset({(4, -2), (5, -1)})),
            137 : ("Build road", frozenset({(5, -1), (4, 0)})),
            138 : ("Build road", frozenset({(4, 0), (5, 1)})),
            139 : ("Build road", frozenset({(5, 1), (4, 2)})),
            140 : ("Build road", frozenset({(5, 3), (4, 2)})),
            141 : ("Build road", frozenset({(5, 3), (4, 4)})),
            142 : ("Build road", frozenset({(4, 4), (5, 5)})),
            143 : ("Build road", frozenset({(6, -5), (5, -5)})),
            144 : ("Build road", frozenset({(5, -3), (6, -3)})),
            145 : ("Build road", frozenset({(5, -1), (6, -1)})),
            146 : ("Build road", frozenset({(6, 1), (5, 1)})),
            147 : ("Build road", frozenset({(5, 3), (6, 3)})),
            148 : ("Build road", frozenset({(5, 5), (6, 5)})),
            149 : ("Build road", frozenset({(6, -5), (7, -4)})),
            150 : ("Build road", frozenset({(6, -3), (7, -4)})),
            151 : ("Build road", frozenset({(7, -2), (6, -3)})),
            152 : ("Build road", frozenset({(7, -2), (6, -1)})),
            153 : ("Build road", frozenset({(7, 0), (6, -1)})),
            154 : ("Build road", frozenset({(6, 1), (7, 0)})),
            155 : ("Build road", frozenset({(6, 1), (7, 2)})),
            156 : ("Build road", frozenset({(6, 3), (7, 2)})),
            157 : ("Build road", frozenset({(7, 4), (6, 3)})),
            158 : ("Build road", frozenset({(7, 4), (6, 5)})),
            159 : ("Build road", frozenset({(8, -4), (7, -4)})),
            160 : ("Build road", frozenset({(7, -2), (8, -2)})),
            161 : ("Build road", frozenset({(7, 0), (8, 0)})),
            162 : ("Build road", frozenset({(8, 2), (7, 2)})),
            163 : ("Build road", frozenset({(7, 4), (8, 4)})),
            164 : ("Build road", frozenset({(8, -4), (9, -3)})),
            165 : ("Build road", frozenset({(8, -2), (9, -3)})),
            166 : ("Build road", frozenset({(8, -2), (9, -1)})),
            167 : ("Build road", frozenset({(8, 0), (9, -1)})),
            168 : ("Build road", frozenset({(9, 1), (8, 0)})),
            169 : ("Build road", frozenset({(8, 2), (9, 1)})),
            170 : ("Build road", frozenset({(8, 2), (9, 3)})),
            171 : ("Build road", frozenset({(9, 3), (8, 4)})),
            172 : ("Build road", frozenset({(9, -3), (10, -3)})),
            173 : ("Build road", frozenset({(10, -1), (9, -1)})),
            174 : ("Build road", frozenset({(9, 1), (10, 1)})),
            175 : ("Build road", frozenset({(9, 3), (10, 3)})),
            176 : ("Build road", frozenset({(11, -2), (10, -3)})),
            177 : ("Build road", frozenset({(10, -1), (11, -2)})),
            178 : ("Build road", frozenset({(10, -1), (11, 0)})),
            179 : ("Build road", frozenset({(10, 1), (11, 0)})),
            180 : ("Build road", frozenset({(11, 2), (10, 1)})),
            181 : ("Build road", frozenset({(11, 2), (10, 3)})),
            182 : ("Draw card", "Wheat"),
            183 : ("Draw card", "Wood"),
            184 : ("Draw card", "Sheep"),
            185 : ("Draw card", "Brick"),
            186 : ("Draw card", "Stone"),
            187 : ("Draw card", "Development"),
            188 : ("Play development", "Knight"),
            189 : ("Play development", "Road building"),
            190 : ("Play development", "Year of plenty"),
            191 : ("Play development", "Monopoly"),
            192 : ("Monopoly rob", "Wheat"),
            193 : ("Monopoly rob", "Wood"),
            194 : ("Monopoly rob", "Sheep"),
            195 : ("Monopoly rob", "Brick"),
            196 : ("Monopoly rob", "Stone"),
            197 : ("Trade", "Wheat", "Wood"),
            198 : ("Trade", "Wheat", "Sheep"),
            199 : ("Trade", "Wheat", "Brick"),
            200 : ("Trade", "Wheat", "Stone"),
            201 : ("Trade", "Wood", "Wheat"),
            202 : ("Trade", "Wood", "Sheep"),
            203 : ("Trade", "Wood", "Brick"),
            204 : ("Trade", "Wood", "Stone"),
            205 : ("Trade", "Sheep", "Wheat"),
            206 : ("Trade", "Sheep", "Wood"),
            207 : ("Trade", "Sheep", "Brick"),
            208 : ("Trade", "Sheep", "Stone"),
            209 : ("Trade", "Brick", "Wheat"),
            210 : ("Trade", "Brick", "Wood"),
            211 : ("Trade", "Brick", "Sheep"),
            212 : ("Trade", "Brick", "Stone"),
            213 : ("Trade", "Stone", "Wheat"),
            214 : ("Trade", "Stone", "Wood"),
            215 : ("Trade", "Stone", "Sheep"),
            216 : ("Trade", "Stone", "Brick"),
            217 : ("Discard", "Wheat"),
            218 : ("Discard", "Wood"),
            219 : ("Discard", "Sheep"),
            220 : ("Discard", "Brick"),
            221 : ("Discard", "Stone"),
            222 : ("Move robber", frozenset([(0, -2), (1, -1), (2, -1), (3, -2), (2, -3), (1, -3)])),
            223 : ("Move robber", frozenset([(0, 0), (1, 1), (2, 1), (3, 0), (2, -1), (1, -1)])),
            224 : ("Move robber", frozenset([(0, 2), (1, 3), (2, 3), (3, 2), (2, 1), (1, 1)])),
            225 : ("Move robber", frozenset([(2, -3), (3, -2), (4, -2), (5, -3), (4, -4), (3, -4)])),
            226 : ("Move robber", frozenset([(2, -1), (3, 0), (4, 0), (5, -1), (4, -2), (3, -2)])),
            227 : ("Move robber", frozenset([(2, 1), (3, 2), (4, 2), (5, 1), (4, 0), (3, 0)])),
            228 : ("Move robber", frozenset([(2, 3), (3, 4), (4, 4), (5, 3), (4, 2), (3, 2)])),
            229 : ("Move robber", frozenset([(4, -4), (5, -3), (6, -3), (7, -4), (6, -5), (5, -5)])),
            230 : ("Move robber", frozenset([(4, -2), (5, -1), (6, -1), (7, -2), (6, -3), (5, -3)])),
            231 : ("Move robber", frozenset([(4, 0), (5, 1), (6, 1), (7, 0), (6, -1), (5, -1)])),
            232 : ("Move robber", frozenset([(4, 2), (5, 3), (6, 3), (7, 2), (6, 1), (5, 1)])),
            233 : ("Move robber", frozenset([(4, 4), (5, 5), (6, 5), (7, 4), (6, 3), (5, 3)])),
            234 : ("Move robber", frozenset([(6, -3), (7, -2), (8, 2), (9, -3), (8, -4), (7, -4)])),
            235 : ("Move robber", frozenset([(6, -1), (7, 0), (8, 0), (9, -1), (8, -2), (7, -2)])),
            236 : ("Move robber", frozenset([(6, 1), (7, 2), (8, 2), (9, 1), (8, 0), (7, 0)])),
            237 : ("Move robber", frozenset([(6, 3), (7, 4), (8, 4), (9, 3), (8, 2), (7, 2)])),
            238: ("Move robber", frozenset([(8, -2), (9, -1), (10, -1), (11, -2), (10, -3), (9, -3)])),
            239 : ("Move robber", frozenset([(8, 0), (9, 1), (10, 1), (11, 0), (10, -1), (9, -1)])),
            240 : ("Move robber", frozenset([(8, 2), (9, 3), (10, 3), (11, 2), (10, 1), (9, 1)])),
            241 : ("Steal card", 0),
            242 : ("Steal card", 1),
            243 : ("Steal card", 2),
            244 : ("Steal card", 3),
            245 : ("Steal card", 4)
        }

    
    ######################
    ## Building methods ##
    ######################

    def buildSettlement(self, player, settlement_pos, turn_type) : 
        """
        Adds a given position to the settlements set.
        Also appends the nearby resources to resources_per_roll.

        Parameters 
        ----------
        :type player: Player
            Player placing the settlement
        :type settlement_pos: tuple
            Position of the new settlement
        :type turn_type: int
            0 if a normal turn, else the setup turn number (1 or 2)
        """

        player.settlements.add(settlement_pos)
        player.buildings["Settlements"] -= 1

        # Add resources to their per turn resources.
        for tile in self.board.tiles : 
            if settlement_pos in tile : 
                resource, number = self.board.tiles[tile]
                    
                if resource != "Desert" :
                    if tile == self.board.robber : 
                        player.robber_per_roll[number].append(resource)
                    else : 
                        player.resources_per_roll[number].append(resource)

        # If it is their second setup turn, give the player a
        # resource for each adjacent resource.
        if turn_type == 2 : 
            resources_gained = []

            for tile in self.board.tiles : 
                if settlement_pos in tile : 
                    resource, number = self.board.tiles[tile]

                    if resource != "Desert": 
                        player.cards[resource] += 1
                        resources_gained.append(resource)

            counted_resources = Counter(resources_gained)
            output_list = [str(counted_resources[key]) + " " + key.lower() for key in counted_resources]

            print("\nPlayer " + str(player.number) + ", you got " + ", ".join(output_list) + ".")
            
        # Update the board.
        self.board.settlements[settlement_pos] = player.number
        self.board.possible_settlement_positions.remove(settlement_pos)
        
        for nearby in self.board.nearby_settlements[settlement_pos] : 
            if nearby in self.board.possible_settlement_positions : 
                self.board.settlements[nearby] = -2
                self.board.possible_settlement_positions.remove(nearby)

        # If it is not a setup move, subtract cards and print current cards
        if turn_type == 0 : 
            player.cards["Wood"] += -1
            player.cards["Brick"] += -1
            player.cards["Wheat"] += -1
            player.cards["Sheep"] += -1

            player.printCards()

        self.updatePossibleRoads()
        self.updatePossibleSettlements()

    def buildCity(self, player, city_pos) : 
        """
        Adds a given position to the cities set. Also appends 
        the nearby resources to resources_per_roll.

        Parameters 
        ----------
        :type player: Player
            Player building the city
        :type city_pos: tuple
            Position of the new city
        """

        # Remove the settlement for the player
        player.settlements.remove(city_pos)
        player.buildings["Settlements"] += 1

        # Add the city for the player
        player.cities.add(city_pos)
        player.buildings["Cities"] -= 1

        # Add resources to their per turn resources.
        for tile in self.board.tiles : 
            if city_pos in tile : 
                resource, number = self.board.tiles[tile]
                
                if resource != "Desert" :
                    if tile == self.board.robber : 
                        player.robber_per_roll[number].append(resource)
                    else : 
                        player.resources_per_roll[number].append(resource)

        # Remove cards from hand
        player.cards["Wheat"] += -2
        player.cards["Stone"] += -3

        player.printCards()

    def buildRoad(self, player, road_pos, turn_type) : 
        """
        Adds a road the roads set and to the adjacency dictionary.
        Removes the road position from possible roads and adds new
        road positions.

        Parameters 
        ----------
        :type player: Player
            Player building the road.
        :type pos: frozen Set
            Positions of the new road.
        :type turn_type: int
            0 if a normal turn, or 1 if not to discard resources
        """

        self.board.roads[road_pos] = player.number

        if turn_type == 0 :
            player.cards["Wood"] += -1
            player.cards["Brick"] += -1
            player.printCards()

        player.roads.add(road_pos)
        
        list_pos = list(road_pos)
        start, end = list_pos[0], list_pos[1]
        
        if start in player.roads_adjacency : 
            player.roads_adjacency[start].append(end)
        else : 
            player.roads_adjacency[start] = [end]
        
        if end in player.roads_adjacency : 
            player.roads_adjacency[end].append(start)
        else : 
            player.roads_adjacency[end] = [start]

        self.updatePossibleRoads()
        self.updatePossibleSettlements()
    
    def updatePossibleSettlements(self) : 
        """
        Updates the possible setlement positions for each 
        player in the game.
        """

        for update_player in self.players : 
            update_player.possible_settlements.clear()

            for possible_settlement in self.board.possible_settlement_positions : 
                if any([possible_settlement in road for road in update_player.roads]) : 
                    update_player.possible_settlements.add(possible_settlement)

    def updatePossibleRoads(self) : 
        """
        Updates the possible road positions for each player
        in the game.
        """

        for player in self.players : 
            player.possible_roads.clear()

            for possible_road in self.board.roads : 
                if self.board.roads[possible_road] == -1 : 
                    if any([not possible_road.isdisjoint(road) for road in player.roads]) :
                        player.possible_roads.append(possible_road)

            player.possible_roads.sort()


    ####################
    ## Robber methods ##
    ####################

    def robberDiscard(self) : 
        """
        Makes the player discard half of their cards if they
        have more than 7 (i.e., >= 8).
        """

        for update_player in self.players : 

            num_resources = update_player.numberOfResources()

            if num_resources >= 8 : 
                discard_number = num_resources // 2 

                for i in range(discard_number) : 
                    update_player.moves.clear()
                    update_player.printCards()

                    for resource in ["Wheat", "Wood", "Sheep", "Brick", "Stone"] :
                        if update_player.cards[resource] != 0 : 
                            update_player.moves.append(("Discard", resource))

                    x, resource = self.makeMove(update_player)

                    update_player.cards[resource] += -1

                update_player.printCards()

    def placeRobber(self, player) : 
        """
        Makes the player who rolled the robber choose a new 
        tile to place it on.

        Parameters 
        ----------
        player
            Player placing the robber.
        """

        ## Step 1: Choose which tile to place the robber
        ## on. 

        player.moves.clear()

        for tile in self.board.tiles : 
            if tile != self.board.robber : 
                player.moves.append(("Move robber", tile))

        x, new_tile = self.makeMove(player)

        player.board.robber = new_tile


        ## Step 2: Steal a resource card from another 
        ## player.

        player.moves.clear()

        possible_players = []

        for player_robbed in self.players : 
            if player_robbed != player and player_robbed.numberOfResources() > 0 : 
                for pos in player_robbed.settlements.union(player_robbed.cities) : 
                    if pos in new_tile : 
                        possible_players.append(player_robbed)
                        break
        
        if possible_players : 
            for player_robbed in possible_players : 
                player.moves.append(("Steal card", player_robbed.number))
            
            x, num = self.makeMove(player)
            player_robbed = self.players[num]
            resource = player_robbed.randomDiscard()
            player.cards[resource] += 1

            print("\nPlayer " + str(player.number) + ", you stole a " + resource.lower() + " from player " + str(player_robbed.number))

            player_robbed.printCards()
            player.printCards()
           
    def updateRobberResources(self) : 
        """
        When a new robber location is chosen, this function
        adjusts the player resource count per roll.
        """

        for update_player in self.players : 

            ## Step 1: Add back the resources which were being stolen
            ## by the robber to the player's resources per roll 
            ## dictionary.

            for number in update_player.robber_per_roll : 
                update_player.resources_per_roll[number].extend(update_player.robber_per_roll[number])
                update_player.robber_per_roll[number].clear()


            ## Step 2: Take away the resources which are now being 
            ## stolen by the robber.

            resource, number = self.board.tiles[self.board.robber]

            if resource == "Desert" :
                break

            for settlement in update_player.settlements : 
                if settlement in self.board.robber :
                    update_player.resources_per_roll[number].remove(resource)

                    update_player.robber_per_roll[number].append(resource)

            for city in update_player.cities : 
                if city in self.board.robber :
                    update_player.resources_per_roll[number].remove(resource)
                    update_player.resources_per_roll[number].remove(resource)

                    update_player.robber_per_roll[number].append(resource)
                    update_player.robber_per_roll[number].append(resource)


    ##############################
    ## Development card methods ##
    ##############################

    def drawDevelopmentCard(self, player) : 
        """
        Draws a development card for the player. Adds it to the
        list of development cards the player cannot play this 
        turn.

        :type player: player
            Player drawing the development card.
        """

        card = self.board.drawDevelopmentCard()

        player.cards["Sheep"] += -1
        player.cards["Wheat"] += -1
        player.cards["Stone"] += -1

        if player.is_human : print("\nPlayer " + str(player.number) + ", you drew a " + card.lower() + " card")

        if card == "Victory point" : 
            player.victory_card_points += 1
        else : 
            player.cards[card] += 1
            player.new_development_cards[card] += 1
            player.printCards()

    def playKnight(self, player, is_development_card) : 
        """
        Allows the player to play a knight, either by rolling
        a 7 or by plaing a development card.

        :type player: player
            Player playing the knight.
        :type is_development_card: boolean
            Tells us whether to increase largest army size.
        """

        self.placeRobber(player)
        self.updateRobberResources()
        
        if is_development_card : 
            player.cards["Knight"] += -1
            player.army_size += 1

    def playRoadBuilding(self, player) : 
        """
        Allows the player to place two roads free of cost
        by playing their road building development card.

        Parameters 
        ----------
        player
            Player playing road building.
        """

        player.moves.clear()

        for i in range(min(2, player.buildings["Roads"])) : 
            for pos in player.possible_roads:
                player.moves.append(("Build road", pos))
        
            if player.moves : 
                x, chosen_road_pos = self.makeMove(player)
                self.buildRoad(player, pos, 0)

            self.updatePossibleRoads()
            self.updatePossibleSettlements()

        player.cards["Road building"] += -1

    def playYearOfPlenty(self, player) : 
        """
        Allows the player to draw two cards free of cost
        by playing their year of plenty card.

        Parameters 
        ----------
        player
            Player playing year of plenty.
        """

        for i in range(2) : 
            player.moves.clear()
            player.moves.append(("Draw card", "Wheat"))
            player.moves.append(("Draw card", "Wood"))
            player.moves.append(("Draw card", "Sheep"))
            player.moves.append(("Draw card", "Brick"))
            player.moves.append(("Draw card", "Stone"))

            _, chosen_resource = self.makeMove(player)

            player.cards[chosen_resource] += 1

        player.cards["Year of plenty"] += -1

        player.printCards()

    def playMonopoly(self, player) : 
        """
        Allows the player to draw two cards free of cost
        by playing their year of plenty card.

        Parameters 
        ----------
        player
            Player playing year of plenty.
        """

        player.moves.clear()
        player.moves.append(("Monopoly rob", "Wheat"))
        player.moves.append(("Monopoly rob", "Wood"))
        player.moves.append(("Monopoly rob", "Sheep"))
        player.moves.append(("Monopoly rob", "Brick"))
        player.moves.append(("Monopoly rob", "Stone"))

        x, chosen_resource = self.makeMove(player)

        for robbed_player in self.players : 
            if robbed_player != player : 
                player.cards[chosen_resource] += robbed_player.cards[chosen_resource]
                robbed_player.cards[chosen_resource] = 0
                robbed_player.printCards()

        player.cards["Monopoly"] += -1
        player.printCards()
        

    ###################
    ## 2 point cards ##
    ###################

    def largestArmy(self) : 
        """
        Updates which player has the largest army card.
        """

        largest_army = 2
        largest_army_player = None

        for player in self.players : 
            if player.has_largest_army : 
                largest_army = player.army_size
                largest_army_player = player

        for player in self.players : 
            if player != largest_army_player and player.army_size > largest_army :
                largest_army = player.army_size

                player.has_largest_army = True

                if largest_army_player != None : 
                    largest_army_player.has_largest_army = False

                largest_army_player = player

    def longestRoad(self) : 
        """
        Updates which player has the longest road card.
        """

        longest_road = 4
        longest_road_player = None

        for player in self.players : 
            if player.has_longest_road :
                longest_road = player.roadLength()
                longest_road_player = player

        for player in self.players :
            player_road_length = player.roadLength()
            if player != longest_road_player and player_road_length > longest_road : 
                longest_road = player_road_length

                player.has_longest_road = True

                if longest_road_player != None : 
                    longest_road_player.has_longest_road = False

                longest_road_player = player


    ####################
    ## Trading method ##
    ####################

    def makeTrade(self, player, trade_resource, new_resource) : 
        """
        Removes the given ratio of the ratio to be traded from 
        the player's cards, and adds one of the new resource.

        :type player: Player
            The player making the trade
        :type trade_resource: str
            The resource being traded
        :type new_resource: str
            The resource being traded for
        """

        trade_ratio = player.trade_ratios[trade_resource]
        player.cards[trade_resource] += -trade_ratio
        player.cards[new_resource] += 1

        player.printCards()


    ###################
    ## Setup methods ##
    ###################

    def startMove(self, player, setup_move_number) : 
        """
        Performs a starting move for a given player.

        Parameters
        ----------
        :type player: Player
            Player who is performing the move.
        :type setup_move_number: int 
            Whether this is the first or second setup move
        """


        ## Step 1: Building a settlement.

        player.moves.clear()
        
        for pos in self.board.possible_settlement_positions : 
            player.moves.append(("Build settlement", pos))

        self.generateInputTensor(player)
        
        x, chosen_settlement_pos = self.makeMove(player)

        self.buildSettlement(player, chosen_settlement_pos, setup_move_number)


        ## Step 2: Building a road.

        player.moves.clear()

        for nearby in self.board.nearby_settlements[chosen_settlement_pos] : 
            player.moves.append(("Build road", frozenset([chosen_settlement_pos, nearby])))

        self.generateInputTensor(player)

        x, chosen_road_pos = self.makeMove(player)
            
        self.buildRoad(player, chosen_road_pos, turn_type=1)

    def setup(self) : 
        """
        Runs the setup stage of the game, by allowing the 
        players to place their first settlement and road.
        This is done in ascending order, then in descending
        order.
        """

        for player in self.players : 
            # Inform player that it is their move
            width = 80
            print("\n" + "─" * width + "\n")

            turn = "PLAYER " + str(player.number) + "'S 1st SETUP TURN"
            print(turn.center(width))

            # Make setup move
            self.startMove(player, setup_move_number=1)

        for player in self.players[::-1] : 
            # Inform player that it is their move
            width = 80
            print("\n" + "─" * width + "\n")

            turn = "PLAYER " + str(player.number) + "'S 2nd SETUP TURN"
            print(turn.center(width))

            # Make setup move
            self.startMove(player, setup_move_number=2)


    ########################
    ## Generation methods ##
    ########################
    
    def generateMoves(self, player) :
        """
        Generates the moves list given the player's 
        current resources, buildings, and positions.

        :type player: Player
            The player whose moves list is being generated
        """

        player.moves.clear()
        
        if player.has_rolled == False : 
            player.moves.append("Roll dice")

        else : 
            player.moves.append("Skip turn")

            # Roads
            if player.cards["Brick"] >= 1 and player.cards["Wood"] >= 1 and player.buildings["Roads"] >= 1 :
                for pos in player.possible_roads:
                    player.moves.append(("Build road", pos))

            # Settlements
            if player.cards["Brick"] >= 1 and player.cards["Wood"] >= 1 and player.cards["Wheat"] >= 1 and player.cards["Sheep"] >= 1 and player.buildings["Settlements"] >= 1 :
                for pos in player.possible_settlements:
                    player.moves.append(("Build settlement", pos))

            # Cities
            if player.cards["Wheat"] >= 2 and player.cards["Stone"] >= 3 and player.buildings["Cities"] >= 1 :
                for pos in player.settlements:
                    player.moves.append(("Build city", pos))

            # Development cards
            if player.cards["Wheat"] >= 1 and player.cards["Sheep"] >= 1 and player.cards["Stone"] >= 1 and self.board.development_cards :
                player.moves.append(("Draw card", "Development"))

            # Trading resources
            for trade_resource in player.trade_ratios : 
                if player.cards[trade_resource] >= player.trade_ratios[trade_resource]:
                    for new_resource in ["Wheat", "Wood", "Sheep", "Brick", "Stone"] : 
                        if new_resource != trade_resource : 
                            player.moves.append(("Trade", trade_resource, new_resource))

        # Play a development card
        if player.has_played_development_card == False : 
            if player.cards["Knight"] - player.new_development_cards["Knight"] >= 1 : 
                player.moves.append((("Play development", "Knight")))
            if player.cards["Road building"] - player.new_development_cards["Road building"] >= 1 : 
                player.moves.append((("Play development", "Road building")))
            if player.cards["Year of plenty"] - player.new_development_cards["Year of plenty"] >= 1 : 
                player.moves.append((("Play development", "Year of plenty")))
            if player.cards["Monopoly"] - player.new_development_cards["Monopoly"] >= 1 : 
                player.moves.append((("Play development", "Monopoly")))
            if player.cards["Victory point"] - player.new_development_cards["Victory point"] >= 1 : 
                player.moves.append((("Play development", "Victory point")))

    def generateMovesTensor(self, player) : 
        """
        Generates the moves list given the player's 
        current resources, buildings, and positions.

        :type player: Player
            The player whose moves tensor is being generated
        """
        
        player.moves_tensor = torch.zeros(246)

        for k in range(len(player.moves)) : 
            if player.moves[k] == "Roll dice" :
                player.moves_tensor[0] = 1
            
            elif player.moves[k] == "Skip turn": 
                player.moves_tensor[1] = 1

            elif player.moves[k][0] == "Build settlement" : 
                settlement_pos = player.moves[k][1]
                relative_pos = self.board.settlements_map[settlement_pos]
                
                player.moves_tensor[2+relative_pos] = 1

            elif player.moves[k][0] == "Build city" : 
                city_pos = player.moves[k][1]
                relative_pos = self.board.settlements_map[city_pos]

                player.moves_tensor[56+relative_pos] = 1

            elif player.moves[k][0] == "Build road" : 
                road_pos = player.moves[k][1]
                relative_pos = self.board.roads_map[road_pos]
                
                player.moves_tensor[110+relative_pos] = 1

            elif player.moves[k][0] == "Draw card" : 
                card = player.moves[k][1]
                if card == "Wheat" :
                    player.moves_tensor[182] = 1
                if card == "Wood" :
                    player.moves_tensor[183] = 1
                if card == "Sheep" :
                    player.moves_tensor[184] = 1
                if card == "Brick" :
                    player.moves_tensor[185] = 1
                if card == "Stone" :
                    player.moves_tensor[186] = 1
                if card == "Development" :
                    player.moves_tensor[187] = 1

            elif player.moves[k][0] == "Play development" : 
                card = player.moves[k][1]
                if card == "Knight" :
                    player.moves_tensor[188] = 1
                if card == "Road building" :
                    player.moves_tensor[189] = 1
                if card == "Year of plenty" :
                    player.moves_tensor[190] = 1
                if card == "Monopoly" :
                    player.moves_tensor[191] = 1

            elif player.moves[k][0] == "Monopoly rob" : 
                card = player.moves[k][1]
                if card == "Wheat" :
                    player.moves_tensor[192] = 1
                if card == "Wood" :
                    player.moves_tensor[193] = 1
                if card == "Sheep" :
                    player.moves_tensor[194] = 1
                if card == "Brick" :
                    player.moves_tensor[195] = 1
                if card == "Stone" :
                    player.moves_tensor[196] = 1

            elif player.moves[k][0] == "Trade" : 
                trade_resource = player.moves[k][1]
                new_resource = player.moves[k][2]

                if trade_resource == "Wheat" : 
                    if new_resource == "Wood" : 
                        player.moves_tensor[197] = 1
                    elif new_resource == "Sheep" : 
                        player.moves_tensor[198] = 1
                    elif new_resource == "Brick" : 
                        player.moves_tensor[199] = 1
                    elif new_resource == "Stone" : 
                        player.moves_tensor[200] = 1
                elif trade_resource == "Wood" : 
                    if new_resource == "Wheat" : 
                        player.moves_tensor[201] = 1
                    elif new_resource == "Sheep" : 
                        player.moves_tensor[202] = 1
                    elif new_resource == "Brick" : 
                        player.moves_tensor[203] = 1
                    elif new_resource == "Stone" : 
                        player.moves_tensor[204] = 1
                elif trade_resource == "Sheep" : 
                    if new_resource == "Wheat" : 
                        player.moves_tensor[205] = 1
                    elif new_resource == "Wood" : 
                        player.moves_tensor[206] = 1
                    elif new_resource == "Brick" : 
                        player.moves_tensor[207] = 1
                    elif new_resource == "Stone" : 
                        player.moves_tensor[208] = 1
                elif trade_resource == "Brick" : 
                    if new_resource == "Wheat" : 
                        player.moves_tensor[209] = 1
                    elif new_resource == "Wood" : 
                        player.moves_tensor[210] = 1
                    elif new_resource == "Sheep" : 
                        player.moves_tensor[211] = 1
                    elif new_resource == "Stone" : 
                        player.moves_tensor[212] = 1
                elif trade_resource == "Stone" : 
                    if new_resource == "Wheat" : 
                        player.moves_tensor[213] = 1
                    elif new_resource == "Wood" : 
                        player.moves_tensor[214] = 1
                    elif new_resource == "Sheep" : 
                        player.moves_tensor[215] = 1
                    elif new_resource == "Brick" : 
                        player.moves_tensor[216] = 1
            
            elif player.moves[k][0] == "Discard" : 
                card = player.moves[k][1]

                if card == "Wheat" : 
                    player.moves_tensor[217] = 1
                elif card == "Wood" : 
                    player.moves_tensor[218] = 1
                elif card == "Sheep" : 
                    player.moves_tensor[219] = 1
                elif card == "Brick" : 
                    player.moves_tensor[220] = 1
                elif card == "Stone" : 
                    player.moves_tensor[221] = 1

            elif player.moves[k][0] == "Move robber" :
                tile = player.moves[k][1]
                relative_pos = self.board.tiles_map[tile]

                player.moves_tensor[222+relative_pos] = 1

            elif player.moves[k][0] == "Steal card" : 
                number = player.moves[k][1]

                player.moves_tensor[241+number] = 1

    def generateInputTensor(self, player) : 
        """
        Generates the output tensor for the given player.

        :type player: Player
            Player whose output tensor is being generated
        """

        player.input_tensor = torch.zeros([256, 6])

        # Player number (0)
        player.input_tensor[0][player.number] = 1

        # Tile resources and numbers (1-38)
        for tile in self.board.tiles : 
            resource, number = self.board.tiles[tile]
            relative_pos = self.board.tiles_map[tile]

            if resource == "Wheat" : 
                player.input_tensor[1+relative_pos][0] = 1
            elif resource == "Wood" :
                player.input_tensor[1+relative_pos][1] = 1
            elif resource == "Sheep" : 
                player.input_tensor[1+relative_pos][2] = 1
            elif resource == "Brick" :
                player.input_tensor[1+relative_pos][3] = 1
            elif resource == "Stone" :
                player.input_tensor[1+relative_pos][4] = 1
            elif resource == "Desert" : 
                player.input_tensor[1+relative_pos][5] = 1

            player.input_tensor[20+relative_pos][0] = number

        # Ship resources (49-57)
        for ship in self.board.ships : 
            resource = self.board.ships[ship]
            relative_pos = self.board.ships_map[ship]

            if resource == "Wheat" : 
                player.input_tensor[49+relative_pos][0] = 1
            elif resource == "Wood" :
                player.input_tensor[49+relative_pos][1] = 1
            elif resource == "Sheep" : 
                player.input_tensor[49+relative_pos][2] = 1
            elif resource == "Brick" :
                player.input_tensor[49+relative_pos][3] = 1
            elif resource == "Stone" :
                player.input_tensor[49+relative_pos][4] = 1
            elif resource == "Anything" : 
                player.input_tensor[49+relative_pos][5] = 1

        # Settlement positions (58-111)
        for settlement in self.board.settlements : 
            case = self.board.settlements[settlement]
            relative_pos = self.board.settlements_map[settlement]

            if case == -2 : 
                player.input_tensor[58+relative_pos][0] = 1
            if case == -1 : 
                continue
            else : 
                settled_player = self.players[case]

                if settlement in settled_player.settlements :
                    player.input_tensor[58+relative_pos][(case - player.number) % len(self.players) + 1] = 1

                elif settlement in settled_player.cities : 
                    player.input_tensor[58+relative_pos][(case - player.number) % len(self.players) + 1] = 2
        
        # Road positions (112-182)
        for road in self.board.roads : 
            case = self.board.roads[road]
            relative_pos = self.board.roads_map[road]

            if case == -1: 
                player.input_tensor[112+relative_pos][0] = 1
            else : 
                player.input_tensor[112+relative_pos][(case - player.number) % len(self.players) + 1] = 1

        # Development cards deck (183)
        player.input_tensor[183][0] = len(self.board.development_cards)
        player.input_tensor[183][1] = self.board.development_cards_played["Knight"]
        player.input_tensor[183][2] = self.board.development_cards_played["Victory point"]
        player.input_tensor[183][3] = self.board.development_cards_played["Road building"]
        player.input_tensor[183][4] = self.board.development_cards_played["Year of plenty"]
        player.input_tensor[183][5] = self.board.development_cards_played["Monopoly"]

        # Development cards hands (184)
        for update_player in self.players : 
            player.input_tensor[184][(update_player.number - player.number) % 3] = update_player.numberOfDevelopmentCards()

        # Players scores (185)
        for update_player in self.players : 
            player.input_tensor[185][(update_player.number - player.number) % 3] = update_player.score()

        # Has largest army (186)
        for update_player in self.players : 
            if update_player.has_largest_army : 
                player.input_tensor[186][(update_player.number - player.number) % 3] = 1

        # Army sizes (187)
        for update_player in self.players : 
            player.input_tensor[187][(update_player.number - player.number) % 3] = update_player.army_size

        # Has longest road (188)
        for update_player in self.players : 
            if update_player.has_longest_road : 
                player.input_tensor[188][(update_player.number - player.number) % 3] = 1 

        # Road lengths (189)
        for update_player in self.players : 
            player.input_tensor[189][(update_player.number - player.number) % 3] = update_player.roadLength() 

        # Trade ratios (190–194)
        for update_player in self.players : 
            player.input_tensor[190][(update_player.number - player.number) % 3] = update_player.trade_ratios["Wheat"]
            player.input_tensor[191][(update_player.number - player.number) % 3] = update_player.trade_ratios["Wood"]
            player.input_tensor[192][(update_player.number - player.number) % 3] = update_player.trade_ratios["Sheep"]
            player.input_tensor[193][(update_player.number - player.number) % 3] = update_player.trade_ratios["Brick"]
            player.input_tensor[194][(update_player.number - player.number) % 3] = update_player.trade_ratios["Stone"]

        # Resources per roll (195–249)
        for update_player in self.players : 
            for roll in range(2, 13) : 
                a = (roll-2) * 5 
                player.input_tensor[195+a][(update_player.number - player.number) % 3] = update_player.resources_per_roll[roll].count("Wheat")
                player.input_tensor[196+a][(update_player.number - player.number) % 3] = update_player.resources_per_roll[roll].count("Wood")
                player.input_tensor[197+a][(update_player.number - player.number) % 3] = update_player.resources_per_roll[roll].count("Sheep")
                player.input_tensor[198+a][(update_player.number - player.number) % 3] = update_player.resources_per_roll[roll].count("Brick")
                player.input_tensor[199+a][(update_player.number - player.number) % 3] = update_player.resources_per_roll[roll].count("Stone")

        # Robber per roll (250)
        resource, number = self.board.tiles[self.board.robber]
        for update_player in self.players : 
            if resource != "Desert" : 
                player.input_tensor[250][(update_player.number - player.number) % 3] = len(update_player.robber_per_roll[number])
            else : 
                player.input_tensor[250][(update_player.number - player.number) % 3] = 0

        # Number of resources (251)
        player.input_tensor[251][0] = player.cards["Wheat"]
        player.input_tensor[251][1] = player.cards["Wood"]
        player.input_tensor[251][2] = player.cards["Sheep"]
        player.input_tensor[251][3] = player.cards["Brick"]
        player.input_tensor[251][4] = player.cards["Stone"]

        # Total resources (252)
        for update_player in self.players : 
            player.input_tensor[252][(update_player.number - player.number) % 3] = update_player.numberOfResources()
        
        # Buildings (253-255)
        for update_player in self.players : 
            player.input_tensor[253][(update_player.number - player.number) % 3] = update_player.buildings["Settlements"]
            player.input_tensor[254][(update_player.number - player.number) % 3] = update_player.buildings["Cities"]
            player.input_tensor[255][(update_player.number - player.number) % 3] = update_player.buildings["Roads"]

        # Generate moves tensor
        self.generateMovesTensor(player)

        # Combine tensors
        player.input_tensor = torch.reshape(player.input_tensor, (1, 1536))
        # player.input_tensor = torch.cat((player.input_tensor, torch.reshape(player.moves_tensor, (1, 246))), 1)


    ############################
    ## Main game play methods ##
    ############################

    def rollDice(self, player) : 
        """
        Rolls the dice and updates the necessary things... FINISH

        Parameters
        ----------
        player : player
            Player who is rolling the dice.
        """

        roll = randint(1, 6) + randint(1, 6)

        print("\nPlayer " + str(player.number) + ", you rolled a " + str(roll) + ".")

        if roll == 7 : 
            self.robberDiscard()
            self.placeRobber(player)
            self.updateRobberResources()
        else : 
            for update_player in self.players : 
                for resource in update_player.resources_per_roll[roll] : 
                    update_player.cards[resource] += 1

                if update_player.resources_per_roll[roll] : 
                    new_counted_resources = Counter(update_player.resources_per_roll[roll])
                    new_resource_list = [str(new_counted_resources[key]) + " " + key.lower() for key in new_counted_resources]

                    total_counted_resources = [str(update_player.cards[key]) + " " + key.lower() for key in update_player.cards if update_player.cards[key] > 0]

                    if player.is_human : 
                        print("\nPlayer " + str(update_player.number) + ", you got " + ", ".join(new_resource_list) + ". You now have " + ", ".join(total_counted_resources) + ".")
                    else : 
                        print("\nPlayer " + str(update_player.number) + ", you got " + ", ".join(new_resource_list) + ".")
            
    def makeMove(self, player) :
        """
        Determines whether the player is a human or computer and then calls
        makeHumanMove or makeBotMove, respectively.

        Returns 
        ----------
        self.moves[chosen_move_int]
            The chosen move (whatever it is).
        """

        if player.is_human :
            return self.makeHumanMove(player)
        else :
            return self.makeBotMove(player)
        
    def makeHumanMove(self, player) : 
        """
        Allows the user to make a move. It first prints
        the possible moves, then asks the user to choose 
        one move as an integer. 

        Returns 
        ----------
        self.moves[chosen_move_int]
            The chosen move (whatever it is).
        """

        # Print the player's possible moves
        print("\n" + "Player " + str(player.number) + ", choose a move:")

        for i in range(len(player.moves)) : 
                if player.moves[i] == "Skip turn" or player.moves[i] == "Roll dice": 
                    print("(" + str(i) + ")" + "\t" + player.moves[i])

                if player.moves[i][0] == "Draw card" : 
                    card = player.moves[i][1].lower()
                    print("(" + str(i) + ")" + "\t" + "Draw a " + card + " card")      

                elif player.moves[i][0] == "Build settlement" : 
                    print("(" + str(i) + ")" + "\t" + "Build a settlement at " + str(player.moves[i][1]))     

                elif player.moves[i][0] == "Build city" : 
                    print("(" + str(i) + ")" + "\t" + "Build a city at " + str(player.moves[i][1]))    

                elif player.moves[i][0] == "Build road" : 
                    start = str(list(player.moves[i][1])[0])
                    end = str(list(player.moves[i][1])[1])
                    print("(" + str(i) + ")" + "\t" + "Build a road from " + start + " to " + end)

                elif player.moves[i][0] == "Play development" : 
                    if player.moves[i][1] == "Knight" : 
                        print("(" + str(i) + ")" + "\t" + "Play knight")

                    elif player.moves[i][1] == "Road building" : 
                        print("(" + str(i) + ")" + "\t" + "Play road building")

                    elif player.moves[i][1] == "Year of plenty" : 
                        print("(" + str(i) + ")" + "\t" + "Play year of plenty")

                    elif player.moves[i][1] == "Monopoly" : 
                        print("(" + str(i) + ")" + "\t" + "Play monopoly")

                    elif player.moves[i][1] == "Victory point" : 
                        print("(" + str(i) + ")" + "\t" + "Play victory point")

                elif player.moves[i][0] == "Monopoly rob" : 
                    print("(" + str(i) + ")" + "\t" + "Steal " + player.moves[i][1].lower())

                elif player.moves[i][0] == "Trade" : 
                    trade_resource = player.moves[i][1]
                    new_resource = player.moves[i][2]
                    trade_ratio = str(player.trade_ratios[trade_resource])
                    print("(" + str(i) + ")" + "\t" + "Trade " + trade_ratio + " " + trade_resource + " for 1 " + new_resource)

                elif player.moves[i][0] == "Discard" : 
                    print("(" + str(i) + ")" + "\t" + "Discard 1 " + player.moves[i][1].lower())

                elif player.moves[i][0] == "Move robber" : 
                    tile = ", ".join([str(x) for x in player.moves[i][1]])
                    print("(" + str(i) + ")" + "\t" + "Move the robber to {" + tile + "}")  
                
                elif player.moves[i][0] == "Steal card" : 
                    print("(" + str(i) + ")" + "\t" + "Steal a card from player " + str(player.moves[i][1]))  

        # Prompt the player to make a move until it is valid.
        invalid_digit = True
        while invalid_digit : 
            chosen_move = input("\nMove number: ")

            if not chosen_move.isdigit() : 
                print("\nPlayer " + str(player.number) + ", you have not entered an integer, please try again.")
                continue 

            chosen_move_int = int(chosen_move)

            if 0 <= chosen_move_int and chosen_move_int < len(player.moves) : 
                invalid_digit = False
            else : 
                print("\nPlayer " + str(player.number) + ", you have entered an invalid move number, please try again.")

        return player.moves[chosen_move_int]
    
    def makeBotMove(self, player) :
        """
        Allows the AI to make a move...

        Returns 
        ----------
        self.moves[chosen_move_int]
            The chosen move (whatever it is).
        """

        # Let the neural network make a move
        self.generateInputTensor(player)

        best_move = None
        best_move_outcome = None
        best_move_output = None

        for i in range(246) : 
            if player.moves_tensor[i] == 1 :
                temp_moves_tensor = torch.zeros(246)
                temp_moves_tensor[i] = 1 
                temp_moves_tensor = torch.cat((player.input_tensor, torch.reshape(temp_moves_tensor, (1, 246))), 1)
                temp_moves_tensor.requires_grad_() 

                net_output = self.net(temp_moves_tensor)

                if best_move_outcome == None or net_output[0][0] < best_move_outcome : 
                    best_move = i
                    best_move_output = net_output[0][0]
                    best_move_outcome = net_output

        # Add the best_move_tensor to training data
        player.predictions.append(best_move_output)
        
        # Return the chosen move
        move_item = self.moves_dict[best_move]

        if move_item != "Skip turn" and move_item != "Roll dice" : 
            print("\n" + "\033[1m" + f"Player {player.number} has chosen {move_item}" + "\033[0m")
        else : 
            print("\n" + f"Player {player.number} has chosen {move_item}")
        
        return move_item
       
    def mainGameTurn(self, player) : 
        """
        Performs a main game move, i.e., it keeps asking for a
        new move until the "Skip turn" move is played.

        :type player: player
            Player whose turn it is.
        """

        # Graphics setting
        width = 80

        # Print updated game information
        print("\n" + "─" * width + "\n")

        turn = "UPDATED GAME INFORMATION: " + str(self.number_of_rolls) + " ROLLS"
        print(turn.center(width))

        print("Scores:")
        for print_player in self.players :
            print("Player " + str(print_player.number) + ": " + str(print_player.score()))

        print("\nRoad lengths:")
        for print_player in self.players : 
            if print_player.has_longest_road : 
                print("Player " + str(print_player.number) + ": " + str(print_player.roadLength()) + " <- Longest road")
            else : 
                print("Player " + str(print_player.number) + ": " + str(print_player.roadLength()))
        
        print("\nArmy sizes:")
        for print_player in self.players : 
            if print_player.has_largest_army : 
                print("Player " + str(print_player.number) + ": " + str(print_player.army_size) + " <- Largest army")
            else : 
                print("Player " + str(print_player.number) + ": " + str(print_player.army_size))

        # Reset the player's current move counters
        player.has_played_development_card = False
        player.has_rolled = False
        player.new_development_cards = dict.fromkeys(player.new_development_cards, 0)

        # Inform player that it is their move
        print("\n" + "─" * width + "\n")

        turn = "PLAYER " + str(player.number) + "'S TURN"
        print(turn.center(width))

        # Print the resources the player currently has 
        total_counted_resources = [str(player.cards[key]) + " " + key.lower() for key in player.cards if player.cards[key] > 0]
        
        if total_counted_resources : 
            print("\nPlayer " + str(player.number) + ", you currently have " + ", ".join(total_counted_resources) + ".")
        else : 
            print("\nPlayer " + str(player.number) + ", you currently have no cards.")

        # Let the player make moves until they no longer can
        while True : 
            self.generateMoves(player)

            move = self.makeMove(player)

            if move == "Skip turn" : 
                break

            if move == "Roll dice" : 
                self.rollDice(player)
                player.has_rolled = True

            # Since the move is not "Skip turn" or "Roll dice",
            # it must be a tuple-type move

            if move[0] == "Draw card" : 
                if move[1] == "Development" : 
                    self.drawDevelopmentCard(player)      

            elif move[0] == "Build settlement" : 
                chosen_settlement_pos = move[1]
                self.buildSettlement(player, chosen_settlement_pos, 0)

            elif move[0] == "Build city" : 
                chosen_city_pos = move[1]
                self.buildCity(player, chosen_city_pos)

            elif move[0] == "Build road" : 
                chosen_road_pos = move[1]
                self.buildRoad(player, chosen_road_pos, 0)

            elif move[0] == "Play development" : 
                player.has_played_development_card = True

                if move[1] == "Knight" : 
                    self.playKnight(player, True)

                elif move[1] == "Road building" : 
                    self.playRoadBuilding(player) 

                elif move[1] == "Year of plenty" : 
                    self.playYearOfPlenty(player)

                elif move[1] == "Monopoly" : 
                    self.playMonopoly(player)

                elif move[1] == "Victory point" : 
                    self.playVictoryPoint(player)

            elif move[0] == "Trade" : 
                trade_resource = move[1]
                new_resource = move[2]
                self.makeTrade(player, trade_resource, new_resource)

            self.longestRoad()
            self.largestArmy()


    #########################
    ## Game running method ##
    #########################

    def runGame(self) : 
        """
        Runs the game.
        """

        self.setup()
        
        curr_player_number = 0
        number_of_turns_made = 0
        while max([player.score() for player in self.players]) < 10 and number_of_turns_made < 250 :
            self.number_of_rolls += 1 

            self.mainGameTurn(self.players[curr_player_number])

            curr_player_number += 1 
            curr_player_number = (curr_player_number % len(self.players))

            number_of_turns_made += 1

        scores = [player.score() for player in self.players]
        scores.sort(reverse=True)

        # Print winner and final scores
        winner = None
        for player in self.players : 
            if player.score() >= 10 : 
                winner = player
                
        width = 80
        print("\n" + "─" * width + "\n")

        if winner != None : 
            result = "PLAYER " + str(winner.number) + " WINS"
        else : 
            result = "NO ONE WINS"
            
        print(result.center(width))

        print(f"\nNumber of turns: {number_of_turns_made}")

        print("\nFinal scores:")
        for player in self.players : 
            print("Player " + str(player.number) + ": " + str(player.score()))

        return number_of_turns_made, scores