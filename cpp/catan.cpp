#include <algorithm>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <tuple>

struct CatanNetwork : torch::nn::Module {
    CatanNetwork() {
        linear_1 = register_module("linear_1", torch::nn::Linear(1782, 500));

        linear_2_1 = register_module("linear_2_1", torch::nn::Linear(500, 500));
        linear_2_2 = register_module("linear_2_2", torch::nn::Linear(500, 500));

        linear_3_1 = register_module("linear_3_1", torch::nn::Linear(500, 500));
        linear_3_2 = register_module("linear_3_2", torch::nn::Linear(500, 500));

        linear_4_1 = register_module("linear_4_1", torch::nn::Linear(500, 500));
        linear_4_2 = register_module("linear_4_2", torch::nn::Linear(500, 500));

        linear_5_1 = register_module("linear_5_1", torch::nn::Linear(500, 500));
        linear_5_2 = register_module("linear_5_2", torch::nn::Linear(500, 500));

        linear_6 = register_module("linear_6", torch::nn::Linear(500, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = linear_1->forward(x);

        auto out = torch::nn::functional::relu(linear_2_1->forward(x));
        out = torch::nn::functional::relu(linear_2_2->forward(out));
        x = out + x;
        x = torch::nn::functional::relu(x);

        out = torch::nn::functional::relu(linear_3_1->forward(x));
        out = torch::nn::functional::relu(linear_3_2->forward(out));
        x = out + x;
        x = torch::nn::functional::relu(x);

        out = torch::nn::functional::relu(linear_4_1->forward(x));
        out = torch::nn::functional::relu(linear_4_2->forward(out));
        x = out + x;
        x = torch::nn::functional::relu(x);

        out = torch::nn::functional::relu(linear_4_1->forward(x));
        out = torch::nn::functional::relu(linear_4_2->forward(out));
        x = out + x;
        x = torch::nn::functional::relu(x);

        out = torch::nn::functional::relu(linear_5_1->forward(x));
        out = torch::nn::functional::relu(linear_5_2->forward(out));
        x = out + x;
        x = torch::nn::functional::relu(x);

        x = linear_6->forward(x);

        return x;
    }

    torch::nn::Linear
        linear_1{ nullptr },
        linear_2_1{ nullptr },
        linear_2_2{ nullptr },
        linear_3_1{ nullptr },
        linear_3_2{ nullptr },
        linear_4_1{ nullptr },
        linear_4_2{ nullptr },
        linear_5_1{ nullptr },
        linear_5_2{ nullptr },
        linear_6{ nullptr };
};

class Board {

public:
    std::map <std::string, int>
        settlements,
        settlements_map,
        development_cards_played;

    std::map <int, std::string>
        settlements_map_back;

    std::map <std::set <std::string>, std::string>
        ships;

    std::map <std::set <std::string>, std::vector <std::string>>
        tiles;

    std::map <std::set <std::string>, int>
        roads, 
        roads_map,
        tiles_map,
        ships_map;

    std::map <int, std::set <std::string>>
        roads_map_back,
        tiles_map_back;

    std::map <std::string, std::set <std::string>>
        nearby_settlements;

    std::vector <std::string>
        development_cards;

    std::set <std::string>
        possible_settlement_positions,
        robber;

    unsigned int curr_development_card_index;

    Board() {
        settlements = {
            {"(0, -2)", -1}, {"(0, 0)", -1}, {"(0, 2)", -1},
            {"(1, -3)", -1}, {"(1, -1)", -1}, {"(1, 1)", -1}, {"(1, 3)", -1},
            {"(2, -3)", -1}, {"(2, -1)", -1}, {"(2, 1)", -1}, {"(2, 3)", -1},
            {"(3, -4)", -1}, {"(3, -2)", -1}, {"(3, 0)", -1}, {"(3, 2)", -1}, {"(3, 4)", -1},
            {"(4, -4)", -1}, {"(4, -2)", -1}, {"(4, 0)", -1}, {"(4, 2)", -1}, {"(4, 4)", -1},
            {"(5, -5)", -1}, {"(5, -3)", -1}, {"(5, -1)", -1}, {"(5, 1)", -1}, {"(5, 3)", -1}, {"(5, 5)", -1},
            {"(6, -5)", -1}, {"(6, -3)", -1}, {"(6, -1)", -1}, {"(6, 1)", -1}, {"(6, 3)", -1}, {"(6, 5)", -1},
            {"(7, -4)", -1}, {"(7, -2)", -1}, {"(7, 0)", -1}, {"(7, 2)", -1}, {"(7, 4)", -1},
            {"(8, -4)", -1}, {"(8, -2)", -1}, {"(8, 0)", -1}, {"(8, 2)", -1}, {"(8, 4)", -1},
            {"(9, -3)", -1}, {"(9, -1)", -1}, {"(9, 1)", -1}, {"(9, 3)", -1},
            {"(10, -3)", -1}, {"(10, -1)", -1}, {"(10, 1)", -1}, {"(10, 3)", -1},
            {"(11, -2)", -1}, {"(11, 0)", -1}, {"(11, 2)", -1}
        };

        settlements_map = {
            {"(0, -2)", 0}, {"(0, 0)", 1}, {"(0, 2)", 2},
            {"(1, -3)", 3}, {"(1, -1)", 4}, {"(1, 1)", 5}, {"(1, 3)", 6},
            {"(2, -3)", 7}, {"(2, -1)", 8}, {"(2, 1)", 9}, {"(2, 3)", 10},
            {"(3, -4)", 11}, {"(3, -2)", 12}, {"(3, 0)", 13}, {"(3, 2)", 14}, {"(3, 4)", 15},
            {"(4, -4)", 16}, {"(4, -2)", 17}, {"(4, 0)", 18}, {"(4, 2)", 19}, {"(4, 4)", 20},
            {"(5, -5)", 21}, {"(5, -3)", 22}, {"(5, -1)", 23}, {"(5, 1)", 24}, {"(5, 3)", 25}, {"(5, 5)", 26},
            {"(6, -5)", 27}, {"(6, -3)", 28}, {"(6, -1)", 29}, {"(6, 1)", 30}, {"(6, 3)", 31}, {"(6, 5)", 32},
            {"(7, -4)", 33}, {"(7, -2)", 34}, {"(7, 0)", 35}, {"(7, 2)", 36}, {"(7, 4)", 37},
            {"(8, -4)", 38}, {"(8, -2)", 39}, {"(8, 0)", 40}, {"(8, 2)", 41}, {"(8, 4)", 42},
            {"(9, -3)", 43}, {"(9, -1)", 44}, {"(9, 1)", 45}, {"(9, 3)", 46},
            {"(10, -3)", 47}, {"(10, -1)", 48}, {"(10, 1)", 49}, {"(10, 3)", 50},
            {"(11, -2)", 51}, {"(11, 0)", 52}, {"(11, 2)", 53}
        };

        settlements_map_back = {
            {0, "(0, -2)"}, {1, "(0, 0)"}, {2, "(0, 2)"},
            {3, "(1, -3)"}, {4, "(1, -1)"}, {5, "(1, 1)"}, {6, "(1, 3)"},
            {7, "(2, -3)"}, {8, "(2, -1)"}, {9, "(2, 1)"}, {10, "(2, 3)"},
            {11, "(3, -4)"}, {12, "(3, -2)"}, {13, "(3, 0)"}, {14, "(3, 2)"}, {15, "(3, 4)"},
            {16, "(4, -4)"}, {17, "(4, -2)"}, {18, "(4, 0)"}, {19, "(4, 2)"}, {20, "(4, 4)"},
            {21, "(5, -5)"}, {22, "(5, -3)"}, {23, "(5, -1)"}, {24, "(5, 1)"}, {25, "(5, 3)"}, {26, "(5, 5)"},
            {27, "(6, -5)"}, {28, "(6, -3)"}, {29, "(6, -1)"}, {30, "(6, 1)"}, {31, "(6, 3)"}, {32, "(6, 5)"},
            {33, "(7, -4)"}, {34, "(7, -2)"}, {35, "(7, 0)"}, {36, "(7, 2)"}, {37, "(7, 4)"},
            {38, "(8, -4)"}, {39, "(8, -2)"}, {40, "(8, 0)"}, {41, "(8, 2)"}, {42, "(8, 4)"},
            {43, "(9, -3)"}, {44, "(9, -1)"}, {45, "(9, 1)"}, {46, "(9, 3)"},
            {47, "(10, -3)"}, {48, "(10, -1)"}, {49, "(10, 1)"}, {50, "(10, 3)"},
            {51, "(11, -2)"}, {52, "(11, 0)"}, {53, "(11, 2)"}
        };

        roads = {
            {{"(1, -3)", "(0, -2)"}, -1},
            {{"(0, -2)", "(1, -1)"}, -1},
            {{"(1, -1)", "(0, 0)"}, -1},
            {{"(0, 0)", "(1, 1)"}, -1},
            {{"(1, 1)", "(0, 2)"}, -1},
            {{"(0, 2)", "(1, 3)"}, -1},
            {{"(1, -3)", "(2, -3)"}, -1},
            {{"(1, -1)", "(2, -1)"}, -1},
            {{"(1, 1)", "(2, 1)"}, -1},
            {{"(1, 3)", "(2, 3)"}, -1},
            {{"(3, -4)", "(2, -3)"}, -1},
            {{"(2, -3)", "(3, -2)"}, -1},
            {{"(3, -2)", "(2, -1)"}, -1},
            {{"(2, -1)", "(3, 0)"}, -1},
            {{"(3, 0)", "(2, 1)"}, -1},
            {{"(2, 1)", "(3, 2)"}, -1},
            {{"(3, 2)", "(2, 3)"}, -1},
            {{"(2, 3)", "(3, 4)"}, -1},
            {{"(3, -4)", "(4, -4)"}, -1},
            {{"(3, -2)", "(4, -2)"}, -1},
            {{"(3, 0)", "(4, 0)"}, -1},
            {{"(3, 2)", "(4, 2)"}, -1},
            {{"(3, 4)", "(4, 4)"}, -1},
            {{"(5, -5)", "(4, -4)"}, -1},
            {{"(4, -4)", "(5, -3)"}, -1},
            {{"(5, -3)", "(4, -2)"}, -1},
            {{"(4, -2)", "(5, -1)"}, -1},
            {{"(5, -1)", "(4, 0)"}, -1},
            {{"(4, 0)", "(5, 1)"}, -1},
            {{"(5, 1)", "(4, 2)"}, -1},
            {{"(4, 2)", "(5, 3)"}, -1},
            {{"(5, 3)", "(4, 4)"}, -1},
            {{"(4, 4)", "(5, 5)"}, -1},
            {{"(5, -5)", "(6, -5)"}, -1},
            {{"(5, -3)", "(6, -3)"}, -1},
            {{"(5, -1)", "(6, -1)"}, -1},
            {{"(5, 1)", "(6, 1)"}, -1},
            {{"(5, 3)", "(6, 3)"}, -1},
            {{"(5, 5)", "(6, 5)"}, -1},
            {{"(6, -5)", "(7, -4)"}, -1},
            {{"(7, -4)", "(6, -3)"}, -1},
            {{"(6, -3)", "(7, -2)"}, -1},
            {{"(7, -2)", "(6, -1)"}, -1},
            {{"(6, -1)", "(7, 0)"}, -1},
            {{"(7, 0)", "(6, 1)"}, -1},
            {{"(6, 1)", "(7, 2)"}, -1},
            {{"(7, 2)", "(6, 3)"}, -1},
            {{"(6, 3)", "(7, 4)"}, -1},
            {{"(7, 4)", "(6, 5)"}, -1},
            {{"(7, -4)", "(8, -4)"}, -1},
            {{"(7, -2)", "(8, -2)"}, -1},
            {{"(7, 0)", "(8, 0)"}, -1},
            {{"(7, 2)", "(8, 2)"}, -1},
            {{"(7, 4)", "(8, 4)"}, -1},
            {{"(8, -4)", "(9, -3)"}, -1},
            {{"(9, -3)", "(8, -2)"}, -1},
            {{"(8, -2)", "(9, -1)"}, -1},
            {{"(9, -1)", "(8, 0)"}, -1},
            {{"(8, 0)", "(9, 1)"}, -1},
            {{"(9, 1)", "(8, 2)"}, -1},
            {{"(8, 2)", "(9, 3)"}, -1},
            {{"(9, 3)", "(8, 4)"}, -1},
            {{"(9, -3)", "(10, -3)"}, -1},
            {{"(9, -1)", "(10, -1)"}, -1},
            {{"(9, 1)", "(10, 1)"}, -1},
            {{"(9, 3)", "(10, 3)"}, -1},
            {{"(10, -3)", "(11, -2)"}, -1},
            {{"(11, -2)", "(10, -1)"}, -1},
            {{"(10, -1)", "(11, 0)"}, -1},
            {{"(11, 0)", "(10, 1)"}, -1},
            {{"(10, 1)", "(11, 2)"}, -1},
            {{"(11, 2)", "(10, 3)"}, -1}
        };

        roads_map = {
            {{"(1, -3)", "(0, -2)"}, 0},
            {{"(0, -2)", "(1, -1)"}, 1},
            {{"(1, -1)", "(0, 0)"}, 2},
            {{"(0, 0)", "(1, 1)"}, 3},
            {{"(1, 1)", "(0, 2)"}, 4},
            {{"(0, 2)", "(1, 3)"}, 5},
            {{"(1, -3)", "(2, -3)"}, 6},
            {{"(1, -1)", "(2, -1)"}, 7},
            {{"(1, 1)", "(2, 1)"}, 8},
            {{"(1, 3)", "(2, 3)"}, 9},
            {{"(3, -4)", "(2, -3)"}, 10},
            {{"(2, -3)", "(3, -2)"}, 11},
            {{"(3, -2)", "(2, -1)"}, 12},
            {{"(2, -1)", "(3, 0)"}, 13},
            {{"(3, 0)", "(2, 1)"}, 14},
            {{"(2, 1)", "(3, 2)"}, 15},
            {{"(3, 2)", "(2, 3)"}, 16},
            {{"(2, 3)", "(3, 4)"}, 17},
            {{"(3, -4)", "(4, -4)"}, 18},
            {{"(3, -2)", "(4, -2)"}, 19},
            {{"(3, 0)", "(4, 0)"}, 20},
            {{"(3, 2)", "(4, 2)"}, 21},
            {{"(3, 4)", "(4, 4)"}, 22},
            {{"(5, -5)", "(4, -4)"}, 23},
            {{"(4, -4)", "(5, -3)"}, 24},
            {{"(5, -3)", "(4, -2)"}, 25},
            {{"(4, -2)", "(5, -1)"}, 26},
            {{"(5, -1)", "(4, 0)"}, 27},
            {{"(4, 0)", "(5, 1)"}, 28},
            {{"(5, 1)", "(4, 2)"}, 29},
            {{"(4, 2)", "(5, 3)"}, 30},
            {{"(5, 3)", "(4, 4)"}, 31},
            {{"(4, 4)", "(5, 5)"}, 32},
            {{"(5, -5)", "(6, -5)"}, 33},
            {{"(5, -3)", "(6, -3)"}, 34},
            {{"(5, -1)", "(6, -1)"}, 35},
            {{"(5, 1)", "(6, 1)"}, 36},
            {{"(5, 3)", "(6, 3)"}, 37},
            {{"(5, 5)", "(6, 5)"}, 38},
            {{"(6, -5)", "(7, -4)"}, 39},
            {{"(7, -4)", "(6, -3)"}, 40},
            {{"(6, -3)", "(7, -2)"}, 41},
            {{"(7, -2)", "(6, -1)"}, 42},
            {{"(6, -1)", "(7, 0)"}, 43},
            {{"(7, 0)", "(6, 1)"}, 44},
            {{"(6, 1)", "(7, 2)"}, 45},
            {{"(7, 2)", "(6, 3)"}, 46},
            {{"(6, 3)", "(7, 4)"}, 47},
            {{"(7, 4)", "(6, 5)"}, 48},
            {{"(7, -4)", "(8, -4)"}, 49},
            {{"(7, -2)", "(8, -2)"}, 50},
            {{"(7, 0)", "(8, 0)"}, 51},
            {{"(7, 2)", "(8, 2)"}, 52},
            {{"(7, 4)", "(8, 4)"}, 53},
            {{"(8, -4)", "(9, -3)"}, 54},
            {{"(9, -3)", "(8, -2)"}, 55},
            {{"(8, -2)", "(9, -1)"}, 56},
            {{"(9, -1)", "(8, 0)"}, 57},
            {{"(8, 0)", "(9, 1)"}, 58},
            {{"(9, 1)", "(8, 2)"}, 59},
            {{"(8, 2)", "(9, 3)"}, 60},
            {{"(9, 3)", "(8, 4)"}, 61},
            {{"(9, -3)", "(10, -3)"}, 62},
            {{"(9, -1)", "(10, -1)"}, 63},
            {{"(9, 1)", "(10, 1)"}, 64},
            {{"(9, 3)", "(10, 3)"}, 65},
            {{"(10, -3)", "(11, -2)"}, 66},
            {{"(11, -2)", "(10, -1)"}, 67},
            {{"(10, -1)", "(11, 0)"}, 68},
            {{"(11, 0)", "(10, 1)"}, 69},
            {{"(10, 1)", "(11, 2)"}, 70},
            {{"(11, 2)", "(10, 3)"}, 71}
        };

        roads_map_back = {
            {0, {"(1, -3)", "(0, -2)"}},
            {1, {"(0, -2)", "(1, -1)"}},
            {2, {"(1, -1)", "(0, 0)"}},
            {3, {"(0, 0)", "(1, 1)"}},
            {4, {"(1, 1)", "(0, 2)"}},
            {5, {"(0, 2)", "(1, 3)"}},
            {6, {"(1, -3)", "(2, -3)"}},
            {7, {"(1, -1)", "(2, -1)"}},
            {8, {"(1, 1)", "(2, 1)"}},
            {9, {"(1, 3)", "(2, 3)"}},
            {10, {"(3, -4)", "(2, -3)"}},
            {11, {"(2, -3)", "(3, -2)"}},
            {12, {"(3, -2)", "(2, -1)"}},
            {13, {"(2, -1)", "(3, 0)"}},
            {14, {"(3, 0)", "(2, 1)"}},
            {15, {"(2, 1)", "(3, 2)"}},
            {16, {"(3, 2)", "(2, 3)"}},
            {17, {"(2, 3)", "(3, 4)"}},
            {18, {"(3, -4)", "(4, -4)"}},
            {19, {"(3, -2)", "(4, -2)"}},
            {20, {"(3, 0)", "(4, 0)"}},
            {21, {"(3, 2)", "(4, 2)"}},
            {22, {"(3, 4)", "(4, 4)"}},
            {23, {"(5, -5)", "(4, -4)"}},
            {24, {"(4, -4)", "(5, -3)"}},
            {25, {"(5, -3)", "(4, -2)"}},
            {26, {"(4, -2)", "(5, -1)"}},
            {27, {"(5, -1)", "(4, 0)"}},
            {28, {"(4, 0)", "(5, 1)"}},
            {29, {"(5, 1)", "(4, 2)"}},
            {30, {"(4, 2)", "(5, 3)"}},
            {31, {"(5, 3)", "(4, 4)"}},
            {32, {"(4, 4)", "(5, 5)"}},
            {33, {"(5, -5)", "(6, -5)"}},
            {34, {"(5, -3)", "(6, -3)"}},
            {35, {"(5, -1)", "(6, -1)"}},
            {36, {"(5, 1)", "(6, 1)"}},
            {37, {"(5, 3)", "(6, 3)"}},
            {38, {"(5, 5)", "(6, 5)"}},
            {39, {"(6, -5)", "(7, -4)"}},
            {40, {"(7, -4)", "(6, -3)"}},
            {41, {"(6, -3)", "(7, -2)"}},
            {42, {"(7, -2)", "(6, -1)"}},
            {43,{"(6, -1)", "(7, 0)"}},
            {44, {"(7, 0)", "(6, 1)"}},
            {45, {"(6, 1)", "(7, 2)"}},
            {46, {"(7, 2)", "(6, 3)"}},
            {47, {"(6, 3)", "(7, 4)"}},
            {48, {"(7, 4)", "(6, 5)"}},
            {49, {"(7, -4)", "(8, -4)"}},
            {50, {"(7, -2)", "(8, -2)"}},
            {51, {"(7, 0)", "(8, 0)"}},
            {52, {"(7, 2)", "(8, 2)"}},
            {53, {"(7, 4)", "(8, 4)"}},
            {54, {"(8, -4)", "(9, -3)"}},
            {55, {"(9, -3)", "(8, -2)"}},
            {56, {"(8, -2)", "(9, -1)"}},
            {57, {"(9, -1)", "(8, 0)"}},
            {58, {"(8, 0)", "(9, 1)"}},
            {59, {"(9, 1)", "(8, 2)"}},
            {60, {"(8, 2)", "(9, 3)"}},
            {61, {"(9, 3)", "(8, 4)"}},
            {62, {"(9, -3)", "(10, -3)"}},
            {63, {"(9, -1)", "(10, -1)"}},
            {64, {"(9, 1)", "(10, 1)"}},
            {65, {"(9, 3)", "(10, 3)"}},
            {66, {"(10, -3)", "(11, -2)"}},
            {67, {"(11, -2)", "(10, -1)"}},
            {68, {"(10, -1)", "(11, 0)"}},
            {69, {"(11, 0)", "(10, 1)"}},
            {70, {"(10, 1)", "(11, 2)"}},
            {71, {"(11, 2)", "(10, 3)"}}
        };
    
        tiles = {
            {{"(0, -2)", "(1, -1)", "(2, -1)", "(3, -2)", "(2, -3)", "(1, -3)"}, {"0", "0"}},
            {{"(0, 0)", "(1, 1)", "(2, 1)", "(3, 0)", "(2, -1)", "(1, -1)"}, {"0", "0"}},
            {{"(0, 2)", "(1, 3)", "(2, 3)", "(3, 2)", "(2, 1)", "(1, 1)"}, {"0", "0"}},
            {{"(2, -3)", "(3, -2)", "(4, -2)", "(5, -3)", "(4, -4)", "(3, -4)"}, {"0", "0"}},
            {{"(2, -1)", "(3, 0)", "(4, 0)", "(5, -1)", "(4, -2)", "(3, -2)"}, {"0", "0"}},
            {{"(2, 1)", "(3, 2)", "(4, 2)", "(5, 1)", "(4, 0)", "(3, 0)"}, {"0", "0"}},
            {{"(2, 3)", "(3, 4)", "(4, 4)", "(5, 3)", "(4, 2)", "(3, 2)"}, {"0", "0"}},
            {{"(4, -4)", "(5, -3)", "(6, -3)", "(7, -4)", "(6, -5)", "(5, -5)"}, {"0", "0"}},
            {{"(4, -2)", "(5, -1)", "(6, -1)", "(7, -2)", "(6, -3)", "(5, -3)"}, {"0", "0"}},
            {{"(4, 0)", "(5, 1)", "(6, 1)", "(7, 0)", "(6, -1)", "(5, -1)"}, {"0", "0"}},
            {{"(4, 2)", "(5, 3)", "(6, 3)", "(7, 2)", "(6, 1)", "(5, 1)"}, {"0", "0"}},
            {{"(4, 4)", "(5, 5)", "(6, 5)", "(7, 4)", "(6, 3)", "(5, 3)"}, {"0", "0"}},
            {{"(6, -3)", "(7, -2)", "(8, 2)", "(9, -3)", "(8, -4)", "(7, -4)"}, {"0", "0"}},
            {{"(6, -1)", "(7, 0)", "(8, 0)", "(9, -1)", "(8, -2)", "(7, -2)"}, {"0", "0"}},
            {{"(6, 1)", "(7, 2)", "(8, 2)", "(9, 1)", "(8, 0)", "(7, 0)"}, {"0", "0"}},
            {{"(6, 3)", "(7, 4)", "(8, 4)", "(9, 3)", "(8, 2)", "(7, 2)"}, {"0", "0"}},
            {{"(8, -2)", "(9, -1)", "(10, -1)", "(11, -2)", "(10, -3)", "(9, -3)"}, {"0", "0"}},
            {{"(8, 0)", "(9, 1)", "(10, 1)", "(11, 0)", "(10, -1)", "(9, -1)"}, {"0", "0"}},
            {{"(8, 2)", "(9, 3)", "(10, 3)", "(11, 2)", "(10, 1)", "(9, 1)"}, {"0", "0"}}
        };

        tiles_map = {
            {{"(0, -2)", "(1, -1)", "(2, -1)", "(3, -2)", "(2, -3)", "(1, -3)"}, 0},
            {{"(0, 0)", "(1, 1)", "(2, 1)", "(3, 0)", "(2, -1)", "(1, -1)"}, 1},
            {{"(0, 2)", "(1, 3)", "(2, 3)", "(3, 2)", "(2, 1)", "(1, 1)"}, 2},
            {{"(2, -3)", "(3, -2)", "(4, -2)", "(5, -3)", "(4, -4)", "(3, -4)"}, 3},
            {{"(2, -1)", "(3, 0)", "(4, 0)", "(5, -1)", "(4, -2)", "(3, -2)"}, 4},
            {{"(2, 1)", "(3, 2)", "(4, 2)", "(5, 1)", "(4, 0)", "(3, 0)"}, 5},
            {{"(2, 3)", "(3, 4)", "(4, 4)", "(5, 3)", "(4, 2)", "(3, 2)"}, 6},
            {{"(4, -4)", "(5, -3)", "(6, -3)", "(7, -4)", "(6, -5)", "(5, -5)"}, 7},
            {{"(4, -2)", "(5, -1)", "(6, -1)", "(7, -2)", "(6, -3)", "(5, -3)"}, 8},
            {{"(4, 0)", "(5, 1)", "(6, 1)", "(7, 0)", "(6, -1)", "(5, -1)"}, 9},
            {{"(4, 2)", "(5, 3)", "(6, 3)", "(7, 2)", "(6, 1)", "(5, 1)"}, 10},
            {{"(4, 4)", "(5, 5)", "(6, 5)", "(7, 4)", "(6, 3)", "(5, 3)"}, 11},
            {{"(6, -3)", "(7, -2)", "(8, 2)", "(9, -3)", "(8, -4)", "(7, -4)"}, 12},
            {{"(6, -1)", "(7, 0)", "(8, 0)", "(9, -1)", "(8, -2)", "(7, -2)"}, 13},
            {{"(6, 1)", "(7, 2)", "(8, 2)", "(9, 1)", "(8, 0)", "(7, 0)"}, 14},
            {{"(6, 3)", "(7, 4)", "(8, 4)", "(9, 3)", "(8, 2)", "(7, 2)"}, 15},
            {{"(8, -2)", "(9, -1)", "(10, -1)", "(11, -2)", "(10, -3)", "(9, -3)"}, 16},
            {{"(8, 0)", "(9, 1)", "(10, 1)", "(11, 0)", "(10, -1)", "(9, -1)"}, 17},
            {{"(8, 2)", "(9, 3)", "(10, 3)", "(11, 2)", "(10, 1)", "(9, 1)"}, 18}
        };

        tiles_map_back = {
            {0, {"(0, -2)", "(1, -1)", "(2, -1)", "(3, -2)", "(2, -3)", "(1, -3)"}},
            {1, {"(0, 0)", "(1, 1)", "(2, 1)", "(3, 0)", "(2, -1)", "(1, -1)"}},
            {2, {"(0, 2)", "(1, 3)", "(2, 3)", "(3, 2)", "(2, 1)", "(1, 1)"}},
            {3, {"(2, -3)", "(3, -2)", "(4, -2)", "(5, -3)", "(4, -4)", "(3, -4)"}},
            {4, {"(2, -1)", "(3, 0)", "(4, 0)", "(5, -1)", "(4, -2)", "(3, -2)"}},
            {5, {"(2, 1)", "(3, 2)", "(4, 2)", "(5, 1)", "(4, 0)", "(3, 0)"}},
            {6, {"(2, 3)", "(3, 4)", "(4, 4)", "(5, 3)", "(4, 2)", "(3, 2)"}},
            {7, {"(4, -4)", "(5, -3)", "(6, -3)", "(7, -4)", "(6, -5)", "(5, -5)"}},
            {8, {"(4, -2)", "(5, -1)", "(6, -1)", "(7, -2)", "(6, -3)", "(5, -3)"}},
            {9, {"(4, 0)", "(5, 1)", "(6, 1)", "(7, 0)", "(6, -1)", "(5, -1)"}},
            {10, {"(4, 2)", "(5, 3)", "(6, 3)", "(7, 2)", "(6, 1)", "(5, 1)"}},
            {11, {"(4, 4)", "(5, 5)", "(6, 5)", "(7, 4)", "(6, 3)", "(5, 3)"}},
            {12, {"(6, -3)", "(7, -2)", "(8, 2)", "(9, -3)", "(8, -4)", "(7, -4)"}},
            {13, {"(6, -1)", "(7, 0)", "(8, 0)", "(9, -1)", "(8, -2)", "(7, -2)"}},
            {14, {"(6, 1)", "(7, 2)", "(8, 2)", "(9, 1)", "(8, 0)", "(7, 0)"}},
            {15, {"(6, 3)", "(7, 4)", "(8, 4)", "(9, 3)", "(8, 2)", "(7, 2)"}},
            {16, {"(8, -2)", "(9, -1)", "(10, -1)", "(11, -2)", "(10, -3)", "(9, -3)"}},
            {17, {"(8, 0)", "(9, 1)", "(10, 1)", "(11, 0)", "(10, -1)", "(9, -1)"}},
            {18, {"(8, 2)", "(9, 3)", "(10, 3)", "(11, 2)", "(10, 1)", "(9, 1)"}}
        };
    
        ships = {
            {{"(0, -2)", "(1, -3)"}, "0"},
            {{"(0, 0)", "(1, 1)"}, "0"},
            {{"(2, 3)", "(3, 4)"}, "0"},
            {{"(3, -4)", "(4, -4)"}, "0"},
            {{"(5, 5)", "(6, 5)"}, "0"},
            {{"(7, -4)", "(8, -4)"}, "0"},
            {{"(8, 4)", "(9, 3)"}, "0"},
            {{"(10, -3)", "(11, -2)"}, "0"},
            {{"(11, 0)", "(10, 1)"}, "0"}
        };
        
        ships_map = {
            {{"(0, -2)", "(1, -3)"}, 0},
            {{"(0, 0)", "(1, 1)"}, 1},
            {{"(2, 3)", "(3, 4)"}, 2},
            {{"(3, -4)", "(4, -4)"}, 3},
            {{"(5, 5)", "(6, 5)"}, 4},
            {{"(7, -4)", "(8, -4)"}, 5},
            {{"(8, 4)", "(9, 3)"}, 6},
            {{"(10, -3)", "(11, -2)"}, 7},
            {{"(11, 0)", "(10, 1)"}, 8}
        };
    
        nearby_settlements = {
            {"(0, -2)", {"(1, -3)", "(1, -1)"}},
            {"(0, 0)", {"(1, -1)", "(1, 1)"}},
            {"(0, 2)", {"(1, 1)", "(1, 3)"}},
            {"(1, -3)", {"(0, -2)", "(2, -3)"}},
            {"(1, -1)", {"(0, -2)", "(0, 0)", "(2, -1)"}},
            {"(1, 1)", {"(0, 0)", "(0, 2)", "(2, 1)"}},
            {"(1, 3)", {"(0, 2)", "(2, 3)"}},
            {"(2, -3)", {"(1, -3)", "(3, -4)", "(3, -2)"}},
            {"(2, -1)", {"(1, -1)", "(3, -2)", "(3, 0)"}},
            {"(2, 1)", {"(1, 1)", "(3, 0)", "(3, 2)"}},
            {"(2, 3)", {"(1, 3)", "(3, 2)", "(3, 4)"}},
            {"(3, -4)", {"(2, -3)", "(4, -4)"}},
            {"(3, -2)", {"(2, -3)", "(2, -1)", "(4, -2)"}},
            {"(3, 0)", {"(2, -1)", "(2, 1)", "(4, 0)"}},
            {"(3, 2)", {"(2, 1)", "(2, 3)", "(4, 2)"}},
            {"(3, 4)", {"(2, 3)", "(4, 4)"}},
            {"(4, -4)", {"(3, -4)", "(5, -5)", "(5, -3)"}},
            {"(4, -2)", {"(3, -2)", "(5, -3)", "(5, -1)"}},
            {"(4, 0)", {"(3, 0)", "(5, -1)", "(5, 1)"}},
            {"(4, 2)", {"(3, 2)", "(5, 1)", "(5, 3)"}},
            {"(4, 4)", {"(3, 4)", "(5, 3)", "(5, 5)"}},
            {"(5, -5)", {"(4, -4)", "(6, -5)"}},
            {"(5, -3)", {"(4, -4)", "(4, -2)", "(6, -3)"}},
            {"(5, -1)", {"(4, -2)", "(4, 0)", "(6, -1)"}},
            {"(5, 1)", {"(4, 0)", "(4, 2)", "(6, 1)"}},
            {"(5, 3)", {"(4, 2)", "(4, 4)", "(6, 3)"}},
            {"(5, 5)", {"(4, 4)", "(6, 5)"}},
            {"(6, -5)", {"(5, -5)", "(7, -4)"}},
            {"(6, -3)", {"(5, -3)", "(7, -4)", "(7, -2)"}},
            {"(6, -1)", {"(5, -1)", "(7, -2)", "(7, 0)"}},
            {"(6, 1)", {"(5, 1)", "(7, 0)", "(7, 2)"}},
            {"(6, 3)", {"(5, 3)", "(7, 2)", "(7, 4)"}},
            {"(6, 5)", {"(5, 5)", "(7, 4)"}},
            {"(7, -4)", {"(6, -5)", "(6, -3)", "(8, -4)"}},
            {"(7, -2)", {"(6, -3)", "(6, -1)", "(8, -2)"}},
            {"(7, 0)", {"(6, -1)", "(6, 1)", "(8, 0)"}},
            {"(7, 2)", {"(6, 1)", "(6, 3)", "(8, 2)"}},
            {"(7, 4)", {"(6, 3)", "(6, 5)", "(8, 4)"}},
            {"(8, -4)", {"(7, -4)", "(9, -3)"}},
            {"(8, -2)", {"(7, -2)", "(9, -3)", "(9, -1)"}},
            {"(8, 0)", {"(7, 0)", "(9, -1)", "(9, 1)"}},
            {"(8, 2)", {"(7, 2)", "(9, 1)", "(9, 3)"}},
            {"(8, 4)", {"(7, 4)", "(9, 3)"}},
            {"(9, -3)", {"(8, -4)", "(8, -2)", "(10, -3)"}},
            {"(9, -1)", {"(8, -2)", "(8, 0)", "(10, -1)"}},
            {"(9, 1)", {"(8, 0)", "(8, 2)", "(10, 1)"}},
            {"(9, 3)", {"(8, 2)", "(8, 4)", "(10, 3)"}},
            {"(10, -3)", {"(9, -3)", "(11, -2)"}},
            {"(10, -1)", {"(9, -1)", "(11, -2)", "(11, 0)"}},
            {"(10, 1)", {"(9, 1)", "(11, 0)", "(11, 2)"}},
            {"(10, 3)", {"(9, 3)", "(11, 2)"}},
            {"(11, -2)", {"(10, -3)", "(10, -1)"}},
            {"(11, 0)", {"(10, -1)", "(10, 1)"}},
            {"(11, 2)", {"(10, 1)", "(10, 3)"}}
        };
    
        possible_settlement_positions = {
            "(0, -2)", "(0, 0)", "(0, 2)",
            "(1, -3)", "(1, -1)", "(1, 1)", "(1, 3)",
            "(2, -3)", "(2, -1)", "(2, 1)", "(2, 3)",
            "(3, -4)", "(3, -2)", "(3, 0)", "(3, 2)", "(3, 4)",
            "(4, -4)", "(4, -2)", "(4, 0)", "(4, 2)", "(4, 4)",
            "(5, -5)", "(5, -3)", "(5, -1)", "(5, 1)", "(5, 3)", "(5, 5)",
            "(6, -5)", "(6, -3)", "(6, -1)", "(6, 1)", "(6, 3)", "(6, 5)",
            "(7, -4)", "(7, -2)", "(7, 0)", "(7, 2)", "(7, 4)",
            "(8, -4)", "(8, -2)", "(8, 0)", "(8, 2)", "(8, 4)",
            "(9, -3)", "(9, -1)", "(9, 1)", "(9, 3)",
            "(10, -3)", "(10, -1)", "(10, 1)", "(10, 3)",
            "(11, -2)", "(11, 0)", "(11, 2)"
        };

        development_cards = {
            "Knight", "Knight", "Knight", "Knight", "Knight",
            "Knight", "Knight", "Knight", "Knight", "Knight",
            "Knight", "Knight", "Knight", "Knight",
            "Victory point", "Victory point", "Victory point", "Victory point", "Victory point",
            "Road building", "Road building",
            "Year of plenty", "Year of plenty",
            "Monopoly", "Monopoly""Knight", "Knight", "Knight", "Knight", "Knight",
            "Knight", "Knight", "Knight", "Knight", "Knight",
            "Knight", "Knight", "Knight", "Knight",
            "Victory point", "Victory point", "Victory point", "Victory point", "Victory point",
            "Road building", "Road building",
            "Year of plenty", "Year of plenty",
            "Monopoly", "Monopoly"
        };

        auto rd = std::random_device{};
        auto rng = std::default_random_engine{ rd() };
        std::shuffle(std::begin(development_cards), std::end(development_cards), rng);

        curr_development_card_index = 0;

        development_cards_played = {
            {"Knight", 0},
            {"Victory point", 0},
            {"Road building", 0},
            {"Year of plenty", 0},
            {"Monopoly", 0}
        };

        genResources();

    }

    void genResources() {
        std::list <std::string> possible_resources = {
            "Wood", "Sheep", "Wheat",
            "Brick", "Stone", "Brick", "Sheep",
            "Desert", "Wood", "Wheat", "Wood", "Wheat",
            "Brick", "Sheep", "Sheep", "Stone",
            "Stone", "Wheat", "Wood"
        };

        std::list <std::string> numbers = {
            "11", "12", "9",
            "4", "6", "5", "10",
            "3", "11", "4", "8",
            "8", "10", "9", "3",
            "5", "2", "6"
        };

        std::map <std::set <std::string>, std::vector <std::string>>::iterator it;

        for (it = tiles.begin(); it != tiles.end(); it++) {
            std::string resource = possible_resources.back();
            possible_resources.pop_back();

            std::string number = "0";
            if (resource == "Desert") number = numbers.back();
            numbers.pop_back();

            tiles[it->first] = {resource, number};
        }
    }

    void genShips() {
        std::list <std::string> values = {
            "Any", "Sheep", "Any",
            "Stone", "Any", "Wheat",
            "Brick", "Any", "Wood"
        };

        std::map <std::set <std::string>, std::string>::iterator ship_it;

        for (ship_it = ships.begin(); ship_it != ships.end(); ship_it++) {
            std::string value = values.back();
            values.pop_back();
            ships[ship_it->first] = value;
        }
    }

    std::string drawDevelopmentCard() {
        if (curr_development_card_index < development_cards.size()) {
            std::string card = development_cards[curr_development_card_index];
            curr_development_card_index += 1;
            return card;
        }
        else {
            return " ";
        };
    }

};

class Player {

public:
    Board board;

    unsigned int number,
        victory_card_points,
        army_size;

    bool has_largest_army,
        has_longest_road,
        has_rolled,
        has_played_development_card;

    std::map <std::string, int>
        cards,
        new_development_cards,
        buildings,
        trade_ratios;

    std::vector <int>
        moves;

    torch::Tensor
        moves_tensor,
        input_tensor;

    std::vector <torch::Tensor>
        predictions;

    std::set <std::string>
        settlements,
        possible_settlements,
        cities;

    std::set <std::set <std::string>>
        roads;

    std::set <std::set <std::string>>
        possible_roads;

    std::map <std::string, std::set <std::string>>
        roads_adjacency;

    std::map <std::string, std::vector <int>>
        resources_per_roll,
        robber_per_roll;

    Player(Board load_board, int player_number) {
        board = load_board;
        number = player_number;
        victory_card_points = 0;
        army_size = 0;
        has_largest_army = false;
        has_longest_road = false;
        has_rolled = false;
        has_played_development_card = false;

        cards = {
            {"Wheat", 0},
            {"Wood", 0},
            {"Sheep", 0},
            {"Brick", 0},
            {"Stone", 0},
            {"Knight", 0},
            {"Victory point", 0},
            {"Road building", 0},
            {"Year of plenty", 0},
            {"Monopoly", 0}
        };

        new_development_cards = {
            {"Knight", 0},
            {"Road building", 0},
            {"Year of plenty", 0},
            {"Monopoly", 0},
            {"Victory point", 0}
        };

        buildings = {
            {"Settlements", 5},
            {"Cities", 4},
            {"Roads", 15}
        };

        moves_tensor = torch::zeros(246);
        input_tensor = torch::zeros({ 256, 6 });
        predictions = {};

        trade_ratios = {
            {"Wheat", 4},
            {"Wood", 4},
            {"Sheep", 4},
            {"Brick", 4},
            {"Stone", 4}
        };

        resources_per_roll = {
            {"2", {0, 0, 0, 0, 0}},
            {"3", {0, 0, 0, 0, 0}},
            {"4", {0, 0, 0, 0, 0}},
            {"5", {0, 0, 0, 0, 0}},
            {"6", {0, 0, 0, 0, 0}},
            {"7", {0, 0, 0, 0, 0}},
            {"8", {0, 0, 0, 0, 0}},
            {"9", {0, 0, 0, 0, 0}},
            {"10", {0, 0, 0, 0, 0}},
            {"11", {0, 0, 0, 0, 0}},
            {"12", {0, 0, 0, 0, 0}}
        };

        robber_per_roll = {
            {"2", { }},
            {"3", { }},
            {"4", { }},
            {"5", { }},
            {"6", { }},
            {"7", { }},
            {"8", { }},
            {"9", { }},
            {"10", { }},
            {"11", { }},
            {"12", { }}
        };
    }

    int numberOfResources() {
        int total_resource_cards = cards["Wheat"] + cards["Wood"] + cards["Sheep"] + cards["Brick"] + cards["Stone"];

        return total_resource_cards;
    }

    int numberOfDevelopmentCards() {
        int number_of_development_cards = cards["Knight"] + cards["Victory point"] + cards["Road building"] + cards["Year of plenty"] + cards["Monopoly"];

        return number_of_development_cards;
    }

    int score() {
        int points = 0;

        points += 5 - buildings["Settlements"];
        points += 2 * (4 - buildings["Cities"]);
        points += victory_card_points;

        if (has_largest_army) points += 2;

        if (has_longest_road) points += 2;

        return points;
    }
};

class Game {

public :
    CatanNetwork model;
    Board board;
    std::vector<Player> players; // fix this error
    std::map <int, int> moves_dict; // I don't think we need this anymore

    std::random_device rd;
    std::mt19937 rng;

    Game(int number_of_players, CatanNetwork load_model) {
        model = load_model;
        board = Board();

        for (int i = 0; i < number_of_players; i++)
            players[i] = Player(board, i);
    }

    int roadLength(Player player) {
        int output = 0;

        std::map <std::string, std::set <std::string>>::iterator road_adjacency_it;

        for (road_adjacency_it = player.roads_adjacency.begin(); road_adjacency_it != player.roads_adjacency.end(); road_adjacency_it++)
            output = std::max(output, roadLengthDFS(player, road_adjacency_it->first, 0));

        return output;
    }

    int roadLengthDFS(Player player, std::string pos, int counter) {
        int output = counter;

        std::set <std::string>::iterator neighbor;

        for (neighbor = player.roads_adjacency[pos].begin(); neighbor != player.roads_adjacency[pos].end(); neighbor++) {
            bool valid_neighbor = true;

            for (unsigned int i = 0; i < players.size(); i++) {
                if (i != player.number) {
                    Player player = players[i];
                    if (player.settlements.count(*neighbor) || player.cities.count(*neighbor)) {
                        valid_neighbor = false;
                    }
                }
            }

            if (valid_neighbor) {
                player.roads_adjacency[pos].erase(*neighbor);
                player.roads_adjacency[*neighbor].erase(pos);

                output = std::max(output, roadLengthDFS(player, *neighbor, counter + 1));

                player.roads_adjacency[pos].insert(*neighbor);
                player.roads_adjacency[*neighbor].insert(pos);
            }

        }

        return output;
    }

    std::string randomDiscard(Player player) {
        std::vector <std::string> possibilities;

        const int number_of_wheat = player.cards["Wheat"];
        const int number_of_wood = player.cards["Wood"];
        const int number_of_sheep = player.cards["Sheep"];
        const int number_of_brick = player.cards["Brick"];
        const int number_of_stone = player.cards["Stone"];

        for (int j = 0; j < number_of_wheat; j++)
            possibilities.push_back("Wheat");
        for (int j = 0; j < number_of_wood; j++)
            possibilities.push_back("Wood");
        for (int j = 0; j < number_of_sheep; j++)
            possibilities.push_back("Sheep");
        for (int j = 0; j < number_of_brick; j++)
            possibilities.push_back("Brick");
        for (int j = 0; j < number_of_stone; j++)
            possibilities.push_back("Stone");

        std::shuffle(possibilities.begin(), possibilities.end(), rng);

        std::string stolen_card = possibilities.front();

        player.cards[stolen_card] += -1;

        return stolen_card;
    }

    void buildSettlement(Player player, std::string settlement_pos, int turn_type) {
        player.settlements.insert(settlement_pos);
        player.buildings["Settlements"] += -1;

        std::map <std::set <std::string>, std::vector <std::string>>::iterator tiles_it;

        for (tiles_it = board.tiles.begin(); tiles_it != board.tiles.end(); tiles_it++) {
            std::set <std::string> tile = tiles_it->first;
            std::vector <std::string> values = tiles_it->second;

            if (tile.count(settlement_pos)) {
                if (values[0] != "Desert") {
                    if (tile == board.robber) {
                        if (values[0] == "Wheat") {
                            player.robber_per_roll[values[1]][0] += 1;
                        } 
                        else if (values[0] == "Wood") {
                            player.robber_per_roll[values[1]][1] += 1;
                        }
                        else if (values[0] == "Sheep") {
                            player.robber_per_roll[values[1]][2] += 1;
                        }
                        else if (values[0] == "Brick") {
                            player.robber_per_roll[values[1]][3] += 1;
                        }
                        else if (values[0] == "Stone") {
                            player.robber_per_roll[values[1]][4] += 1;
                        }
                    }
                    else {
                        if (values[0] == "Wheat") {
                            player.resources_per_roll[values[1]][0] += 1;
                        }
                        else if (values[0] == "Wood") {
                            player.resources_per_roll[values[1]][1] += 1;
                        }
                        else if (values[0] == "Sheep") {
                            player.resources_per_roll[values[1]][2] += 1;
                        }
                        else if (values[0] == "Brick") {
                            player.resources_per_roll[values[1]][3] += 1;
                        }
                        else if (values[0] == "Stone") {
                            player.resources_per_roll[values[1]][4] += 1;
                        }
                    }
                
                    if (turn_type == 2) {
                        player.cards[values[0]] += 1;
                    }
                }
            }
        }

        board.settlements[settlement_pos] = player.number;
        board.possible_settlement_positions.erase(settlement_pos);

        std::set <std::string>::iterator nearby_settlement_it;

        for (nearby_settlement_it = board.nearby_settlements[settlement_pos].begin(); nearby_settlement_it != board.nearby_settlements[settlement_pos].begin(); nearby_settlement_it++) {
            std::string nearby = *nearby_settlement_it;

            if (board.possible_settlement_positions.count(nearby)) {
                board.settlements[nearby] = -2;
                board.possible_settlement_positions.erase(nearby);
            }
        }

        if (turn_type == 0) {
            player.cards["Wood"] += -1;
            player.cards["Brick"] += -1;
            player.cards["Wheat"] += -1;
            player.cards["Sheep"] += -1;
        }

        updatePossibleRoads();
        updatePossibleSettlements();
    }

    void buildCity(Player player, std::string city_pos) {
        player.settlements.erase(city_pos);
        player.buildings["Settlements"] += 1;
        player.cities.insert(city_pos);
        player.buildings["Cities"] += -1;

        std::map <std::set <std::string>, std::vector <std::string>>::iterator tiles_it;

        for (tiles_it = board.tiles.begin(); tiles_it != board.tiles.end(); tiles_it++) {
            std::set <std::string> tile = tiles_it->first;
            std::vector <std::string> values = tiles_it->second;

            if (tile.count(city_pos)) {
                if (values[0] != "Desert") {
                    if (tile == board.robber) {
                        if (values[0] == "Wheat") {
                            player.robber_per_roll[values[1]][0] += 1;
                        }
                        else if (values[0] == "Wood") {
                            player.robber_per_roll[values[1]][1] += 1;
                        }
                        else if (values[0] == "Sheep") {
                            player.robber_per_roll[values[1]][2] += 1;
                        }
                        else if (values[0] == "Brick") {
                            player.robber_per_roll[values[1]][3] += 1;
                        }
                        else if (values[0] == "Stone") {
                            player.robber_per_roll[values[1]][4] += 1;
                        }
                    }
                    else {
                        if (values[0] == "Wheat") {
                            player.resources_per_roll[values[1]][0] += 1;
                        }
                        else if (values[0] == "Wood") {
                            player.resources_per_roll[values[1]][1] += 1;
                        }
                        else if (values[0] == "Sheep") {
                            player.resources_per_roll[values[1]][2] += 1;
                        }
                        else if (values[0] == "Brick") {
                            player.resources_per_roll[values[1]][3] += 1;
                        }
                        else if (values[0] == "Stone") {
                            player.resources_per_roll[values[1]][4] += 1;
                        }
                    }
                }
            }
        }

        player.cards["Wheat"] += -2;
        player.cards["Stone"] += -3;
    }

    void buildRoad(Player player, std::set <std::string> road_pos, int turn_type) {
        board.roads[road_pos] = player.number;

        if (turn_type == 0) {
            player.cards["Wood"] += -1;
            player.cards["Brick"] += -1;
        }

        player.roads.insert(road_pos);

        std::set <std::string>::iterator road_pos_it;

        road_pos_it = road_pos.begin();
        std::string start = *road_pos_it;
        road_pos_it++;
        std::string end = *road_pos_it;

        if (player.roads_adjacency.count(start)) {
            player.roads_adjacency[start].insert(end);
        }
        else {
            player.roads_adjacency[start] = { end };
        }

        if (player.roads_adjacency.count(end)) {
            player.roads_adjacency[end].insert(start);
        }
        else {
            player.roads_adjacency[end] = { start };
        }

        updatePossibleRoads();
        updatePossibleSettlements();
    }

    void updatePossibleSettlements() {
        std::vector <Player>::iterator players_it;

        for (players_it = players.begin(); players_it != players.end(); players_it++) {
            Player update_player = *players_it;

            std::set <std::string>::iterator possible_settlement_it;

            for (possible_settlement_it = board.possible_settlement_positions.begin(); possible_settlement_it != board.possible_settlement_positions.end(); possible_settlement_it++) {
                std::string possible_settlement = *possible_settlement_it;

                std::set <std::set <std::string>>::iterator road_it;
                
                for (road_it = update_player.roads.begin(); road_it != update_player.roads.end(); road_it++) {
                    std::set <std::string> road = *road_it;

                    if (road.count(possible_settlement)) {
                        update_player.possible_settlements.insert(possible_settlement);
                    }
                }
            }
        }
    }

    void updatePossibleRoads() {
        for (unsigned int i = 0; i < players.size(); i++) {
            Player player = players[i];
            player.possible_roads.clear();

            std::map <std::set <std::string>, int>::iterator road_it;

            for (road_it = board.roads.begin(); road_it != board.roads.end(); road_it++) {
                std::set <std::string> possible_road = road_it->first;

                std::set <std::string>::iterator possible_road_it;

                possible_road_it = possible_road.begin();
                std::string possible_road_start = *possible_road_it;
                possible_road_it++;
                std::string possible_road_end = *possible_road_it;

                if (board.roads[possible_road] == -1) {
                    std::set <std::set <std::string>>::iterator road_it_2;

                    for (road_it_2 = player.roads.begin(); road_it_2 != player.roads.end(); road_it_2++) {
                        std::set <std::string> road = *road_it_2;

                        possible_road_it = road.begin();
                        std::string road_start = *possible_road_it;
                        possible_road_it++;
                        std::string road_end = *possible_road_it;

                        if (road_start == possible_road_start || road_start == possible_road_end || road_end == possible_road_start || road_end == possible_road_end) {
                            player.possible_roads.insert(possible_road);
                        }
                    }
                }
            }
        }
    }

    void robberDiscard() {
        std::vector <Player>::iterator player_it;

        for (player_it = players.begin(); player_it != players.end(); player_it++) {
            Player update_player = *player_it;

            int num_resources = update_player.numberOfResources();

            if (num_resources >= 8) {
                int discard_number = num_resources / 2;

                for (int i = 0; i < discard_number; i++) {
                    update_player.moves.clear();

                    if (update_player.cards["Wheat"] > 0)
                        update_player.moves_tensor[217] = 1;
                    if (update_player.cards["Wood"] > 0)
                        update_player.moves_tensor[218] = 1;
                    if (update_player.cards["Sheep"] > 0)
                        update_player.moves_tensor[219] = 1;
                    if (update_player.cards["Brick"] > 0)
                        update_player.moves_tensor[220] = 1;
                    if (update_player.cards["Stone"] > 0)
                        update_player.moves_tensor[221] = 1;

                    int move = makeMove(update_player);

                    if (move == 217)
                        update_player.cards["Wheat"] += -1;
                    else if (move == 218)
                        update_player.cards["Wood"] += -1;
                    else if (move == 219)
                        update_player.cards["Sheep"] += -1;
                    else if (move == 220)
                        update_player.cards["Brick"] += -1;
                    else if (move == 221)
                        update_player.cards["Stone"] += -1;
                }
            }
        }
    }

    void placeRobber(Player player) {
        std::map <std::set <std::string>, std::vector <std::string>>::iterator tile_it;

        for (tile_it = board.tiles.begin(); tile_it != board.tiles.end(); tile_it++) {
            std::set <std::string> tile = tile_it->first;

            if (tile != board.robber) {
                int relative_pos = board.tiles_map[tile];

                player.moves_tensor[222 + relative_pos] = 1;
            }
        }

        int move = makeMove(player);

        board.robber = board.tiles_map_back[move - 222];
     
        std::vector<int> possible_players; 

        std::vector<Player>::iterator player_robbed_it;
       
        for (player_robbed_it = players.begin(); player_robbed_it != players.end(); player_robbed_it++) {
            Player player_robbed = *player_robbed_it;
            
            if (player_robbed.number != player.number && player_robbed.numberOfResources() > 0) {
                std::set <std::string>::iterator settlement_it;

                
                for (settlement_it = player_robbed.settlements.begin(); settlement_it != player_robbed.settlements.end(); settlement_it++) {
                    std::string settlement = *settlement_it;

                    if (board.robber.count(settlement))
                        possible_players.push_back(player_robbed.number); 
                }

                for (settlement_it = player_robbed.cities.begin(); settlement_it != player_robbed.cities.end(); settlement_it++) {
                    std::string city = *settlement_it;

                    if (board.robber.count(city))
                        possible_players.push_back(player_robbed.number);
                } 
            }
        }
        
        if (possible_players.size() > 0) {
            for (unsigned int i = 0; i < possible_players.size(); i++) {
                Player possible_player = players[possible_players[i]];
                player.moves_tensor[241 + possible_player.number] = 1;
            }

            int move = makeMove(player);
            Player player_robbed = players[move - 241];
            std::string resource = randomDiscard(player_robbed);
            player.cards[resource] += 1;
        }
    }

    void updateRobberResources() {
        std::vector <Player>::iterator update_player_it;

        for (update_player_it = players.begin(); update_player_it != players.end(); update_player_it++) {
            Player update_player = *update_player_it;

            for (unsigned int i = 2; i <= 12; i++) {
                std::string roll = std::to_string(i);

                for (int j = 0; j < 5; j++) {
                    update_player.resources_per_roll[roll][j] += update_player.robber_per_roll[roll][j];
                    update_player.robber_per_roll[roll][j] = 0;
                }
            }

            std::string resource = board.tiles[board.robber][0];
            std::string roll = board.tiles[board.robber][1];

            if (resource != "Desert") {
                std::set <std::string>::iterator settlement_it;

                for (settlement_it = update_player.settlements.begin(); settlement_it != update_player.settlements.end(); settlement_it++) {
                    std::string settlement = *settlement_it;

                    if (board.robber.count(settlement)) {
                        if (resource == "Wheat")
                            update_player.resources_per_roll[roll][0] += -1;
                        else if (resource == "Wood")
                            update_player.resources_per_roll[roll][1] += -1;
                        else if (resource == "Sheep")
                            update_player.resources_per_roll[roll][2] += -1;
                        else if (resource == "Brick")
                            update_player.resources_per_roll[roll][3] += -1;
                        else if (resource == "Stone")
                            update_player.resources_per_roll[roll][4] += -1;
                    }
                }

                for (settlement_it = update_player.cities.begin(); settlement_it != update_player.cities.end(); settlement_it++) {
                    std::string city = *settlement_it;

                    if (board.robber.count(city)) {
                        if (resource == "Wheat")
                            update_player.resources_per_roll[roll][0] += -2;
                        else if (resource == "Wood")
                            update_player.resources_per_roll[roll][1] += -2;
                        else if (resource == "Sheep")
                            update_player.resources_per_roll[roll][2] += -2;
                        else if (resource == "Brick")
                            update_player.resources_per_roll[roll][3] += -2;
                        else if (resource == "Stone")
                            update_player.resources_per_roll[roll][4] += -2;
                    }
                }
            }
        }
    }

    void drawDevelopmentCard(Player player) {
        std::string card = board.drawDevelopmentCard();

        player.cards["Sheep"] += -1;
        player.cards["Wheat"] += -1;
        player.cards["Stone"] += -1;

        if (card == "Victory point") {
            player.victory_card_points += 1;
        }
        else {
            player.cards[card] += 1;
            player.new_development_cards[card] += 1;
        }
    }

    void playKnight(Player player, bool is_development_card) {
        placeRobber(player);
        updateRobberResources();

        if (is_development_card == true) {
            player.cards["Knight"] += -1;
            player.army_size += 1;
        }
    }

    void playRoadBuilding(Player player) {
        for (int i = 0; i < std::min(2, player.buildings["Roads"]); i++) {
            std::set <std::set <std::string>>::iterator possible_roads_it;

            bool possible_road = false;

            for (possible_roads_it = player.possible_roads.begin(); possible_roads_it != player.possible_roads.end(); possible_roads_it++) {
                int relative_pos = board.roads_map[*possible_roads_it];
                player.moves_tensor[110 + relative_pos] = 1;
                possible_road = true;
            }

            if (possible_road == true) {
                int move = makeMove(player);
                std::set <std::string> pos = board.roads_map_back[move - 110];
                buildRoad(player, pos, 0);
            }

            updatePossibleRoads();
            updatePossibleSettlements();
        }

        player.cards["Road building"] += -1;
    }
    
    void playYearOfPlenty(Player player) {
        for (unsigned int i = 0; i < 2; i++) {
            player.moves_tensor[182] = 1;
            player.moves_tensor[183] = 1;
            player.moves_tensor[184] = 1;
            player.moves_tensor[185] = 1;
            player.moves_tensor[186] = 1;

            int move = makeMove(player);

            if (move == 182)
                player.cards["Wheat"] += 1;
            else if (move == 183)
                player.cards["Wood"] += 1;
            else if (move == 184)
                player.cards["Sheep"] += 1;
            else if (move == 185)
                player.cards["Brick"] += 1;
            else if (move == 186)
                player.cards["Stone"] += 1;
        }

        player.cards["Year of plenty"] += -1;
    }

    void playMonopoly(Player player) {
        player.moves_tensor[192] = 1;
        player.moves_tensor[193] = 1;
        player.moves_tensor[194] = 1;
        player.moves_tensor[195] = 1;
        player.moves_tensor[196] = 1;

        int move = makeMove(player);

        std::string chosen_resource;

        if (move == 192)
            chosen_resource = "Wheat";
        else if (move == 193)
            chosen_resource = "Wood";
        else if (move == 194)
            chosen_resource = "Sheep";
        else if (move == 195)
            chosen_resource = "Brick";
        else if (move == 196)
            chosen_resource = "Stone";

        std::vector <Player>::iterator robbed_player_it;

        for (robbed_player_it = players.begin(); robbed_player_it != players.end(); robbed_player_it++) {
            Player robbed_player = *robbed_player_it;

            if (robbed_player.number != player.number) {
                player.cards[chosen_resource] += robbed_player.cards[chosen_resource];
                robbed_player.cards[chosen_resource] = 0;
            }
        }

        player.cards["Monopoly"] += -1;
    }

    void largestArmy() {
        unsigned int largest_army = 2;
        Player largest_army_player = players[0];
        bool exists_largest_army = false;

        std::vector <Player>::iterator player_it;

        for (player_it = players.begin(); player_it != players.end(); player_it++) {
            Player player = *player_it;

            if (player.has_largest_army) {
                largest_army = player.army_size;
                largest_army_player = player;
                exists_largest_army = true;
            }
        }

        for (player_it = players.begin(); player_it != players.end(); player_it++) {
            Player player = *player_it;

            if (player.army_size > largest_army) {
                if (exists_largest_army == true) 
                    largest_army_player.has_largest_army = false;

                player.has_largest_army = true;
                largest_army = player.army_size;
                largest_army_player = player;
                exists_largest_army = true;
            }
        }
    }

    void longestRoad() {
        int longest_road = 4;
        Player longest_road_player = players[0];
        bool exists_longest_road = false;

        std::vector <Player>::iterator player_it;

        for (player_it = players.begin(); player_it != players.end(); player_it++) {
            Player player = *player_it;

            if (player.has_largest_army) {
                longest_road = roadLength(player);
                longest_road_player = player;
                exists_longest_road = true;
            }
        }

        for (player_it = players.begin(); player_it != players.end(); player_it++) {
            Player player = *player_it;
            int player_road_length = roadLength(player);

            if (player_road_length > longest_road) {
                if (exists_longest_road == true)
                    longest_road_player.has_longest_road = false;

                player.has_longest_road = true;
                longest_road = player.army_size;
                longest_road_player = player;
                exists_longest_road = true;
            }
        }
    }

    void makeTrade(Player player, std::string trade_resource, std::string new_resource) {
        int trade_ratio = player.trade_ratios[trade_resource];
        player.cards[trade_resource] += -trade_ratio;
        player.cards[new_resource] += 1;
    }

    void startMove(Player player, int setup_move_number) {
        std::set <std::string>::iterator possible_settlement_it;

        for (possible_settlement_it = board.possible_settlement_positions.begin(); possible_settlement_it != board.possible_settlement_positions.end(); possible_settlement_it++) {
            int relative_pos = board.settlements_map[*possible_settlement_it];
            player.moves_tensor[2 + relative_pos] = 1;
        }

        int move = makeMove(player);

        std::string chosen_settlement = board.settlements_map_back[move - 2];

        buildSettlement(player, chosen_settlement, setup_move_number);


        std::set <std::string>::iterator nearby_settlement_it;

        for (nearby_settlement_it = board.nearby_settlements[chosen_settlement].begin(); nearby_settlement_it != board.nearby_settlements[chosen_settlement].end(); nearby_settlement_it++) {
            std::string nearby_settlement = *nearby_settlement_it;
            std::set <std::string> possible_road = { chosen_settlement, nearby_settlement };
            int relative_pos = board.roads_map[possible_road];
            player.moves_tensor[110 + relative_pos] = 1;
        }

        move = makeMove(player);

        std::set <std::string> chosen_road = board.roads_map_back[move - 110];

        buildRoad(player, chosen_road, 1);
    }

    void setup() {
        std::vector <Player>::iterator player_it;

        for (player_it = players.begin(); player_it != players.end(); player_it++) {
            Player player = *player_it;
            startMove(player, 1);
        }

        for (player_it = players.end(); player_it != players.begin(); player_it--) {
            Player player = *player_it;
            startMove(player, 1);
        }
    }

    void generateMovesTensor(Player player) {
        player.moves_tensor = torch::zeros(246);

        if (player.has_rolled == false) {
            player.moves_tensor[0] = 1; // roll dice
        }
        else {
            player.moves_tensor[1] = 1; // skip turn

            // build settlement
            if (player.cards["Brick"] >= 1 && player.cards["Wood"] >= 1 && player.cards["Wheat"] >= 1 && player.cards["Sheep"] >= 1 && player.buildings["Settlements"] >= 1) {
                std::set <std::string>::iterator possible_settlement_it;

                for (possible_settlement_it = player.possible_settlements.begin(); possible_settlement_it != player.possible_settlements.end(); possible_settlement_it++) {
                    int relative_pos = board.settlements_map[*possible_settlement_it];
                    player.moves_tensor[2 + relative_pos] = 1;
                }
            }

            // build city 
            if (player.cards["Wheat"] >= 2 && player.cards["Stone"] >= 3 && player.buildings["Cities"] >= 1) {
                std::set <std::string>::iterator possible_cities_it;

                for (possible_cities_it = player.settlements.begin(); possible_cities_it != player.settlements.end(); possible_cities_it++) {
                    int relative_pos = board.settlements_map[*possible_cities_it];
                    player.moves_tensor[56 + relative_pos] = 1;
                }
            }

            // build road
            if (player.cards["Brick"] >= 1 && player.cards["Wood"] >= 1 && player.buildings["Roads"] >= 1) {
                std::set <std::set <std::string>>::iterator possible_road_it;

                for (possible_road_it = player.possible_roads.begin(); possible_road_it != player.possible_roads.end(); possible_road_it++) {
                    int relative_pos = board.roads_map[*possible_road_it];
                    player.moves_tensor[110 + relative_pos] = 1;
                }
            }

            // draw development card
            if (player.cards["Wheat"] >= 1 && player.cards["Sheep"] >= 1 && player.cards["Stone"] >= 1 && board.development_cards.size() > 0) {
                player.moves_tensor[187] = 1;
            }

            // play development card 
            if (player.has_played_development_card == false) {
                if (player.cards["Knight"] - player.new_development_cards["Knight"] >= 1) {
                    player.moves_tensor[188] = 1;
                }
                if (player.cards["Road building"] - player.new_development_cards["Road building"] >= 1) {
                    player.moves_tensor[189] = 1;
                }
                if (player.cards["Year of plenty"] - player.new_development_cards["Year of plenty"] >= 1) {
                    player.moves_tensor[190] = 1;
                }
                if (player.cards["Monopoly"] - player.new_development_cards["Monopoly"] >= 1) {
                    player.moves_tensor[191] = 1;
                }
            }

            // trade 
            if (player.cards["Wheat"] >= player.trade_ratios["Wheat"]) {
                player.moves_tensor[197] = 1;
                player.moves_tensor[198] = 1;
                player.moves_tensor[199] = 1;
                player.moves_tensor[200] = 1;
            }
            if (player.cards["Wood"] >= player.trade_ratios["Wood"]) {
                player.moves_tensor[201] = 1;
                player.moves_tensor[202] = 1;
                player.moves_tensor[203] = 1;
                player.moves_tensor[204] = 1;
            }
            if (player.cards["Sheep"] >= player.trade_ratios["Sheep"]) {
                player.moves_tensor[205] = 1;
                player.moves_tensor[206] = 1;
                player.moves_tensor[207] = 1;
                player.moves_tensor[208] = 1;
            }
            if (player.cards["Brick"] >= player.trade_ratios["Brick"]) {
                player.moves_tensor[209] = 1;
                player.moves_tensor[210] = 1;
                player.moves_tensor[211] = 1;
                player.moves_tensor[212] = 1;
            }
            if (player.cards["Stone"] >= player.trade_ratios["Stone"]) {
                player.moves_tensor[213] = 1;
                player.moves_tensor[214] = 1;
                player.moves_tensor[215] = 1;
                player.moves_tensor[216] = 1;
            }
        }
    }

    void generateInputTensor(Player player) {
        player.input_tensor[0][player.number] = 1;

        std::map <std::set <std::string>, std::vector <std::string>>::iterator tiles_it;

        for (tiles_it = board.tiles.begin(); tiles_it != board.tiles.end(); tiles_it++) {
            std::vector<std::string> values = tiles_it->second;
            int relative_pos = board.tiles_map[tiles_it->first];

            if (values[0] == "Wheat") {
                player.input_tensor[1 + relative_pos][0] = 1;
            }
            else if (values[0] == "Wood") {
                player.input_tensor[1 + relative_pos][1] = 1;
            }
            else if (values[0] == "Sheep") {
                player.input_tensor[1 + relative_pos][2] = 1;
            }
            else if (values[0] == "Brick") {
                player.input_tensor[1 + relative_pos][3] = 1;
            }
            else if (values[0] == "Stone") {
                player.input_tensor[1 + relative_pos][4] = 1;
            }
            else if (values[0] == "Desert") {
                player.input_tensor[1 + relative_pos][5] = 1;
            }

            player.input_tensor[20 + relative_pos][0] = std::stoi(values[1]);
        }

        std::map <std::set <std::string>, std::string>::iterator ship_it;

        for (ship_it = board.ships.begin(); ship_it != board.ships.end(); ship_it++) {
            std::string resource = ship_it->second;
            int relative_pos = board.ships_map[ship_it->first];

            if (resource == "Wheat") {
                player.input_tensor[49 + relative_pos][0] = 1;
            }
            else if (resource == "Wood") {
                player.input_tensor[49 + relative_pos][1] = 1;
            }
            else if (resource == "Sheep") {
                player.input_tensor[49 + relative_pos][2] = 1;
            }
            else if (resource == "Brick") {
                player.input_tensor[49 + relative_pos][3] = 1;
            }
            else if (resource == "Stone") {
                player.input_tensor[49 + relative_pos][4] = 1;
            }
            else if (resource == "Anything") {
                player.input_tensor[49 + relative_pos][5] = 1;
            }
        }

        std::map <std::string, int>::iterator settlements_it;

        for (settlements_it = board.settlements.begin(); settlements_it != board.settlements.end(); settlements_it++) {
            std::string settlement = settlements_it->first;
            int settlement_case = settlements_it->second;
            int relative_pos = board.settlements_map[settlement];

            if (settlement_case == -2) {
                player.input_tensor[58 + relative_pos][0] = 1;
            }
            else if (settlement_case != -1) {
                Player settled_player = players[settlement_case];

                if (settled_player.settlements.count(settlement)) {
                    player.input_tensor[58 + relative_pos][(settlement_case - player.number) % players.size() + 1] = 1;
                }
                else if (settled_player.settlements.count(settlement)) {
                    player.input_tensor[58 + relative_pos][(settlement_case - player.number) % players.size() + 1] = 2;
                }
            }
        }

        std::map <std::set <std::string>, int>::iterator roads_it;

        for (roads_it = board.roads.begin(); roads_it != board.roads.end(); roads_it++) {
            std::set <std::string> road = roads_it->first;
            int road_case = roads_it->second;
            int relative_pos = board.roads_map[road];

            if (road_case == -1) {
                player.input_tensor[112 + relative_pos][0] = 1;
            }
            else {
                player.input_tensor[112 + relative_pos][(road_case - player.number) % players.size() + 1] = 1;
            }
        }

        player.input_tensor[183][0] = board.development_cards.size();
        player.input_tensor[183][1] = board.development_cards_played["Knight"];
        player.input_tensor[183][2] = board.development_cards_played["Victory point"];
        player.input_tensor[183][3] = board.development_cards_played["Road building"];
        player.input_tensor[183][4] = board.development_cards_played["Year of plenty"];
        player.input_tensor[183][5] = board.development_cards_played["Monopoly"];

        std::vector <Player>::iterator update_player_it;

        for (update_player_it = players.begin(); update_player_it != players.end(); update_player_it++) {
            Player update_player = *update_player_it;

            player.input_tensor[184][(update_player.number - player.number) % players.size()] = update_player.numberOfDevelopmentCards();
            player.input_tensor[185][(update_player.number - player.number) % players.size()] = update_player.score();
            
            if (update_player.has_largest_army)
                player.input_tensor[186][(update_player.number - player.number) % players.size()] = 1;
        
            player.input_tensor[187][(update_player.number - player.number) % players.size()] = update_player.army_size;

            if (update_player.has_longest_road)
                player.input_tensor[188][(update_player.number - player.number) % players.size()] = 1;

            player.input_tensor[189][(update_player.number - player.number) % players.size()] = roadLength(update_player);
        
            player.input_tensor[190][(update_player.number - player.number) % players.size()] = update_player.trade_ratios["Wheat"];
            player.input_tensor[191][(update_player.number - player.number) % players.size()] = update_player.trade_ratios["Wood"];
            player.input_tensor[192][(update_player.number - player.number) % players.size()] = update_player.trade_ratios["Sheep"];
            player.input_tensor[193][(update_player.number - player.number) % players.size()] = update_player.trade_ratios["Brick"];
            player.input_tensor[194][(update_player.number - player.number) % players.size()] = update_player.trade_ratios["Stone"];
        
            for (int roll = 2; roll <= 12; roll++) {
                int a = (roll - 2) * 5;

                player.input_tensor[195 + a][(update_player.number - player.number) % players.size()] = update_player.resources_per_roll[std::to_string(roll)][0];
                player.input_tensor[196 + a][(update_player.number - player.number) % players.size()] = update_player.resources_per_roll[std::to_string(roll)][1];
                player.input_tensor[197 + a][(update_player.number - player.number) % players.size()] = update_player.resources_per_roll[std::to_string(roll)][2];
                player.input_tensor[198 + a][(update_player.number - player.number) % players.size()] = update_player.resources_per_roll[std::to_string(roll)][3];
                player.input_tensor[199 + a][(update_player.number - player.number) % players.size()] = update_player.resources_per_roll[std::to_string(roll)][4];
            }

            std::vector <std::string> robber_values = board.tiles[board.robber];

            if (robber_values[0] != "Desert") {
                int robbed_per_roll = 0;
                robbed_per_roll += player.robber_per_roll[robber_values[1]][0];
                robbed_per_roll += player.robber_per_roll[robber_values[1]][1];
                robbed_per_roll += player.robber_per_roll[robber_values[1]][2];
                robbed_per_roll += player.robber_per_roll[robber_values[1]][3];
                robbed_per_roll += player.robber_per_roll[robber_values[1]][4];

                player.input_tensor[250][(update_player.number - player.number) % players.size()] = robbed_per_roll;
            }
            else {
                player.input_tensor[250][(update_player.number - player.number) % players.size()] = 0;
            }

            player.input_tensor[252][(update_player.number - player.number) % players.size()] = update_player.numberOfResources();

            player.input_tensor[253][(update_player.number - player.number) % players.size()] = update_player.buildings["Settlements"];
            player.input_tensor[254][(update_player.number - player.number) % players.size()] = update_player.buildings["Cities"];
            player.input_tensor[255][(update_player.number - player.number) % players.size()] = update_player.buildings["Roads"];
        }

        player.input_tensor[251][0] = player.cards["Wheat"];
        player.input_tensor[251][1] = player.cards["Wood"];
        player.input_tensor[251][2] = player.cards["Sheep"];
        player.input_tensor[251][3] = player.cards["Brick"];
        player.input_tensor[251][4] = player.cards["Stone"];


    }

    void rollDice(Player player) {
        std::uniform_int_distribution<> dist6(1, 6);

        std::string roll = std::to_string(dist6(rng) + dist6(rng));

        if (roll == "7") {
            robberDiscard();
            placeRobber(player);
            updateRobberResources();
        }
        else {
            std::vector <Player>::iterator update_player_it;

            for (update_player_it = players.begin(); update_player_it != players.end(); update_player_it++) {
                Player update_player = *update_player_it;

                update_player.cards["Wheat"] += update_player.resources_per_roll[roll][0];
                update_player.cards["Wood"] += update_player.resources_per_roll[roll][1];
                update_player.cards["Sheep"] += update_player.resources_per_roll[roll][2];
                update_player.cards["Brick"] += update_player.resources_per_roll[roll][3];
                update_player.cards["Stone"] += update_player.resources_per_roll[roll][4];
            }
        }
    }

    int makeMove(Player player) {
        generateInputTensor(player);

        int best_move = -1;
        torch::Tensor best_move_output;
        bool best_move_output_init = false;

        for (unsigned int i = 0; i < 246; i++) {
            if (player.moves_tensor[i].equal(torch::ones(1))) {
                torch::Tensor temp_moves_tensor = torch::zeros(246);
                temp_moves_tensor[i] = 1;
                temp_moves_tensor = torch::cat({ player.input_tensor, torch::reshape(temp_moves_tensor, {1, 246}) }, 1);
                temp_moves_tensor.requires_grad_();

                torch::Tensor net_output = model.forward(temp_moves_tensor);
                net_output = net_output[0][0];

                if (best_move_output_init == false || net_output.item<float>() < best_move_output.item<float>()) {
                    best_move = i;
                    best_move_output = net_output;
                }
            }
        }

        player.predictions.push_back(best_move_output);

        player.moves_tensor = torch::zeros(246);
        player.input_tensor = torch::zeros({ 256, 6 });
          
        return best_move;
    }

    void mainGameTurn(Player player) {
        player.has_played_development_card = false;
        player.has_rolled = false;
        player.new_development_cards.clear();

        bool chose_skip = false;
        while (chose_skip == true) {
            generateMovesTensor(player);
            int move = makeMove(player);

            if (move == 0) { // skip turn
                chose_skip = false;
            }
            else if (move == 1) {  // roll dice
                rollDice(player);
                player.has_rolled = true;
            }
            else if (2 <= move && move <= 55) { // build settlement
                std::string chosen_settlement_pos = board.settlements_map_back[move - 2];
                buildSettlement(player, chosen_settlement_pos, 0);
            }
            else if (56 <= move && move <= 109) { // build city
                std::string chosen_city_pos = board.settlements_map_back[move - 56];
                buildCity(player, chosen_city_pos);
            }
            else if (110 <= move && move <= 181) { // build road
                std::set <std::string> chosen_road_pos = board.roads_map_back[move - 110];
                buildRoad(player, chosen_road_pos, 0);
            }
            else if (move == 187) { // draw development card
                drawDevelopmentCard(player);
            }
            else if (move == 188) { // play knight
                playKnight(player, true);
            }
            else if (move == 189) { // play road building
                playRoadBuilding(player);
            }
            else if (move == 190) { // play year of plenty
                playYearOfPlenty(player);
            }
            else if (move == 191) { // play monopoly
                playMonopoly(player);
            }
            else if (move == 197) { // trade wheat for wood
                makeTrade(player, "Wheat", "Wood");
            }
            else if (move == 198) { // trade wheat for sheep
                makeTrade(player, "Wheat", "Sheep");
            }
            else if (move == 199) { // trade wheat for brick
                makeTrade(player, "Wheat", "Brick");
            }
            else if (move == 200) { // trade wheat for stone
                makeTrade(player, "Wheat", "Stone");
            }
            else if (move == 201) { // trade wood for wheat
                makeTrade(player, "Wood", "Wheat");
            }
            else if (move == 202) { // trade wood for sheep
                makeTrade(player, "Wood", "Sheep");
            }
            else if (move == 203) { // trade wood for brick
                makeTrade(player, "Wood", "Brick");
            }
            else if (move == 204) { // trade wood for stone
                makeTrade(player, "Wood", "stone");
            }
            else if (move == 205) { // trade sheep for wheat
                makeTrade(player, "Sheep", "Wheat");
            }
            else if (move == 206) { // trade sheep for wood
                makeTrade(player, "Sheep", "Wood");
            }
            else if (move == 207) { // trade sheep for brick
                makeTrade(player, "Sheep", "Brick");
            }
            else if (move == 208) { // trade sheep for stone
                makeTrade(player, "Sheep", "Stone");
            }
            else if (move == 209) { // trade brick for wheat
                makeTrade(player, "Brick", "Wheat");
            }
            else if (move == 210) { // trade brick for wood
                makeTrade(player, "Brick", "Wood");
            }
            else if (move == 211) { // trade brick for sheep
                makeTrade(player, "Brick", "Sheep");
            }
            else if (move == 212) { // trade brick for stone
                makeTrade(player, "Brick", "Stone");
            }
            else if (move == 213) { // trade stone for wheat
                makeTrade(player, "Stone", "Wheat");
            }
            else if (move == 214) { // trade stone for wood
                makeTrade(player, "Stone", "Wood");
            }
            else if (move == 215) { // trade stone for sheep
                makeTrade(player, "Stone", "Sheep");
            }
            else if (move == 216) { // trade stone for brick
                makeTrade(player, "Stone", "Brick");
            }

            longestRoad();
            largestArmy();
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, int, std::vector<int>> runGame() {
        setup();

        int curr_player_number = 0;
        int number_of_turns_made = 0;
        int max_score = 0;
        std::vector <Player>::iterator player_it;

        while (max_score < 0 && number_of_turns_made < 250) {
            mainGameTurn(players[curr_player_number]);

            curr_player_number += 1;
            curr_player_number = (curr_player_number % players.size());

            number_of_turns_made += 1;

            for (player_it = players.begin(); player_it != players.end(); player_it++)
                max_score = std::max(max_score, (*player_it).score());
        }

        std::vector <int> scores;
       
        for (player_it = players.begin(); player_it != players.end(); player_it++)
            scores[(*player_it).number] = (*player_it).score();

        std::sort(scores.begin(), scores.end(), std::greater<int>());

        torch::Tensor predictions;
        torch::Tensor labels;
        bool exists_predictions = false;
        bool exists_labels = false;

        for (player_it = players.begin(); player_it != players.end(); player_it++) {
            Player player = *player_it;

            torch::Tensor player_predictions = torch::stack(player.predictions);

            float player_place = players.size() - 1;

            for (unsigned int i = scores.size() - 1; i > 0; i--) {
                if (scores[i] == player.score()) {
                    player_place = i;
                }
            }

            torch::Tensor player_places = torch::full(player.predictions.size(), player_place);

            if (exists_labels == false)
                labels = player_places;
            else
                labels = torch::cat({ labels, player_places });

            if (exists_predictions == false)
                predictions = player_predictions;
            else
                predictions = torch::cat({ predictions, player_predictions });
        }

        return std::tuple{ predictions, labels, number_of_turns_made, scores };
    }
};

int main() {
    std::cout << "here 1";

    CatanNetwork model;

    std::cout << "here 2";

    model.train();
    
    torch::optim::AdamW optimizer(model.parameters(), 0.00005);

    unsigned int games_per_epoch = 8;

    std::cout << "\nLearning rate 5e-5, games " << games_per_epoch << " AdamW, MSE\n";

    for (int epoch = 0; epoch < 1000000; epoch++) {
        std::time_t start_time, end_time;
        time(&start_time);

        optimizer.zero_grad();
        torch::Tensor loss;
        int total_turns = 0;

        for (unsigned int i = 0; i < games_per_epoch; i++) {
            Game game(3, model);

            std::tuple<torch::Tensor, torch::Tensor, int, std::vector<int>> outcome = game.runGame();
            
            torch::Tensor predictions = std::get<0>(outcome);
            torch::Tensor labels = std::get<1>(outcome);
            int number_of_turns = std::get<2>(outcome);
            std::vector <int> scores = std::get<3>(outcome);

            int best_score = scores[0];
            total_turns += number_of_turns;

            torch::Tensor game_loss = torch::nn::MSELoss()(predictions, labels);
            game_loss *= ((number_of_turns / 250) * (number_of_turns / 250)) * (10 / std::min(best_score, 10)) / games_per_epoch;

            loss += game_loss;
        }

        loss.backward();
        optimizer.step();

        int output_epoch = epoch + 1;
        float output_turns = total_turns / games_per_epoch;
        time(&end_time);


        std::cout << output_epoch << "\t" << output_turns << "\t" << loss.item() << "\t" << difftime(end_time, start_time);
    }

    return 0;
}