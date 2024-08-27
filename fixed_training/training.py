######################
## # # # # # # # # # #
# # # CATAN BOT # # ##
## # # # # # # # # # #
######################

import time

import torch
import torch.optim as optim

from game import CatanGame
from network import CatanNetwork

if __name__ == "__main__" : 
    # set device
    device=torch.device("cpu")

    # allow torch to use all threads
    torch.set_num_threads(24)
    torch.set_num_interop_threads(24)

    # redirect standard out
    output = open("out.txt", "a")

    # set learning rate
    learning_rate = 5e-5

    # set net and optimizer
    net = CatanNetwork(neurons=500) 
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    curr_epoch = 0

    # -- load net from checkpoint --
    # checkpoint = torch.load("checkpoint")
    # net = checkpoint["net"]
    # optimizer = checkpoint["optimizer"]
    # curr_epoch = checkpoint["epoch"]
    #
    # print(f"\n -- Restarting on epoch {curr_epoch+1} -- \n")

    # setup net
    net.to(device)
    net.train()

    # change the learning rate 
    for p in optimizer.param_groups: p["lr"] = learning_rate

    # set criterion and training schedule
    criterion = torch.nn.SmoothL1Loss()
    games_per_epoch = 2
    start_time = time.time()
    games_played = 0

    print(f"\nLearning rate {learning_rate}, games {games_per_epoch}, AdamW momentum, loss normal\n")

    for epoch in range(curr_epoch+1, 10000000) :         
        optimizer.zero_grad()
        loss = 0
        total_turns = 0 

        # create predictions and labels tensors to compute loss
        predictions = None
        labels = None 

        # iterate over number of desired games
        for i in range(games_per_epoch) : 
            # run game and return outcome
            game = CatanGame(3, net, optimizer, device)
            game_predictions, game_labels, number_of_turns, scores = game.runGame()

            # record turns metric
            total_turns += number_of_turns

            # record game labels and predictions
            if labels == None : 
                labels = game_labels
            else : 
                labels = torch.cat((labels, game_labels))

            if predictions == None : 
                predictions = game_predictions
            else : 
                predictions = torch.cat((predictions, game_predictions))

        # compute, clip and backpropogate loss
        loss = criterion(predictions, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=0.1, norm_type=2.0)
        optimizer.step()

        # record output information
        games_played += 1

        print(f"{epoch+1} \t {total_turns / games_per_epoch} \t {round(1000 * loss.item(), 3)} \t {round((time.time() - start_time) / games_played, 2)}")
        output.write(f"{epoch+1} \t {total_turns / games_per_epoch} \t {round(1000 * loss.item(), 3)} \t {round((time.time() - start_time) / games_played, 2)}\n")

        # save game data
        if (epoch+1) % 10 == 0 : 
            checkpoint = { 
                "epoch": epoch,
                "net": net,
                "optimizer": optimizer
            }

            torch.save(checkpoint, "checkpoint")

            output.close()
            output = open("out.txt", "a")

    