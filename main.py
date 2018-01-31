#!/usr/bin/env python3

from game import Game
from ai import AI


def main():

    ai = AI()

    print('=' * 80)
    print("Training...")
    print('=' * 80)
    ai.train()

    print("AI is ready!")

    while True:
        print("Starting new game...")
        game = Game(ai)
        print(game)

        first = input("Do you wish to play first? ")
        if first and first[0].lower() == 'y':
            my_move = int(input("Enter your move: "))
            print(game.apply(my_move))

        while not game.over:
            ai_move = game.best_action
            print("AI plays:", ai_move)
            print(game.apply(ai_move))

            if game.over:
                break

            my_move = int(input("Enter your move: "))
            print(game.apply(my_move))

        print(f"Game over: {game.winner} wins!")


if __name__ == '__main__':
    main()
