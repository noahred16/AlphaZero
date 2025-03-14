import numpy as np
import random
from connect4.connect4 import Connect4

def main():
    game = Connect4()
    print("Welcome to Connect 4!")
    game.print_pretty()
    
    while True:
        # User's move
        legal_moves = game.get_legal_moves()
        user_move = None
        while user_move not in legal_moves:
            try:
                user_move = int(input(f"Your turn! Choose a column {legal_moves}: "))
                if user_move not in legal_moves:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Please enter a valid column number.")
        
        game.make_move(user_move)
        game.print_pretty()
        
        # Check for win/tie after user's move
        if game.evaluate_board() is not None:
            result = game.evaluate_board()
            if result > 0:
                print("Congratulations! You win!")
            elif result == 0:
                print("It's a tie!")
            break
        
        # Opponent's move (random)
        opponent_move = random.choice(game.get_legal_moves())
        print(f"Opponent chooses column {opponent_move}")
        game.make_move(opponent_move)
        game.print_pretty()
        
        # Check for win/tie after opponent's move
        if game.evaluate_board() is not None:
            result = game.evaluate_board()
            if result < 0:
                print("Opponent wins! Better luck next time.")
            elif result == 0:
                print("It's a tie!")
            break

if __name__ == "__main__":
    main()
