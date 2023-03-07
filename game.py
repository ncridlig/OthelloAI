import pygame
import torch
from model import Conv8
import os
import numpy as np
import time
import pdb

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (144, 238, 144)
GREY = (105, 105, 105)
RED = (220, 0, 0)
use_ai = True

# Set the dimensions of the game board
BOARD_SIZE = (480, 480)
SQUARE_SIZE = 60

# Initialize pygame
pygame.init()

# Set the title of the window
pygame.display.set_caption("Othello")

# Create the game board
board = [[0] * 8 for _ in range(8)]
board[3][3] = 1
board[4][4] = 1
board[3][4] = -1
board[4][3] = -1

# Create the game window
screen = pygame.display.set_mode(BOARD_SIZE)

# Load the game font
font = pygame.font.SysFont('Calibri', 25)

def make_board_matrices(board, tile):

    np_board = np.array(board)
    opp_tile = 1 if tile == -1 else -1
    player_positions = np.argwhere(np_board == tile)
    opponent_positions = np.argwhere(np_board == opp_tile)

    # Make player and opponent matrices
    player_matrix = np.zeros(shape=(8,8))
    player_matrix[player_positions[:,0], player_positions[:,1]] = 1

    opponent_matrix = np.zeros(shape=(8,8))
    opponent_matrix[opponent_positions[:,0], opponent_positions[:,1]] = 1

    # Stack matrices together and add regularization matrix
    board_matrix = np.array([player_matrix, opponent_matrix, np.ones((8,8))])

    return board_matrix

# Define a function to draw the game board
def draw_board():
    screen.fill(GREEN)
    for i in range(8):
        for j in range(8):
            rect = pygame.Rect(j * SQUARE_SIZE, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, GREY, rect, 1)
            if board[i][j] == 1:
                pygame.draw.circle(screen, BLACK, rect.center, SQUARE_SIZE // 2 - 5)
            elif board[i][j] == -1:
                pygame.draw.circle(screen, WHITE, rect.center, SQUARE_SIZE // 2 - 5)


# Define a function to check if a move is valid
def is_valid_move(row, col, player):
    if board[row][col] != 0:
        return False
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            r = row + i
            c = col + j
            if r < 0 or r >= 8 or c < 0 or c >= 8:
                continue
            if board[r][c] == -player:
                while r >= 0 and r < 8 and c >= 0 and c < 8:
                    if board[r][c] == 0:
                        break
                    elif board[r][c] == player:
                        return True
                    else:
                        r += i
                        c += j
    return False

# Define a function to make a move
def make_move(row, col, player):
    if not is_valid_move(row, col, player):
        return False
    board[row][col] = player
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            r = row + i
            c = col + j
            flip_list = []
            while r >= 0 and r < 8 and c >= 0 and c < 8:
                if board[r][c] == 0:
                    break
                elif board[r][c] == player:
                    for flip_row, flip_col in flip_list:
                        board[flip_row][flip_col] = player
                    break
                else:
                    flip_list.append((r, c))
                    r += i
                    c += j
            else:
                flip_list = []
    return True

# Define a function to get the current score
def get_score():
    black_score = 0
    white_score = 0
    for row in board:
        for cell in row:
            if cell == 1:
                black_score += 1
            elif cell == -1:
                white_score += 1
    return (black_score, white_score)

def convert_move_label(prediction):

    if prediction <= 26:
        row = prediction // 8
        col = prediction - row*8
    elif 27 <= prediction <= 32:
        row = (prediction+2) // 8
        col = (prediction+2) - row * 8
    else:
        row = (prediction + 4) // 8
        col = (prediction + 4) - row * 8
    return (row,col)

def ai_vs_ai_game_loop(model):
    turn = 1
    game_over = False

    # Turn model into evaluation mode
    model.eval()

    while not game_over:
        for event in pygame.event.get():

            # If there are no available switch turns
            if not is_valid_move_available(turn):
                print("Player", turn, "has not valid moves. Skipping turn.")
                turn *= -1
                if not is_valid_move_available(turn):
                    print('THE GAME IS OVER. THANK YOU FOR PLAYING AND HAVE A NICE DAY')
                    time.sleep(7)
                    game_over = True

            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                player = turn
                start_time = time.time()

                # Get 3x8x8 representation of board
                board_matrix = torch.tensor(make_board_matrices(board, turn)).float()

                # Add dimension to make 4D input
                board_matrix = board_matrix.unsqueeze(0)
                logits = model(board_matrix)
                predictions = torch.argsort(logits, dim=1, descending=True).flatten()
                for i in range(len(predictions)):
                    prediction = predictions[i].item()
                    row, col = convert_move_label(prediction)
                    if make_move(row, col, player):
                        turn *= -1
                        if i > 0: print("Attempts: ", i)
                        end_time = time.time()
                        elapsed = (end_time - start_time) * 1000
                        print("Prediction Time: {:.2f} ms".format(elapsed))
                        break

        screen.fill(BLACK)
        draw_board()
        black_score, white_score = get_score()
        black_text = font.render("Black: {}".format(black_score), True, RED)
        white_text = font.render("White: {}".format(white_score), True, RED)
        screen.blit(black_text, (10, 10))
        screen.blit(white_text, (10, 40))
        pygame.display.flip()

def ai_vs_player_game_loop(model):
    turn = 1
    game_over = False

    # Turn model into evaluation mode
    model.eval()

    while not game_over:
        for event in pygame.event.get():

            # If there are no available switch turns
            if not is_valid_move_available(turn):
                print("Player", turn, "has not valid moves. Skipping turn.")
                turn *= -1
                if not is_valid_move_available(turn):
                    print('THE GAME IS OVER. THANK YOU FOR PLAYING AND HAVE A NICE DAY')
                    time.sleep(7)
                    game_over = True

            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.MOUSEBUTTONDOWN and turn == 1:
                player = turn
                pos = pygame.mouse.get_pos()
                row = pos[1] // SQUARE_SIZE
                col = pos[0] // SQUARE_SIZE
                if make_move(row, col, player):
                    turn *= -1
            elif turn == -1:
                player = turn
                time.sleep(1)
                start_time = time.time()
                # Get 3x8x8 representation of board
                board_matrix = torch.tensor(make_board_matrices(board, turn)).float()
                # Add dimension to make 4D input
                board_matrix = board_matrix.unsqueeze(0)
                logits = model(board_matrix)
                predictions = torch.argsort(logits, dim=1, descending=True).flatten()
                for i in range(len(predictions)):
                    prediction = predictions[i].item()
                    row, col = convert_move_label(prediction)
                    if make_move(row, col, player):
                        turn *= -1
                        if i > 0: print("Attempts: ", i)
                        end_time = time.time()
                        elapsed = (end_time - start_time) * 1000
                        print("Prediction Time: {:.2f} ms".format(elapsed))
                        break

        screen.fill(BLACK)
        draw_board()
        black_score, white_score = get_score()
        black_text = font.render("Black: {}".format(black_score), True, RED)
        white_text = font.render("White: {}".format(white_score), True, RED)
        screen.blit(black_text, (10, 10))
        screen.blit(white_text, (10, 40))
        pygame.display.flip()


# Define the game loop
def game_loop():

    # Which player turn it is
    turn = 1
    game_over = False
    while not game_over:
        for event in pygame.event.get():

            # If there are no available switch turns
            if not is_valid_move_available(turn):
                print("Player", turn, "has not valid moves. Skipping turn.")
                turn *= -1
                if not is_valid_move_available(turn):
                    print('THE GAME IS OVER. THANK YOU FOR PLAYING AND HAVE A NICE DAY')
                    game_over = True

            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                player = turn
                pos = pygame.mouse.get_pos()
                row = pos[1] // SQUARE_SIZE
                col = pos[0] // SQUARE_SIZE
                if make_move(row, col, player):
                    turn *= -1

        screen.fill(BLACK)
        draw_board()
        black_score, white_score = get_score()
        black_text = font.render("Black: {}".format(black_score), True, RED)
        white_text = font.render("White: {}".format(white_score), True, RED)
        screen.blit(black_text, (10, 10))
        screen.blit(white_text, (10, 40))
        pygame.display.flip()

# Define a function to check if a valid move is available for the given player
def is_valid_move_available(player):
    for i in range(8):
        for j in range(8):
            if is_valid_move(i, j, player):
                return True
    return False


if __name__ == '__main__':

    # Load the best model and evaluate it on the test set
    best_model = Conv8()
    timestr = '20230306-231839'
    best_model.load_state_dict(torch.load(os.path.join('models', timestr + '.pth')))

    # Start the game loop
    ai_vs_player_game_loop(best_model)
    # ai_vs_ai_game_loop(best_model)

# Quit pygame
pygame.quit()