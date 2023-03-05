import pygame

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (144, 238, 144)
GREY = (105, 105, 105)
RED = (220, 0, 0)

# Set the dimensions of the game board
BOARD_SIZE = (400, 400)
SQUARE_SIZE = 50

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

# Define the game loop
def game_loop():
    turn = 1
    game_over = False
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            # elif event.type == pygame.VIDEORESIZE: TODO IMPLEMENT RESIZABLE WINDOW
                # BOARD_SIZE = (event.w, event.h)
                #screen = pygame.display.set_mode(BOARD_SIZE, pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if turn == 1:
                    player = 1
                else:
                    player = -1
                pos = pygame.mouse.get_pos()
                row = pos[1] // SQUARE_SIZE
                col = pos[0] // SQUARE_SIZE
                if make_move(row, col, player):
                    if is_valid_move_available(-player):
                        turn *= -1
                    else:
                        print("Player", player, "has no valid moves. Skipping turn.")
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
    # Start the game loop
    game_loop()

# Quit pygame
pygame.quit()