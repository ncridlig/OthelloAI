import numpy as np
from itertools import product

def isValidMove(board, tile, xstart, ystart):

    # Returns False if the player's move on space xstart, ystart is invalid.
    # If it is a valid move, returns a list of spaces that would become the player's if they made a move here.
    if board[xstart][ystart] != -1 or not isOnBoard(xstart, ystart):
        return False

    board[xstart][ystart] = tile # temporarily set the tile on the board

    otherTile = int(not bool(tile))

    tilesToFlip = []
    for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:

        x, y = xstart, ystart
        x += xdirection  # first step in the direction
        y += ydirection  # first step in the direction
        if isOnBoard(x, y) and board[x][y] == otherTile:
            # There is a piece belonging to the other player next to our piece.
            x += xdirection
            y += ydirection
            if not isOnBoard(x, y):
                continue
            while board[x][y] == otherTile:
                x += xdirection
                y += ydirection
                if not isOnBoard(x,y): # break out of while loop, then continue in for loop
                    break
            if not isOnBoard(x,y):
                continue
            if board[x][y] == tile:
                # There are pieces to flip over. Go in the reverse direction until we reach the original
                # space. noting all the tiles along the way
                while True:
                    x -= xdirection
                    y -= ydirection
                    if x == xstart and y == ystart:
                        break
                    tilesToFlip.append([x,y])

    board[xstart][ystart] = -1
    if len(tilesToFlip) == 0:
        return False
    return tilesToFlip

def isOnBoard(x,y):
    return 0 <= x <= 7 and 0 <= y <= 7

def getValidMoves(board, tile):

    grid_positions = list(product(range(8), range(8)))

    # Returns a list of [x,y] lists of valid moves for the given player on the given board.
    return [[x,y] for x,y in grid_positions if isinstance(isValidMove(board, tile, x,y), list)]