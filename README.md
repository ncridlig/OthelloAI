# OthelloAI

Welcome to Nicolas Cridlig and Akshay Gopalkrishnan's creation of an AI to play the game of Othello.

Our goal is to create a CNN model to play the game of Othello at an expert level. Based on previous research in this field,  Convolutional Neural Networks can achieve strong performance because Othello can be interpreted as an image-like matrix and CNN's are skilled at computer vision tasks. This problem can be motivated as a multi-class classification problem in which our model must choose 60 different moves for a current Othello board state. We use a Kaggle Othello dataset of games played by expert Othello players to construct a dataset of over 8 million board states and moves to train our player model. For our best model, we achieve an accuracy of 55.91\%, a top-3 accuracy of 87.47\%, and top-5 accuracy of 96.25\%, making it a difficult Othello player to beat. We also perform qualitative analysis of our Othello AI player model by playing against it, in which it easily wins the majority of the games. 

## Prerequesites

You just need to have Python 3.7+ installed (3.10+ for the simplified typehinting).

## Usage

1. Create a Python virtual environment

    ```bash
    python3 -m venv env
    ```

2. Start the virtual environment

    ```bash
    source env/bin/activate
    ```

3. Install dependencies

    ```bash
    pip install -U pip
    pip install -r requirements.txt
    ```

4. Run the game

    ```bash
    python game.py
    ```