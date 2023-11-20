import numpy as np
import tensorflow as tf
import chess

from utils import to_bitboard

def get_legal_moves(board):
    legal_moves = []
    
    for move in board.legal_moves:
        legal_moves.append(move.uci())
    
    return legal_moves

def get_bitboard(board):
    bitboard = to_bitboard(board).astype(dtype=np.single)
    return bitboard

def games_moves_and_evals(game, model):
    boards_array = []
    games = []
    moves = get_legal_moves(game)

    if len(moves) < 1:
        return [], [], []
    
    for move in moves:
        new_game = game.copy()
        new_game.push(chess.Move.from_uci(move))
        games.append(new_game)
        bitboard = get_bitboard(new_game)
        boards_array.append(bitboard)
    
    bitboards_array = np.array(boards_array)

    evals = model(bitboards_array).numpy()
        
    return games, moves, evals

def get_nodes(node, model):
    games, moves, evals = games_moves_and_evals(node['game'], model)
    nodes = []

    for i, move in enumerate(moves):
        new_node = {
            'move': move,
            'eval': evals[i][0],
            'game': games[i],
            'parent': node
        }
        nodes.append(new_node)
    
    return nodes

def minimax(node, model, maximizing_player, alpha, beta, depth):
    if node['game'].outcome() != None:
        winner = node['game'].outcome().winner
        termination_node = {
            'move': node['move'],
            'eval': 0,
            'game': node['game'],
            'parent': node['parent']
        }
        if winner != None:
            termination_node['eval'] = int(winner == True) * 200 - 100
        return termination_node
    
    if depth == 0:
        return node
    
    nodes = get_nodes(node, model)
    
    if maximizing_player == True:
        max_eval = -1000
        max_node = None
        for new_node in nodes:
            node_eval = minimax(new_node, model, False, alpha, beta, depth-1)
            if node_eval['eval'] > max_eval:
                max_eval = node_eval['eval']
                max_node = new_node
            alpha = max(alpha, node_eval['eval'])
            if beta <= alpha:
                break
        return max_node
    else:
        min_eval = 1000
        min_node = None
        for new_node in nodes:
            node_eval = minimax(new_node, model, True, alpha, beta, depth-1)
            if node_eval['eval'] < min_eval:
                min_eval = node_eval['eval']
                min_node = new_node
            beta = min(beta, node_eval['eval'])
            if beta <= alpha:
                break
        return min_node

def ai_move(game, model, maximizing_player, depth):
    node = {
        'move': None,
        'eval': None,
        'game': game,
        'parent': None
    }
    result = minimax(node, model, maximizing_player, -1000, 1000, depth)
    return result['move'], result['eval']

def get_ai_move(fen, model):
    game = chess.Board(fen)
    player = False

    move, eval = ai_move(game, model, player, 3)

    return {
        "move": str(move),
        "eval": str(eval)
    }

def get_model(path):
    model = tf.keras.saving.load_model(path)
    return model
