import numpy as np

def to_bitboard(board):
    piece_types = [1, 2, 3, 4, 5, 6]
    colors = [True, False]
    bitboard = np.array([], dtype=bool)
    
    for ptype in piece_types:
        for color in colors:
            new_bitboard = np.array([0] * 64, dtype=bool)
            piece_indices = list(board.pieces(ptype, color))
            
            for index in piece_indices:
                new_bitboard[index] = True
            
            bitboard = np.concatenate((bitboard, new_bitboard))
    
    bitboard = np.concatenate((bitboard, meta_to_bits(board)))

    return bitboard

def meta_to_bits(board):
    turn = np.array([board.turn], dtype=bool)
    move_number = board.fullmove_number
    move_number = np.array([int(x) for x in np.binary_repr(move_number, width=8)], dtype=bool)
    
    en_passant = board.ep_square
    
    if en_passant == None:
        en_passant = 0
    
    en_passant = np.array([int(x) for x in np.binary_repr(en_passant, width=7)], dtype=bool)
    
    meta_bits = np.concatenate((turn, en_passant, move_number))
        
    return meta_bits

