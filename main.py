import chess
import chess.polyglot
from typing import Tuple, Optional, List, Dict
import time

class ChessMoveDecider:
    """
    Advanced chess move decider with comprehensive evaluation features.
    Uses minimax algorithm with alpha-beta pruning and multiple evaluation criteria.
    """
    
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.transposition_table = {}
        self.nodes_evaluated = 0
        
        # Piece values in centipawns (100 = 1 pawn)
        self.PIECE_VALUES = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Piece-square tables for positional evaluation
        # Values represent bonus points for pieces on specific squares
        self.PAWN_TABLE = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        self.KNIGHT_TABLE = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        self.BISHOP_TABLE = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        self.ROOK_TABLE = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        self.QUEEN_TABLE = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        self.KING_MIDDLE_GAME_TABLE = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
        
        self.KING_END_GAME_TABLE = [
            -50,-40,-30,-20,-20,-30,-40,-50,
            -30,-20,-10,  0,  0,-10,-20,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-30,  0,  0,  0,  0,-30,-30,
            -50,-30,-30,-30,-30,-30,-30,-50
        ]
    
    def get_piece_square_value(self, piece: chess.Piece, square: int, is_endgame: bool) -> int:
        """Get positional bonus for a piece on a specific square."""
        piece_type = piece.piece_type
        
        # Flip square index for black pieces
        if piece.color == chess.BLACK:
            square = chess.square_mirror(square)
        
        if piece_type == chess.PAWN:
            return self.PAWN_TABLE[square]
        elif piece_type == chess.KNIGHT:
            return self.KNIGHT_TABLE[square]
        elif piece_type == chess.BISHOP:
            return self.BISHOP_TABLE[square]
        elif piece_type == chess.ROOK:
            return self.ROOK_TABLE[square]
        elif piece_type == chess.QUEEN:
            return self.QUEEN_TABLE[square]
        elif piece_type == chess.KING:
            return self.KING_END_GAME_TABLE[square] if is_endgame else self.KING_MIDDLE_GAME_TABLE[square]
        
        return 0
    
    def is_endgame(self, board: chess.Board) -> bool:
        """Determine if position is in endgame phase."""
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        minors = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE)) +
                  len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK)))
        
        # Endgame if no queens or very few minor pieces
        return queens == 0 or (queens == 2 and minors <= 2)
    
    def evaluate_position(self, board: chess.Board) -> int:
        """
        Comprehensive position evaluation considering multiple features:
        - Material balance
        - Piece positioning (piece-square tables)
        - Mobility (number of legal moves)
        - King safety
        - Pawn structure
        - Control of center
        """
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        self.nodes_evaluated += 1
        score = 0
        is_endgame = self.is_endgame(board)
        
        # Material and positional evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.piece_type]
                value += self.get_piece_square_value(piece, square, is_endgame)
                
                score += value if piece.color == chess.WHITE else -value
        
        # Mobility evaluation
        mobility_score = len(list(board.legal_moves))
        board.turn = not board.turn
        mobility_score -= len(list(board.legal_moves))
        board.turn = not board.turn
        score += mobility_score * 10
        
        # Center control (e4, e5, d4, d5)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                bonus = 30 if piece.piece_type in [chess.PAWN, chess.KNIGHT] else 10
                score += bonus if piece.color == chess.WHITE else -bonus
        
        # Pawn structure evaluation
        score += self.evaluate_pawn_structure(board)
        
        # King safety in middle game
        if not is_endgame:
            score += self.evaluate_king_safety(board, chess.WHITE)
            score -= self.evaluate_king_safety(board, chess.BLACK)
        
        # Bishop pair bonus
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            score += 50
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            score -= 50
        
        return score if board.turn == chess.WHITE else -score
    
    def evaluate_pawn_structure(self, board: chess.Board) -> int:
        """Evaluate pawn structure (doubled, isolated, passed pawns)."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            multiplier = 1 if color == chess.WHITE else -1
            
            # Check for doubled and isolated pawns
            for file in range(8):
                file_pawns = [sq for sq in pawns if chess.square_file(sq) == file]
                
                # Doubled pawns penalty
                if len(file_pawns) > 1:
                    score -= 50 * multiplier * (len(file_pawns) - 1)
                
                # Isolated pawns penalty
                if file_pawns:
                    has_neighbor = False
                    for neighbor_file in [file - 1, file + 1]:
                        if 0 <= neighbor_file < 8:
                            if any(chess.square_file(sq) == neighbor_file for sq in pawns):
                                has_neighbor = True
                                break
                    if not has_neighbor:
                        score -= 20 * multiplier
        
        return score
    
    def evaluate_king_safety(self, board: chess.Board, color: bool) -> int:
        """Evaluate king safety based on pawn shield and piece attacks."""
        king_square = board.king(color)
        if king_square is None:
            return 0
        
        safety_score = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Pawn shield bonus
        pawn_shield_squares = []
        if color == chess.WHITE:
            if king_rank < 7:
                for file_offset in [-1, 0, 1]:
                    file = king_file + file_offset
                    if 0 <= file < 8:
                        pawn_shield_squares.append(chess.square(file, king_rank + 1))
        else:
            if king_rank > 0:
                for file_offset in [-1, 0, 1]:
                    file = king_file + file_offset
                    if 0 <= file < 8:
                        pawn_shield_squares.append(chess.square(file, king_rank - 1))
        
        for square in pawn_shield_squares:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                safety_score += 20
        
        return safety_score
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        Order moves for better alpha-beta pruning efficiency.
        Priority: captures > checks > other moves
        """
        def move_priority(move):
            score = 0
            
            # Prioritize captures
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
                    score += 10000 + self.PIECE_VALUES[victim.piece_type] - self.PIECE_VALUES[attacker.piece_type]
            
            # Prioritize checks
            board.push(move)
            if board.is_check():
                score += 5000
            board.pop()
            
            # Prioritize promotions
            if move.promotion:
                score += 8000
            
            # Prioritize castling
            if board.is_castling(move):
                score += 3000
            
            return score
        
        return sorted(moves, key=move_priority, reverse=True)
    
    def alpha_beta(self, board: chess.Board, depth: int, alpha: int, beta: int, 
                   maximizing_player: bool) -> Tuple[int, Optional[chess.Move]]:
        """
        Alpha-beta pruning algorithm with transposition table.
        Returns (evaluation_score, best_move)
        """
        # Check transposition table
        board_hash = chess.polyglot.zobrist_hash(board)
        if board_hash in self.transposition_table:
            stored_depth, stored_eval, stored_move = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_eval, stored_move
        
        # Base case: evaluate position at leaf nodes
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board), None
        
        legal_moves = list(board.legal_moves)
        legal_moves = self.order_moves(board, legal_moves)
        best_move = legal_moves[0] if legal_moves else None
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval_score, _ = self.alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            
            # Store in transposition table
            self.transposition_table[board_hash] = (depth, max_eval, best_move)
            return max_eval, best_move
        
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score, _ = self.alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            # Store in transposition table
            self.transposition_table[board_hash] = (depth, min_eval, best_move)
            return min_eval, best_move
    
    def find_best_move(self, board: chess.Board, time_limit: Optional[float] = None) -> Dict:
        """
        Find the best move for the current position.
        Returns dictionary with move, evaluation, and statistics.
        """
        start_time = time.time()
        self.nodes_evaluated = 0
        
        # Iterative deepening
        best_move = None
        best_eval = 0
        depth_reached = 0
        
        for depth in range(1, self.max_depth + 1):
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            eval_score, move = self.alpha_beta(
                board, depth, float('-inf'), float('inf'), 
                board.turn == chess.WHITE
            )
            
            if move:
                best_move = move
                best_eval = eval_score
                depth_reached = depth
        
        elapsed_time = time.time() - start_time
        
        return {
            'move': best_move,
            'evaluation': best_eval / 100.0,  # Convert to pawns
            'depth': depth_reached,
            'nodes': self.nodes_evaluated,
            'time': elapsed_time,
            'nps': int(self.nodes_evaluated / elapsed_time) if elapsed_time > 0 else 0
        }


# Example usage and demonstration
def main():
    # Create a chess board with a position
    board = chess.Board()
    
    # Example: Set up a specific position (or use default starting position)
    # board.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    
    print("Current position:")
    print(board)
    print()
    
    # Create the move decider
    decider = ChessMoveDecider(max_depth=4)
    
    # Find the best move
    print("Analyzing position...")
    result = decider.find_best_move(board, time_limit=10.0)
    
    print(f"\n{'='*50}")
    print("BEST MOVE ANALYSIS")
    print(f"{'='*50}")
    print(f"Best Move: {result['move']}")
    print(f"Evaluation: {result['evaluation']:.2f} pawns")
    print(f"Depth Searched: {result['depth']}")
    print(f"Nodes Evaluated: {result['nodes']:,}")
    print(f"Time Taken: {result['time']:.3f} seconds")
    print(f"Nodes per Second: {result['nps']:,}")
    print(f"{'='*50}")
    
    # Apply the move
    if result['move']:
        board.push(result['move'])
        print("\nPosition after best move:")
        print(board)


if __name__ == "__main__":
    main()
