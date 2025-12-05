import streamlit as st
import chess
from chess_engine import ChessMoveDecider  # or paste the class here

st.set_page_config(page_title="Chess Best Move Engine", layout="wide")
st.title("Chess Best Move Decider")

# Sidebar controls
st.sidebar.header("Engine Settings")
depth = st.sidebar.slider("Search depth", 1, 5, 3)

# Board FEN input
st.subheader("Position (FEN)")
default_fen = chess.STARTING_FEN
fen = st.text_input("FEN string", value=default_fen)

# Try to load board
error_box = st.empty()
try:
    board = chess.Board(fen)
    error_box.empty()
except Exception as e:
    error_box.error(f"Invalid FEN: {e}")
    st.stop()

# Show board as ASCII (simple text view)
st.text("Current board:")
st.text(str(board))

col1, col2 = st.columns(2)

with col1:
    st.write("Side to move:", "White" if board.turn == chess.WHITE else "Black")
    if st.button("Find best move"):
        engine = ChessMoveDecider(max_depth=depth)
        result = engine.find_best_move(board, time_limit=5.0)

        st.markdown("### Best Move")
        st.write(str(result["move"]))
        st.write(f"Evaluation: {result['evaluation']:.2f} pawns")
        st.write(f"Depth searched: {result['depth']}")
        st.write(f"Nodes: {result['nodes']:,}")
        st.write(f"Time: {result['time']:.3f} s")
        st.write(f"NPS: {result['nps']:,}")
    else:
        st.info("Click **Find best move** to analyze this position.")

with col2:
    st.markdown("### Make a move (optional)")
    move_uci = st.text_input("Enter your move in UCI (e.g. e2e4, g1f3)")
    if st.button("Apply move"):
        try:
            move = chess.Move.from_uci(move_uci.strip())
            if move in board.legal_moves:
                board.push(move)
                st.success("Move played. Copy this new FEN into the FEN box above to analyze next.")
                st.code(board.fen())
            else:
                st.error("Illegal move for this position.")
        except Exception as e:
            st.error(f"Invalid move: {e}")
