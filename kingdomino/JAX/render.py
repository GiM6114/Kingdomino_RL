from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

colors = [
    "orange",      # 0
    "black",       # 1
    "lightgreen",  # 2
    "darkgreen",   # 3
    "yellow",      # 4
    "blue",        # 5
    "brown",       # 6
    "grey",        # 7
    "black",       # 8
]
type2color = {i:color for i,color in zip(range(-2,7),colors)}

def render_tile(tile, ax, title=None):
    """
    tile:
    (
        tile_type_1,
        tile_type_2,
        crown_1,
        crown_2,
        value,
        player_id_belong
    )
    """
    print(tile)

    t1, t2, c1, c2, value, owner = [int(x) for x in tile]

    ax.clear()

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # left half
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=type2color[t1],
            edgecolor="black",
            linewidth=2,
        )
    )

    # right half
    ax.add_patch(
        Rectangle(
            (1, 0),
            1,
            1,
            facecolor=type2color[t2],
            edgecolor="black",
            linewidth=2,
        )
    )

    # crowns
    if c1 > 0:
        ax.text(
            0.5,
            0.5,
            "♕" * c1,
            ha="center",
            va="center",
            fontsize=16,
            color="black",
            fontweight="bold",
        )

    if c2 > 0:
        ax.text(
            0.5,
            0.5,
            "♕" * c2,
            ha="center",
            va="center",
            fontsize=16,
            color="black",
            fontweight="bold",
        )

    # tile value
    ax.text(
        1,
        -0.15,
        f"{value}",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    # owner
    if owner >= 0:
        ax.text(
            1,
            1.8,
            f"P{owner}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="red",
            fontweight="bold",
        )

    if title is not None:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])
    
def render_state(state):
    n_players = state.boards.shape[0]

    fig = plt.figure(figsize=(4 * n_players, 5))
    
    gs = gridspec.GridSpec(
        3,
        n_players,
        height_ratios=[1, 8, 1]
    )
    #
    # TOP ROW: current tiles
    #
    current_axes = []

    for i in range(n_players):
        ax = fig.add_subplot(gs[0, i])

        render_tile(
            np.array(state.current_tiles[i]),
            ax,
            title=f"Current {i}",
        )

        current_axes.append(ax)

    #
    # MIDDLE ROW: boards
    #
    board_axes = []

    for i in range(n_players):
        ax = fig.add_subplot(gs[1, i])
        render_board(
            np.array(state.boards[i]),
            np.array(state.crowns[i]),
            ax,
            title=f"P{i} score={int(state.scores[i])}",
        )

        board_axes.append(ax)

    #
    # BOTTOM ROW: previous tiles
    #
    prev_axes = []

    for i in range(n_players):
        ax = fig.add_subplot(gs[2, i])

        render_tile(
            state.previous_tiles[i],
            ax,
            title=f"Previous P{i}",
        )

        prev_axes.append(ax)

    fig.suptitle(
        f"Turn {int(state.turn_id)} | Current player: P{int(state.current_player_id)}",
        fontsize=16,
    )

    plt.tight_layout()

    return fig

def render_board(board, crowns, ax, title=""):
    s = board.shape[0]

    ax.clear()
    ax.set_title(title)

    ax.set_xlim(-0.5, s - 0.5)
    ax.set_ylim(s - 0.5, -0.5)
    ax.set_aspect("equal")

    # Grid lines
    for x in range(s + 1):
        ax.axvline(x - 0.5, color="black", lw=1)
        ax.axhline(x - 0.5, color="black", lw=1)

    for i in range(s):
        for j in range(s):
            env = board[i, j]
            c = crowns[i, j]

            color = type2color[int(env)]

            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="none",
                )
            )

            if c > 0:
                ax.text(
                    j,
                    i,
                    "♕" * int(c),
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="black",
                    fontweight="bold",
                )

    ax.set_xticks([])
    ax.set_yticks([])