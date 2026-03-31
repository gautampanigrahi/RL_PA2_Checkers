# Checkers 6x6 (PettingZoo AEC)

## Overview

-   Agents: `player_1`, `player_2`
-   Turn-based gameplay
-   Pieces:
    -   `1`, `2` -> player_1 (regular piece, king)
    -   `-1`, `-2` -> player_2 (regular piece, king)

------------------------------------------------------------------------

## Observation Space

``` python
Box(low=-2, high=2, shape=(6, 6), dtype=np.int8)
```

-   6X6 board
-   Values:
    -   `0`: empty
    -   `1/-1`: regular pieces
    -   `2/-2`: kings

Each agent observes the full board.

------------------------------------------------------------------------

## Action Space

``` python
Tuple((Discrete(6), Discrete(6), Discrete(6), Discrete(6)))
```

Action format:

``` python
(from_row, from_col, to_row, to_col)
```

-   Represents moving a piece from one position to another
-   Only legal moves are allowed
-   Legal actions are provided via:

``` python
info["legal_moves"]
```

------------------------------------------------------------------------

## Rewards

-   `+1` -> win
-   `-1` -> loss
-   `+0.25` -> capturing a piece
-   `+0.5` each -> draw (only kings remain)

------------------------------------------------------------------------

## Termination Conditions

The game ends when:

1.  A player has no pieces left\
2.  A player has no legal moves\
3.  Only kings remain (draw)

------------------------------------------------------------------------

## Notes

-   Captures are mandatory if available\
-   Multi-capture is enforced (same piece must continue capturing)\
-   No time-limit truncation is used
