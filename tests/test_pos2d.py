from dctn.pos2d import Pos2D, pos_to_index, index_to_pos


def test_pos_index_conversion() -> None:
    max_w = 3
    # 0 1 2  3
    # 4 5 6  7
    # 8 9 10 11
    for (pos, index) in (
        (Pos2D(0, 0), 0),
        (Pos2D(1, 0), 4),
        (Pos2D(1, 1), 5),
        (Pos2D(2, 3), 11),
    ):
        assert pos_to_index(max_w, pos) == index
        assert index_to_pos(max_w, index) == pos

    max_w = 0
    # 0
    # 1
    # 2
    # 3
    # 4
    for (pos, index) in (
        (Pos2D(0, 0), 0),
        (Pos2D(1, 0), 1),
        (Pos2D(2, 0), 2),
        (Pos2D(3, 0), 3),
    ):
        assert pos_to_index(max_w, pos) == index
        assert index_to_pos(max_w, index) == pos
