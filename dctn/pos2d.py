from attr import attrs

@attrs(auto_attribs=True, frozen=True)
class Pos2D:
  h: int
  w: int


def pos_to_index(max_w: int, pos: Pos2D) -> int:
  """Returns index which `pos` has in the enumeration of all positions with the height
  varying in [0, ..., max_h] and width varying in [0, ..., max_w]. The order is like this:
  0 1 2  3
  4 5 6  7
  8 9 10 11"""
  assert pos.w <= max_w
  return pos.h * (max_w+1) + pos.w


def index_to_pos(max_w: int, index: int) -> Pos2D:
  """The inverse of `partial(pos_to_index, max_w)`"""
  assert (w := index % (max_w + 1)) <= max_w
  return Pos2D(index // (max_w+1), w)
