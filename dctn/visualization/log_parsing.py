import re
from typing import Dict, Tuple, Union, List, Optional, Iterable, Callable, TypeVar, Any

from attr import attrs, attrib

T = TypeVar("T")


def get_increasing_subsequence(
    xs: Iterable[T], calc_key: Callable[[T], Any] = lambda x: x
) -> Iterable[T]:
    iterator = iter(xs)
    try:
        x = next(iterator)
        max_key: T = calc_key(x)
        yield x
    except StopIteration:
        return
    for x in iterator:
        key = calc_key(x)
        if key > max_key:
            max_key = key
            yield x


@attrs(auto_attribs=True, frozen=True)
class Record:
    nitd: int
    trmce: float
    vmce: float
    tracc: float
    vacc: float


def _maybe_extract_info(line: str) -> Optional[Record]:
    pattern = r"After (?P<nitd>\d+) iters: train/val mean_ce=(?P<trmce>\d+\.\d+)/(?P<vmce>\d+\.\d+) acc=(?P<tracc>\d+\.\d+)%/(?P<vacc>\d+\.\d+)"
    match = re.search(pattern, line)
    if match:
        return Record(
            nitd=int(match["nitd"]),
            trmce=float(match["trmce"]),
            vmce=float(match["vmce"]),
            tracc=float(match["tracc"]) / 100.0,
            vacc=float(match["vacc"]) / 100.0,
        )


def load_records(log_fname: str, increasing_tracc: bool = False) -> Tuple[Record, ...]:
    with open(log_fname, encoding="utf-8") as f:
        lines = f.readlines()
    records = (record for line in lines if (record := _maybe_extract_info(line)))
    if increasing_tracc:
        records = get_increasing_subsequence(records, lambda record: record.tracc)
    return tuple(records)
