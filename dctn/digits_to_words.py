from functools import reduce

digits = tuple(str(digit) for digit in range(10))
words = (
    "__zero__",
    "__one__",
    "__two__",
    "__three__",
    "__four__",
    "__five__",
    "__six__",
    "__seven__",
    "__eight__",
    "__nine__",
)

_d2w_dict = dict(zip(digits, words))
_w2d_dict = dict(zip(words, digits))


def d2w(x: str, check_valid: bool = True) -> str:
    """Digits to words."""
    result = reduce((lambda y, digit: y.replace(digit, _d2w_dict[digit])), digits, x)
    if check_valid:
        reconstruction = w2d(result)
        if reconstruction != x:
            raise ValueError("digits_to_words doesn't work with this word")
    return result


def w2d(x: str, check_valid: bool = True) -> str:
    """Words to digits."""
    return reduce((lambda y, word: y.replace(word, _w2d_dict[word])), words, x)
