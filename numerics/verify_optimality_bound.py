#!/usr/bin/env python3
"""Optional historical check of a scalar inequality for the small-k construction.

The current manuscript gives an analytic optimality proof and does not rely on
this calculation. This script is retained as a supplemental reproducibility
record of an earlier proof approach.

This certificate uses Python's standard library and exact rational arithmetic.
For every t in [3/5, 7/5], define

    H = ((1+t)/t) log(1+t) - 1,
    p = (1-2H)/(1+2H-t/4),
    delta = 1-pt,                    ell = pt/2,
    A = p-1-pt/2+2(1+p)log(1+t),    W = 1-p-pt/2,
    v = (2A-ell)/(2+p),
    F = (delta-A)(v+v^2/2) - (W-A).

The verified conclusions are F >= 1/1000, delta-A >= 1/6, and v >= 1/3.
Consequently (delta-A)(exp(v)-1)-(W-A) >= 1/1000 as well, because
exp(v)-1 >= v+v^2/2 for v >= 0.

In that approach, p=k-1, delta=delta_0, A=-d(x_0), W=w(x_0), and ell=x_0+1.
The connection of these quantities to the construction is not established by
this standalone script; it verifies only the scalar bounds stated above.

Every accepted interval has an exact rational enclosure satisfying the three
claimed bounds. Intervals that do not yet certify the bounds are bisected.
Both children are retained, so success certifies the whole closed interval.
The total length of the accepted partition is also checked exactly. There is
no reliance on floating-point arithmetic, external packages, or assertions
that Python's optimization flag could disable.

Run: python3 numerics/verify_optimality_bound.py
"""

from dataclasses import dataclass
from fractions import Fraction


Rational = int | Fraction


class NeedsSubdivision(ArithmeticError):
    """An interval enclosure is too wide for a division to be justified."""


def rational(value: Rational) -> Fraction:
    if not isinstance(value, (int, Fraction)):
        raise TypeError("Only integers and exact Fractions are permitted.")
    return Fraction(value)


@dataclass(frozen=True)
class Interval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", rational(self.lower))
        object.__setattr__(self, "upper", rational(self.upper))
        if self.lower > self.upper:
            raise ValueError("Interval endpoints are reversed.")

    @classmethod
    def point(cls, value: Rational) -> "Interval":
        value = rational(value)
        return cls(value, value)

    @staticmethod
    def coerce(value: "Interval | Rational") -> "Interval":
        return value if isinstance(value, Interval) else Interval.point(value)

    def __add__(self, other: "Interval | Rational") -> "Interval":
        other = self.coerce(other)
        return Interval(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other: "Interval | Rational") -> "Interval":
        return self + -self.coerce(other)

    def __rsub__(self, other: "Interval | Rational") -> "Interval":
        return self.coerce(other) + -self

    def __mul__(self, other: "Interval | Rational") -> "Interval":
        other = self.coerce(other)
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(min(products), max(products))

    __rmul__ = __mul__

    def reciprocal(self) -> "Interval":
        if self.lower <= 0 <= self.upper:
            raise NeedsSubdivision("Cannot invert an interval containing zero.")
        return Interval(1 / self.upper, 1 / self.lower)

    def __truediv__(self, other: "Interval | Rational") -> "Interval":
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other: "Interval | Rational") -> "Interval":
        return self.coerce(other) * self.reciprocal()


def log_one_plus_point(t: Fraction, terms: int = 14) -> Interval:
    """Enclose log(1+t) by rational bounds, for rational t >= 0.

    For z=t/(2+t), log(1+t)=2 sum_{j>=0} z^(2j+1)/(2j+1).
    After `terms` summands, every remaining denominator is at least
    2*terms+1. The positive tail is therefore bounded above by
    2*z^(2*terms+1)/((2*terms+1)*(1-z^2)), a geometric series.
    """
    t = rational(t)
    if t < 0 or not isinstance(terms, int) or terms < 1:
        raise ValueError("Require t >= 0 and a positive integer term count.")
    z = t / (2 + t)
    z_squared = z * z
    power = z
    lower = Fraction(0)
    for index in range(terms):
        lower += 2 * power / (2 * index + 1)
        power *= z_squared
    remainder = 2 * power / ((2 * terms + 1) * (1 - z_squared))
    return Interval(lower, lower + remainder)


def scalar_enclosures(left: Fraction, right: Fraction) -> tuple[Interval, ...]:
    t = Interval(left, right)
    # log is increasing, so enclosing its values at the endpoints suffices.
    logarithm = Interval(
        log_one_plus_point(left).lower,
        log_one_plus_point(right).upper,
    )
    H = (1 + t) / t * logarithm - 1
    denominator = 1 + 2 * H - t / 4
    if denominator.lower <= 0:
        raise NeedsSubdivision("Positivity of the denominator is unresolved.")
    p = (1 - 2 * H) / denominator
    A = p - 1 - p * t / 2 + 2 * (1 + p) * logarithm
    W = 1 - p - p * t / 2
    ell = p * t / 2
    delta = 1 - p * t
    if (2 + p).lower <= 0:
        raise NeedsSubdivision("Positivity of 2+p is unresolved.")
    v = (2 * A - ell) / (2 + p)
    multiplier = delta - A
    F = multiplier * (v + v * v / 2) - (W - A)
    return F, multiplier, v


def self_check() -> None:
    """Exercise sign-sensitive arithmetic and one independent log enclosure."""
    cases = (
        (Interval(-2, 3) * Interval(-4, 5), Interval(-12, 15)),
        (Interval(2, 4) / Interval(-4, -2), Interval(-2, Fraction(-1, 2))),
        (2 - Interval(-1, 3), Interval(-1, 3)),
    )
    for actual, expected in cases:
        if actual != expected:
            raise RuntimeError("An exact interval-arithmetic self-check failed.")
    try:
        Interval(-1, 1).reciprocal()
    except NeedsSubdivision:
        pass
    else:
        raise RuntimeError("Division by an interval containing zero was allowed.")
    log_two = log_one_plus_point(Fraction(1))
    if not (
        Fraction(693147180559, 10**12) < log_two.lower
        <= log_two.upper < Fraction(693147180560, 10**12)
    ):
        raise RuntimeError("The rational enclosure of log(2) failed its check.")


def verify() -> None:
    self_check()
    left, right = Fraction(3, 5), Fraction(7, 5)
    targets = (Fraction(1, 1000), Fraction(1, 6), Fraction(1, 3))
    pending = [(left, right, 0)]
    accepted_count = 0
    accepted_length = Fraction(0)
    maximum_depth = 0
    smallest_lower_bound = None
    while pending:
        start, end, depth = pending.pop()
        try:
            enclosures = scalar_enclosures(start, end)
        except NeedsSubdivision:
            enclosures = None
        if enclosures is not None and all(
            bound.lower >= target for bound, target in zip(enclosures, targets)
        ):
            accepted_count += 1
            accepted_length += end - start
            maximum_depth = max(maximum_depth, depth)
            lower_bound = enclosures[0].lower
            if smallest_lower_bound is None or lower_bound < smallest_lower_bound:
                smallest_lower_bound = lower_bound
            continue
        if depth >= 20:
            raise RuntimeError(
                f"Verification unresolved on [{start}, {end}] at depth {depth}."
            )
        midpoint = (start + end) / 2
        pending.extend(((start, midpoint, depth + 1), (midpoint, end, depth + 1)))

    if accepted_length != right - left or smallest_lower_bound is None:
        raise RuntimeError("The certified partition does not cover the domain.")
    # Round downward rationally only to keep the printed certificate concise.
    scale = 10**9
    rounded_down = Fraction(
        smallest_lower_bound.numerator * scale // smallest_lower_bound.denominator,
        scale,
    )
    if rounded_down < targets[0]:
        raise RuntimeError("The reported rational lower bound is insufficient.")
    print("Verified for every rational or real t in [3/5, 7/5]:")
    print("  F(t) >= 1/1000; delta_0-A_0 >= 1/6; v >= 1/3.")
    print(f"  Accepted intervals: {accepted_count}; maximum depth: {maximum_depth}.")
    print(f"  Exact total interval length: {accepted_length}.")
    print(f"  Smallest cell lower bound is at least {rounded_down}.")


if __name__ == "__main__":
    verify()
