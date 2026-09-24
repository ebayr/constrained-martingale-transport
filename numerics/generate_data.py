#!/usr/bin/env python3
"""Reproduce the numerical examples in the manuscript.

The implementation deliberately uses only the Python standard library so that the
data can be regenerated on a stock Python installation.  Every transport map is
evaluated from the parametrizations proved in the paper; no ODE solver or shooting
method is used.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def tex_scientific(value: float, digits: int = 3) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def bisect_root(
    function: Callable[[float], float],
    left: float,
    right: float,
    *,
    tolerance: float = 5.0e-14,
    iterations: int = 100,
) -> float:
    """Return the unique bracketed root of a continuous monotone function."""

    f_left = function(left)
    f_right = function(right)
    if abs(f_left) <= tolerance:
        return left
    if abs(f_right) <= tolerance:
        return right
    if f_left * f_right > 0.0:
        raise ValueError(
            f"root is not bracketed on [{left}, {right}]: "
            f"f(left)={f_left}, f(right)={f_right}"
        )
    for _ in range(iterations):
        midpoint = 0.5 * (left + right)
        f_midpoint = function(midpoint)
        if abs(f_midpoint) <= tolerance or right - left <= tolerance:
            return midpoint
        if f_left * f_midpoint <= 0.0:
            right = midpoint
            f_right = f_midpoint
        else:
            left = midpoint
            f_left = f_midpoint
    return 0.5 * (left + right)


def simpson(function: Callable[[float], float], left: float, right: float, n: int) -> float:
    """Composite Simpson quadrature with an even number of panels."""

    if right <= left:
        return 0.0
    if n % 2:
        n += 1
    step = (right - left) / n
    total = function(left) + function(right)
    for index in range(1, n):
        coefficient = 4.0 if index % 2 else 2.0
        total += coefficient * function(left + index * step)
    return total * step / 3.0


def lambert_w0(value: float) -> float:
    """Principal real Lambert W on the interval needed in this paper."""

    if value < -1.0 / math.e or value > 0.0:
        raise ValueError("lambert_w0 is implemented here only on [-1/e, 0]")
    if value == 0.0:
        return 0.0
    # The paper's arguments lie between roughly -0.26 and 0.  This initial
    # value stays safely on the principal branch, and Halley's method is cubic.
    iterate = -0.5 if value < -0.2 else value
    for _ in range(30):
        exponential = math.exp(iterate)
        residual = iterate * exponential - value
        denominator = exponential * (iterate + 1.0)
        denominator -= (iterate + 2.0) * residual / (2.0 * iterate + 2.0)
        update = residual / denominator
        iterate -= update
        if abs(update) <= 2.0e-15:
            break
    return iterate


def active_upper_displacement(k: float, x: float) -> float:
    """Evaluate U(x)-x from the principal Lambert-W representation."""

    argument = -(k - 1.0) / (2.0 * k) * math.exp((2.0 - x - k) / (2.0 * k))
    return k + 2.0 * k * lambert_w0(argument)


class LargeRegime:
    """Explicit coupling for 2 <= k <= 3."""

    def __init__(self, k: float) -> None:
        if not 2.0 <= k <= 3.0:
            raise ValueError("LargeRegime requires 2 <= k <= 3")
        self.k = k
        self.a = 2.0 * k - 5.0
        self.b = k - 2.0
        self.eta_b = self.b + active_upper_displacement(k, self.b)

    def E(self, eta: float) -> float:
        return (self.k - 1.0) * math.exp((2.0 - eta) / (2.0 * self.k))

    def rho(self, eta: float) -> float:
        return self.E(eta) + eta - self.k

    def R(self, eta: float) -> float:
        radicand = self.k * self.k - (eta + 2.0) * self.E(eta)
        return math.sqrt(max(0.0, radicand))

    def middle_x(self, eta: float) -> float:
        return self.E(eta) - 2.0 + self.R(eta)

    def maps(self, x: float) -> tuple[float, float, float]:
        if self.k == 3.0 or x <= self.a + 2.0e-13:
            lower = -0.5 * x - 1.5
            upper = 1.5 * x + 0.5
        elif x < self.b - 2.0e-13:
            eta = bisect_root(
                lambda value: self.middle_x(value) - x,
                self.eta_b,
                2.0,
            )
            e_value = self.E(eta)
            r_value = self.R(eta)
            lower = e_value + eta - 2.0 * self.k
            upper = e_value - 2.0 + 2.0 * r_value
        else:
            lower = x - self.k
            upper = x + active_upper_displacement(self.k, x)
        gap = upper - lower
        probability_upper = 0.5 if abs(gap) < 1.0e-13 else (x - lower) / gap
        return lower, upper, probability_upper


class SmallRegime:
    """Residual scalar construction for 1 < k < 2."""

    def __init__(self, k: float) -> None:
        if not 1.0 < k < 2.0:
            raise ValueError("SmallRegime requires 1 < k < 2")
        self.k = k
        self.b = k - 2.0
        self.delta_b = active_upper_displacement(k, self.b)
        self.delta_0 = self._find_delta_0()
        self.x_0 = -0.5 * (1.0 + self.delta_0)
        self.delta_star = bisect_root(lambda delta: self.y(delta) + 1.0, self.delta_b, 1.0)

    def z(self, delta: float) -> float:
        return (
            2.0
            - delta
            + 2.0 * self.k * math.log((self.k - 1.0) / (self.k - delta))
        )

    def y(self, delta: float) -> float:
        return self.z(delta) - self.k

    def A(self, delta: float) -> float:
        k = self.k
        return (
            (2.0 - k) * delta
            - 0.5 * delta * delta
            + 2.0
            * k
            * (
                delta * math.log(k - 1.0)
                + (k - delta) * math.log(k - delta)
                - (k - delta)
            )
        )

    def T(self, delta: float) -> float:
        return self.A(1.0) - self.A(delta) - (0.25 * (1.0 + delta) ** 2 - 1.0)

    def W_squared(self, delta: float) -> float:
        k = self.k
        b = self.b
        return (
            b * b
            - 2.0 * b * (delta - 1.0)
            + 2.0
            * k
            * ((k - delta) * math.log((k - delta) / (k - 1.0)) + delta - 1.0)
        )

    def W(self, delta: float) -> float:
        return math.sqrt(max(0.0, self.W_squared(delta)))

    def middle_x(self, delta: float) -> float:
        return self.b - delta + self.W(delta)

    def _find_delta_0(self) -> float:
        left = self.delta_b
        f_left = self.T(left)
        if f_left >= 0.0:
            raise ArithmeticError("the transition root has no negative left bracket")
        previous = left
        for index in range(1, 2001):
            current = left + (1.0 - left) * index / 2001.0
            if self.T(current) > 0.0:
                return bisect_root(self.T, previous, current)
            previous = current
        raise ArithmeticError("failed to bracket delta_0 before the trivial root at one")

    def maps(self, x: float) -> tuple[float, float, float]:
        if x >= self.b - 2.0e-13:
            delta = active_upper_displacement(self.k, x)
            lower = x - self.k
            upper = x + delta
        elif x >= self.x_0 - 2.0e-13:
            delta = bisect_root(
                lambda value: self.middle_x(value) - x,
                self.delta_b,
                self.delta_0,
            )
            lower = self.y(delta)
            upper = self.b - delta + 2.0 * self.W(delta)
        else:
            width = 2.0 * (x + 1.0)
            if width <= 1.0e-13:
                lower = upper = -1.0
            else:
                function = lambda left: (
                    self.A(left + width)
                    - self.A(left)
                    - (0.25 * width * width - width)
                )
                left_coordinate = bisect_root(
                    function,
                    self.delta_b,
                    1.0 - width,
                )
                right_coordinate = left_coordinate + width
                lower = self.y(left_coordinate)
                upper = self.y(right_coordinate)
        gap = upper - lower
        probability_upper = 0.5 if abs(gap) < 1.0e-13 else (x - lower) / gap
        return lower, upper, probability_upper


def evenly_spaced(left: float, right: float, count: int) -> Iterable[float]:
    for index in range(count):
        yield left + (right - left) * index / (count - 1)


def write_support_data(name: str, coupling: LargeRegime | SmallRegime) -> None:
    points = set(evenly_spaced(-1.0, 1.0, 1201))
    if isinstance(coupling, LargeRegime):
        points.update((coupling.a, coupling.b))
    else:
        points.update((coupling.x_0, coupling.b))
    path = FIGURES / name
    with path.open("w", encoding="ascii") as stream:
        stream.write("x D U B q slack w\n")
        for x in sorted(points):
            lower, upper, probability_upper = coupling.maps(x)
            boundary = x - coupling.k
            stream.write(
                f"{x:.12f} {lower:.12f} {upper:.12f} {boundary:.12f} "
                f"{probability_upper:.12f} {lower-boundary:.12f} {upper-x:.12f}\n"
            )


def write_histogram_data(
    name: str,
    coupling: LargeRegime | SmallRegime,
    *,
    source_points: int = 80000,
    bins: int = 64,
) -> tuple[float, float, float]:
    masses = [0.0] * bins
    width = 4.0 / bins
    point_mass = 1.0 / source_points
    mean_error = 0.0
    minimum_slack = math.inf
    for index in range(source_points):
        x = -1.0 + 2.0 * (index + 0.5) / source_points
        lower, upper, probability_upper = coupling.maps(x)
        mean_error = max(
            mean_error,
            abs((1.0 - probability_upper) * lower + probability_upper * upper - x),
        )
        minimum_slack = min(minimum_slack, lower - (x - coupling.k))
        for y, mass in (
            (lower, point_mass * (1.0 - probability_upper)),
            (upper, point_mass * probability_upper),
        ):
            bin_index = min(bins - 1, max(0, int((y + 2.0) / width)))
            masses[bin_index] += mass
    densities = [mass / width for mass in masses]
    l1_error = sum(abs(density - 0.25) * width for density in densities)
    max_error = max(abs(density - 0.25) for density in densities)
    with (FIGURES / name).open("w", encoding="ascii") as stream:
        stream.write("y density target error_scaled\n")
        for index, density in enumerate(densities):
            midpoint = -2.0 + (index + 0.5) * width
            stream.write(
                f"{midpoint:.12f} {density:.12f} 0.250000000000 "
                f"{1.0e4*(density-0.25):.12f}\n"
            )
    return l1_error, max_error, max(mean_error, max(0.0, -minimum_slack))


def target_cdf_error(
    coupling: LargeRegime | SmallRegime,
    *,
    source_points: int = 20000,
    target_points: int = 20001,
) -> float:
    atoms: list[tuple[float, float]] = []
    point_mass = 1.0 / source_points
    for index in range(source_points):
        x = -1.0 + 2.0 * (index + 0.5) / source_points
        lower, upper, probability_upper = coupling.maps(x)
        atoms.append((lower, point_mass * (1.0 - probability_upper)))
        atoms.append((upper, point_mass * probability_upper))
    atoms.sort()
    cumulative = 0.0
    atom_index = 0
    maximum_error = 0.0
    for index in range(target_points):
        target_value = -2.0 + 4.0 * index / (target_points - 1)
        while atom_index < len(atoms) and atoms[atom_index][0] <= target_value:
            cumulative += atoms[atom_index][1]
            atom_index += 1
        exact = 0.25 * (target_value + 2.0)
        maximum_error = max(maximum_error, abs(cumulative - exact))
    return maximum_error


def write_density_decomposition(coupling: LargeRegime) -> None:
    with (FIGURES / "density_decomposition_k25.dat").open("w", encoding="ascii") as stream:
        stream.write("y middle boundary total target\n")
        for eta in evenly_spaced(coupling.eta_b, 2.0, 501):
            e_value = coupling.E(eta)
            y = e_value + eta - 2.0 * coupling.k
            middle = e_value / (4.0 * (2.0 * coupling.k - e_value))
            boundary = (coupling.k - e_value) / (
                2.0 * (2.0 * coupling.k - e_value)
            )
            stream.write(
                f"{y:.12f} {middle:.12f} {boundary:.12f} "
                f"{middle+boundary:.12f} 0.250000000000\n"
            )


def integration_intervals(
    coupling: LargeRegime | SmallRegime,
) -> list[tuple[float, float]]:
    if isinstance(coupling, LargeRegime):
        breakpoints = (-1.0, coupling.a, coupling.b, 1.0)
    else:
        breakpoints = (-1.0, coupling.x_0, coupling.b, 1.0)
    distinct: list[float] = []
    for point in breakpoints:
        point = min(1.0, max(-1.0, point))
        if not distinct or abs(point - distinct[-1]) > 1.0e-13:
            distinct.append(point)
    return list(zip(distinct[:-1], distinct[1:]))


def integrate_by_branch(
    coupling: LargeRegime | SmallRegime,
    function: Callable[[float], float],
    *,
    panels_per_branch: int = 2400,
) -> float:
    return sum(
        simpson(function, left, right, panels_per_branch)
        for left, right in integration_intervals(coupling)
    )


def expectation(coupling: LargeRegime | SmallRegime, function: Callable[[float], float]) -> float:
    def integrand(x: float) -> float:
        lower, upper, probability_upper = coupling.maps(x)
        return 0.5 * (
            (1.0 - probability_upper) * function(lower - x)
            + probability_upper * function(upper - x)
        )

    return integrate_by_branch(coupling, integrand)


def target_second_moment(coupling: LargeRegime | SmallRegime) -> float:
    def integrand(x: float) -> float:
        lower, upper, probability_upper = coupling.maps(x)
        return 0.5 * (
            (1.0 - probability_upper) * lower * lower
            + probability_upper * upper * upper
        )

    return integrate_by_branch(coupling, integrand)


def boundary_mass(coupling: LargeRegime) -> float:
    if coupling.k == 3.0:
        return 0.0

    def integrand(x: float) -> float:
        _, _, probability_upper = coupling.maps(x)
        return 0.5 * (1.0 - probability_upper)

    return simpson(integrand, coupling.b, 1.0, 2400)


def write_value_curve() -> None:
    with (FIGURES / "cubic_value.dat").open("w", encoding="ascii") as stream:
        stream.write("k value boundary_mass active_source_mass\n")
        for index in range(101):
            k = 2.0 + index / 100.0
            coupling = LargeRegime(k)
            value = expectation(coupling, lambda displacement: displacement**3)
            stream.write(
                f"{k:.8f} {value:.12f} {boundary_mass(coupling):.12f} "
                f"{0.5*(3.0-k):.12f}\n"
            )


def write_generated_tex(
    histogram_diagnostics: dict[str, tuple[float, float, float]],
) -> None:
    large_rows = []
    for k in (2.0, 2.25, 2.5, 2.75, 3.0):
        coupling = LargeRegime(k)
        value = expectation(coupling, lambda displacement: displacement**3)
        large_rows.append(
            f"{k:.2f} & {coupling.a:.6f} & {coupling.b:.6f} & "
            f"{coupling.eta_b:.6f} & {boundary_mass(coupling):.6f} & "
            f"{value:.6f} \\\\"
        )
    small_rows = []
    for k in (1.25, 1.5, 1.75):
        coupling = SmallRegime(k)
        small_rows.append(
            f"{k:.2f} & {coupling.delta_b:.6f} & {coupling.delta_0:.6f} & "
            f"{coupling.x_0:.6f} & {target_second_moment(coupling):.8f} \\\\"
        )

    k15 = SmallRegime(1.5)
    k25 = LargeRegime(2.5)
    second_moment_15 = target_second_moment(k15)
    second_moment_25 = target_second_moment(k25)
    cdf_error_15 = target_cdf_error(k15)
    cdf_error_25 = target_cdf_error(k25)
    if abs(second_moment_15 - 4.0 / 3.0) > 5.0e-8:
        raise AssertionError("the k=1.5 target second moment check failed")
    if abs(second_moment_25 - 4.0 / 3.0) > 5.0e-8:
        raise AssertionError("the k=2.5 target second moment check failed")
    if max(cdf_error_15, cdf_error_25) > 5.0e-5:
        raise AssertionError("the deterministic target-CDF check failed")
    lines = [
        "% Generated by numerics/generate_data.py; do not edit by hand.",
        f"\\newcommand{{\\KOneFiveDeltaB}}{{{k15.delta_b:.9f}}}",
        f"\\newcommand{{\\KOneFiveDeltaZero}}{{{k15.delta_0:.9f}}}",
        f"\\newcommand{{\\KOneFiveDeltaStar}}{{{k15.delta_star:.9f}}}",
        f"\\newcommand{{\\KOneFiveXZero}}{{{k15.x_0:.9f}}}",
        f"\\newcommand{{\\KTwoFiveEtaB}}{{{k25.eta_b:.9f}}}",
        f"\\newcommand{{\\KOneFiveSecondMoment}}{{{second_moment_15:.9f}}}",
        f"\\newcommand{{\\KTwoFiveSecondMoment}}{{{second_moment_25:.9f}}}",
        f"\\newcommand{{\\KOneFiveHistLone}}{{{tex_scientific(histogram_diagnostics['k15'][0])}}}",
        f"\\newcommand{{\\KTwoFiveHistLone}}{{{tex_scientific(histogram_diagnostics['k25'][0])}}}",
        f"\\newcommand{{\\KOneFivePointwiseResidual}}{{{tex_scientific(histogram_diagnostics['k15'][2])}}}",
        f"\\newcommand{{\\KTwoFivePointwiseResidual}}{{{tex_scientific(histogram_diagnostics['k25'][2])}}}",
        f"\\newcommand{{\\KOneFiveCdfResidual}}{{{tex_scientific(cdf_error_15)}}}",
        f"\\newcommand{{\\KTwoFiveCdfResidual}}{{{tex_scientific(cdf_error_25)}}}",
        "\\newcommand{\\LargeRegimeRows}{%",
        *large_rows,
        "}",
        "\\newcommand{\\SmallRegimeRows}{%",
        *small_rows,
        "}",
        "",
    ]
    (ROOT / "numerics" / "generated_values.tex").write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    k15 = SmallRegime(1.5)
    k25 = LargeRegime(2.5)
    write_support_data("support_k15.dat", k15)
    write_support_data("support_k25.dat", k25)
    write_density_decomposition(k25)
    histogram_diagnostics = {
        "k15": write_histogram_data("marginal_k15.dat", k15),
        "k25": write_histogram_data("marginal_k25.dat", k25),
    }
    if max(diagnostic[2] for diagnostic in histogram_diagnostics.values()) > 1.0e-12:
        raise AssertionError("a pointwise martingale or support check failed")
    write_value_curve()
    write_generated_tex(histogram_diagnostics)

    print("Generated all numerical data.")
    for label, coupling in (("k=1.5", k15), ("k=2.5", k25)):
        value = expectation(coupling, lambda displacement: displacement**3)
        second_moment = target_second_moment(coupling)
        print(
            f"{label}: E[(Y-X)^3]={value:.10f}, "
            f"E[Y^2]={second_moment:.10f}, "
            f"histogram L1={histogram_diagnostics[label.replace('=', '').replace('.', '')][0]:.3e}"
        )


if __name__ == "__main__":
    main()
