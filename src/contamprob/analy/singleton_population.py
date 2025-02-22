"""Analytical solution to the constant contamination period scenario."""

import math
from typing import NamedTuple
from collections.abc import Iterable
from collections import UserDict

import numpy as np

from ..problem_setup import ContaminationProcess, PoissonProcess, SingletonPopulation
from ..analytical import Case


def kronecker_delta(i: int, j: int) -> int:
    """Kronecker delta function."""
    return 1 if i == j else 0


def get_polynomial(coeffs: Iterable):
    """Get a polynomial from its coefficients."""
    coeffs_ = list(coeffs)
    if len(coeffs_) == 0:
        return lambda _: 0
    return np.polynomial.Polynomial(coeffs_)


class MainIndice(NamedTuple):
    """Quadruple indice used to label the main coefficients."""

    n: int
    l: int  # noqa: E741
    m: int
    j: int


class LVIndice(NamedTuple):
    """Indice used to label the coefficients of L & V integrals."""

    q: int
    r: int
    m: int
    j: int


class Table:
    """
    Represents a table used in the loop solver.
    """

    class PCoefficients(UserDict[MainIndice, float]):
        def __setitem__(self, key: tuple[int, int, int, int], item: float) -> None:
            return super().__setitem__(MainIndice(*key), item)

        def __getitem__(self, key: tuple[int, int, int, int]) -> float:
            key_ = MainIndice(*key)
            if key_.m == -1 or key_.m == key_.n + 1:
                return 0
            return self.data[key_]

    class LVCoefficients(UserDict[LVIndice, float]):
        def __setitem__(self, key: tuple[int, int, int, int], item: float) -> None:
            return super().__setitem__(LVIndice(*key), item)

        def __getitem__(self, key: tuple[int, int, int, int]) -> float:
            key_ = LVIndice(*key)
            if key_.m == -1:
                return 0
            return self.data[key_]

    def __init__(self) -> None:
        # Initialize the tables
        self.a = self.PCoefficients()
        self.A = self.PCoefficients()
        # Tables for the L & V integrals
        # Not used for the l=0 case
        self.ξ = self.LVCoefficients()
        self.Ξ = self.LVCoefficients()
        self.ζ = self.LVCoefficients()
        self.Z = self.LVCoefficients()
        self.η = self.LVCoefficients()
        self.θ = self.LVCoefficients()
        # Initialize the maximum indices
        self.max_n = -1
        self.max_l = -1

    @property
    def max_indice(self) -> Case:
        """Get the maximum indice."""
        return Case(self.max_n, self.max_l)

    @max_indice.setter
    def max_indice(self, value: tuple[int, int]):
        """Set the maximum indice."""
        self.max_n, self.max_l = value


class LoopSolver:
    """Iterative solver to the double Poisson processes problem using the compact notation approach."""

    def __init__(
        self,
        ctnm_proc: ContaminationProcess[SingletonPopulation],
        event_proc: PoissonProcess,
    ) -> None:
        self.μ = event_proc.rate
        self.λ = ctnm_proc.process.rate
        self.τ = ctnm_proc.contamination.value
        self.table = Table()

    def fill_in_ksi(self, ind: tuple[int, int, int, int]):
        """Fill in the ξ coefficients."""
        q, r, m, j = ind
        if r > j:
            self.table.ξ[(q, r, m, j)] = 0
        else:
            if q == 0:
                self.table.ξ[(0, r, m, j)] = (
                    -math.factorial(j) / math.factorial(r) / (-self.μ) ** (1 + j - r)
                )
            elif j == 0:
                self.table.ξ[(q, 0, m, 0)] = 1 / self.μ
            else:
                self.table.ξ[(q, r, m, j)] = (
                    self.table.ξ[(q - 1, r, m, j)]
                    - j / self.μ * self.table.ξ[(q, r, m, j - 1)]
                )

    def fill_in_zeta(self, ind: tuple[int, int, int, int]):
        """Fill in the ζ coefficients."""
        q, r, m, j = ind
        if r > j:
            self.table.ζ[(q, r, m, j)] = 0
        else:
            if q == 0:
                self.table.ζ[(0, r, m, j)] = (
                    np.exp(-self.μ * self.τ)
                    * math.factorial(j)
                    / math.factorial(r)
                    / (-self.μ) ** (1 + j - r)
                )
            elif j == 0:
                self.table.ζ[(q, 0, m, 0)] = (
                    -np.exp(-self.μ * self.τ)
                    * get_polynomial([1 / math.factorial(k) for k in range(q + 1)])(
                        self.μ * self.τ
                    )
                    / self.μ
                )
            else:
                term_1 = (
                    (self.μ * self.τ) ** q
                    / math.factorial(q)
                    * np.exp(-self.μ * self.τ)
                    / (-self.μ)
                    * kronecker_delta(r, j)
                )
                term_2_3 = (
                    self.table.ζ[(q - 1, r, m, j)]
                    - j / self.μ * self.table.ζ[(q, r, m, j - 1)]
                )
                self.table.ζ[(q, r, m, j)] = term_1 + term_2_3
                del term_1, term_2_3

    def fill_in_Ksi(self, ind: tuple[int, int, int, int]):
        """Fill in the Ξ coefficients."""
        q, r, m, j = ind
        if r > q:
            self.table.Ξ[(q, r, m, j)] = 0
        else:
            if q == 0:
                self.table.Ξ[(0, 0, m, j)] = (
                    np.exp(self.μ * m * self.τ)
                    * math.factorial(j)
                    / (-self.μ) ** (1 + j)
                )
            elif j == 0:
                self.table.Ξ[(q, r, m, 0)] = (
                    -1
                    / math.factorial(r)
                    * self.μ ** (r - 1)
                    * np.exp(self.μ * m * self.τ)
                )
            else:
                self.table.Ξ[(q, r, m, j)] = (
                    self.table.Ξ[(q - 1, r, m, j)]
                    - j / self.μ * self.table.Ξ[(q, r, m, j - 1)]
                )

    def fill_in_Zeta(self, ind: tuple[int, int, int, int]):
        """Fill in the Z coefficients."""
        q, r, m, j = ind
        if r > q:
            self.table.Z[(q, r, m, j)] = 0
        else:
            if q == 0:
                self.table.Z[(0, 0, m, j)] = (
                    -np.exp(self.μ * m * self.τ)
                    * math.factorial(j)
                    / (-self.μ) ** (1 + j)
                )
            elif j == 0:
                self.table.Z[(q, r, m, 0)] = np.exp(self.μ * m * self.τ) * sum(
                    1
                    / math.factorial(k)
                    * self.μ ** (k - 1)
                    * math.comb(k, r)
                    * self.τ ** (k - r)
                    for k in range(r, q + 1)
                )
            else:
                self.table.Z[(q, r, m, j)] = (
                    self.table.Z[(q - 1, r, m, j)]
                    - j / self.μ * self.table.Z[(q, r, m, j - 1)]
                )

    def fill_in_eta(self, ind: tuple[int, int, int, int]):
        """Fill in the η coefficients."""
        q, r, m, j = ind
        if r > j + q + 1:
            self.table.η[(q, r, m, j)] = 0
        elif q == 0:
            if r == j + 1:
                self.table.η[(0, r, m, j)] = 1 / (j + 1)
            else:
                self.table.η[(0, r, m, j)] = 0
        elif j == 0:
            if r == q + 1:
                self.table.η[(q, r, m, 0)] = self.μ**q / math.factorial(q + 1)
            else:
                self.table.η[(q, r, m, 0)] = 0
        else:
            self.table.η[(q, r, m, j)] = (
                self.μ / (j + 1) * self.table.η[(q - 1, r, m, j + 1)]
            )

    def fill_in_theta(self, ind: tuple[int, int, int, int]):
        """Fill in the θ coefficients."""
        q, r, m, j = ind
        if r > j + q + 1:
            self.table.θ[(q, r, m, j)] = 0
        elif q == 0:
            if r == j + 1:
                self.table.θ[(0, r, m, j)] = -1 / (j + 1)
            else:
                self.table.θ[(0, r, m, j)] = 0
        elif j == 0:
            if r > q + 1:
                self.table.θ[(q, r, m, 0)] = 0
            else:
                term_1 = (
                    self.μ**q
                    / math.factorial(q + 1)
                    * self.τ ** (q + 1)
                    * kronecker_delta(r, 0)
                )
                term_2 = (
                    -(self.μ**q)
                    / math.factorial(q + 1)
                    * math.comb(q + 1, r)
                    * self.τ ** (q + 1 - r)
                )
                self.table.θ[(q, r, m, 0)] = term_1 + term_2
                del term_1, term_2
        else:
            term_1 = (
                -1
                / (j + 1)
                * (self.μ * self.τ) ** q
                / math.factorial(q)
                * kronecker_delta(r, j + 1)
            )
            term_2 = self.μ / (j + 1) * self.table.θ[(q - 1, r, m, j + 1)]
            self.table.θ[(q, r, m, j)] = term_1 + term_2
            del term_1, term_2

    def fill_in_a(self, ind: tuple[int, int, int]):
        n, l, m = ind  # noqa: E741
        if n == 0:
            for j in range(n + l + 1):
                self.table.a[(0, l, 0, j)] = kronecker_delta(j, 0)
        elif l == 0:
            # This case is actually included in the l != 0 case
            # But we first implemented this case to check the correctness of the formula
            # for its simplicity
            term_1 = (
                self.λ
                * np.exp(-self.μ * m * self.τ)
                * sum(
                    self.table.A[(n - 1, 0, m - 1, k)]
                    * math.factorial(k)
                    / (self.μ ** (1 + k))
                    for k in range(n)
                )
            )
            term_2_3 = self.λ * sum(
                (
                    -self.table.a[(n - 1, 0, m, k)]
                    + self.table.a[(n - 1, 0, m - 1, k)] * np.exp(-self.μ * self.τ)
                )
                * math.factorial(k)
                / (-self.μ) ** (1 + k)
                for k in range(n)
            )
            self.table.a[(n, 0, m, 0)] = term_1 + term_2_3
            del term_1, term_2_3
            # The case j > 0
            for j in range(1, n + 1):
                term_1 = (
                    self.λ
                    * np.exp(-self.μ * self.τ)
                    * self.table.a[(n - 1, 0, m - 1, j - 1)]
                    / j
                )
                term_2_3 = self.λ * sum(
                    (
                        -self.table.a[(n - 1, 0, m, k)]
                        + self.table.a[(n - 1, 0, m - 1, k)] * np.exp(-self.μ * self.τ)
                    )
                    * math.factorial(k)
                    / math.factorial(j)
                    / (-self.μ) ** (1 + k - j)
                    for k in range(j, n)
                )
                self.table.a[(n, 0, m, j)] = term_1 + term_2_3
                del term_1, term_2_3
        else:
            # The case j = 0
            term_1 = (
                self.λ
                * np.exp(-self.μ * m * self.τ)
                * get_polynomial(
                    [
                        1
                        / math.factorial(q)
                        * sum(
                            self.table.A[(n - 1, l - q, m - 1, k)]
                            * math.factorial(k)
                            / self.μ ** (k + 1)
                            for k in range(n + l - q)
                        )
                        for q in range(l + 1)
                    ]
                )(self.μ * self.τ)
            )
            term_2_3 = self.λ * sum(
                sum(
                    self.table.a[(n - 1, l - q, m, k)] * self.table.ξ[(q, 0, m, k)]
                    + self.table.a[(n - 1, l - q, m - 1, k)]
                    * self.table.ζ[(q, 0, m - 1, k)]
                    for k in range(n + l - q)
                )
                for q in range(l + 1)
            )
            self.table.a[(n, l, m, 0)] = term_1 + term_2_3
            del term_1, term_2_3
            # The case j > 0
            for j in range(1, n + l + 1):
                term_1 = (
                    self.λ
                    * np.exp(-self.μ * self.τ)
                    * get_polynomial(
                        [
                            self.table.a[(n - 1, l - q, m - 1, j - 1)]
                            / j
                            / math.factorial(q)
                            for q in range(min(l + 1, n + l + 1 - j))
                        ]
                    )(self.μ * self.τ)
                )
                term_2_3 = self.λ * sum(
                    sum(
                        self.table.a[(n - 1, l - q, m, k)] * self.table.ξ[(q, j, m, k)]
                        + self.table.a[(n - 1, l - q, m - 1, k)]
                        * self.table.ζ[(q, j, m - 1, k)]
                        for k in range(j, n + l - q)
                    )
                    for q in range(l + 1)
                )
                self.table.a[(n, l, m, j)] = term_1 + term_2_3
                del term_1, term_2_3

    def fill_in_A(self, ind: tuple[int, int, int]):
        n, l, m = ind  # noqa: E741
        if n == 0:
            for j in range(n + l + 1):
                self.table.A[(0, l, 0, j)] = 0
        elif l == 0:
            # This case is actually included in the l != 0 case
            # But we first implemented this case to check the correctness of the formula
            # for its simplicity
            # The case j = 0
            term_1 = -self.λ * sum(
                self.table.A[(n - 1, 0, m - 1, k)]
                * math.factorial(k)
                / self.μ ** (k + 1)
                for k in range(n)
            )
            term_2_3 = (
                self.λ
                * np.exp(self.μ * m * self.τ)
                * sum(
                    (
                        self.table.a[(n - 1, 0, m, k)]
                        - self.table.a[(n - 1, 0, m - 1, k)] * np.exp(-self.μ * self.τ)
                    )
                    * math.factorial(k)
                    / (-self.μ) ** (k + 1)
                    for k in range(n)
                )
            )
            self.table.A[(n, 0, m, 0)] = term_1 + term_2_3
            del term_1, term_2_3
            # The case j > 0
            for j in range(1, n + 1):
                term_1 = -self.λ * sum(
                    self.table.A[(n - 1, 0, m - 1, k)]
                    * math.factorial(k)
                    / math.factorial(j)
                    / self.μ ** (k + 1 - j)
                    for k in range(j, n)
                )
                term_2_3 = (
                    self.λ
                    / j
                    * (
                        self.table.A[(n - 1, 0, m, j - 1)]
                        - self.table.A[(n - 1, 0, m - 1, j - 1)]
                    )
                )
                self.table.A[(n, 0, m, j)] = term_1 + term_2_3
                del term_1, term_2_3
        else:
            for j in range(n + l + 1):
                term_1 = -self.λ * get_polynomial(
                    [
                        1
                        / math.factorial(q)
                        * sum(
                            self.table.A[(n - 1, l - q, m - 1, k)]
                            * math.factorial(k)
                            / math.factorial(j)
                            / self.μ ** (1 + k - j)
                            for k in range(j, n + l - q)
                        )
                        for q in range(l + 1)
                    ]
                )(self.μ * self.τ)
                term_2_3 = self.λ * sum(
                    sum(
                        self.table.a[(n - 1, l - q, m, k)] * self.table.Ξ[(q, j, m, k)]
                        + self.table.a[(n - 1, l - q, m - 1, k)]
                        * self.table.Z[(q, j, m - 1, k)]
                        for k in range(n + l - q)
                    )
                    for q in range(j, l + 1)
                )
                term_4_5 = self.λ * sum(
                    sum(
                        self.table.A[(n - 1, l - q, m, k)] * self.table.η[(q, j, m, k)]
                        + self.table.A[(n - 1, l - q, m - 1, k)]
                        * self.table.θ[(q, j, m - 1, k)]
                        for k in range(max(j - q - 1, 0), n + l - q)
                    )
                    for q in range(l + 1)
                )
                self.table.A[(n, l, m, j)] = term_1 + term_2_3 + term_4_5
                del term_1, term_2_3, term_4_5

    def fill_in_all(self, ind: tuple[int, int]):
        """Fill in all the coefficients up to the given indice."""
        max_ind = Case(*ind)

        for n in range(self.table.max_indice.n + 1, max_ind.n + 1):
            # # The case l = 0
            # for m in range(n + 1):
            #     self.fill_in_a((n, 0, m))
            #     self.fill_in_A((n, 0, m))
            # # The case l >= 0
            if n == 0:
                for l in range(self.table.max_indice.l + 1, max_ind.l + 1):  # noqa: E741
                    for j in range(n + l + 1):
                        self.table.a[(0, l, 0, j)] = kronecker_delta(j, 0)
                        self.table.A[(0, l, 0, j)] = 0
            else:
                for l in range(self.table.max_indice.l + 1, max_ind.l + 1):  # noqa: E741
                    for m in range(n + 1):
                        for j in range(n + l + 1):
                            for r in range(j + 1):
                                self.fill_in_ksi((0, r, m, j))
                                self.fill_in_zeta((0, r, m, j))
                                self.fill_in_Ksi((0, r, m, j))
                                self.fill_in_Zeta((0, r, m, j))
                        for j in range(n + l + 2):
                            for r in range(j + 2):
                                self.fill_in_eta((0, r, m, j))
                                self.fill_in_theta((0, r, m, j))
                        for q in range(l + 1):
                            for r in range(q + 1):
                                self.fill_in_ksi((q, r, m, 0))
                                self.fill_in_zeta((q, r, m, 0))
                                self.fill_in_Ksi((q, r, m, 0))
                                self.fill_in_Zeta((q, r, m, 0))
                            for r in range(q + 2):
                                self.fill_in_eta((q, r, m, 0))
                                self.fill_in_theta((q, r, m, 0))
                        for q in range(1, l + 1):
                            for j in range(1, n + l - q + 1):
                                for r in range(q + j + 1):
                                    self.fill_in_ksi((q, r, m, j))
                                    self.fill_in_zeta((q, r, m, j))
                                    self.fill_in_Ksi((q, r, m, j))
                                    self.fill_in_Zeta((q, r, m, j))
                            for j in range(1, n + l - q + 1):
                                for r in range(q + j + 2):
                                    self.fill_in_eta((q, r, m, j))
                                    self.fill_in_theta((q, r, m, j))
                        self.fill_in_a((n, l, m))
                        self.fill_in_A((n, l, m))

        self.table.max_indice = max_ind

    def get_piecewise_monomial(self, m: int, j: int):
        def piecewise_monomial(t):
            return np.piecewise(t, [t < 0, t >= 0], [0, lambda t: t**j])

        return lambda t: piecewise_monomial(t - m * self.τ)

    def __call__(self, case: Case):
        """Compute the P probability of the event at the given time."""
        self.fill_in_all((case.n, case.l))

        def call(observation_time):
            term_1 = np.exp(-self.λ * observation_time) * sum(
                self.table.a[(case.n, case.l, m, j)]
                * self.get_piecewise_monomial(m, j)(observation_time)
                for m in range(case.n + 1)
                for j in range(case.n + case.l + 1)
            )
            term_2 = np.exp(-(self.μ + self.λ) * observation_time) * sum(
                self.table.A[(case.n, case.l, m, j)]
                * self.get_piecewise_monomial(m, j)(observation_time)
                for m in range(case.n + 1)
                for j in range(case.n + case.l + 1)
            )
            return term_1 + term_2

        return call
