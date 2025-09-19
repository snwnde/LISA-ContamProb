import ast
import logging
from typing import Unpack, Literal
import argparse
import pathlib
import matplotlib.pyplot as plt
import lovelyplots  # type: ignore[import]
import numpy as np
import joblib  # type: ignore[import]
import contamprob  # type: ignore[import]


def set_plt_style(
    lovelyplot_style: list[str] = ["paper", "colors5", "use_tex"], rcParams: dict = {}
):
    """Set the plotting style for matplotlib."""
    plt.style.use(lovelyplot_style)
    plt.rcParams.update(rcParams)


del lovelyplots


log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Compare simulation and analytical "
    "approximation of contamination process."
)


parser = argparse.ArgumentParser(
    description="Compare simulation and analytical "
    "approximation of contamination process."
)
parser.add_argument(
    "--observation_time",
    type=float,
    default=365,
    help="Observation time in days.",
)
parser.add_argument(
    "--max_k",
    type=int,
    default=-1,
    help="Cutoff for the degree of analytical solution. Default -1 means no cutoff.",
)
parser.add_argument(
    "--ctmn_population",
    type=str,
    help="Contamination population, constant, exponential or uniform.",
)
parser.add_argument(
    "--ctmn_scenario",
    type=str,
    help="Contamination scenario, constant, merged or reset.",
)
parser.add_argument(
    "--critical_value",
    type=float,
    help="Critical value for the contaminations.",
)
parser.add_argument(
    "--x_lim",
    type=float,
    default=4,
)
parser.add_argument(
    "--y_lim",
    type=float,
    default=9_000,
)
parser.add_argument(
    "--x_length",
    type=int,
    default=500,
)
parser.add_argument(
    "--y_length",
    type=int,
    default=500,
)
parser.add_argument(
    "--save_path",
    type=str,
    default="figures",
    help="Path to save the figures.",
)
parser.add_argument(
    "--plt_style",
    type=str,
    default="['paper', 'colors5', 'use_tex']",
    help="Lovelyplot style to use.",
)
parser.add_argument(
    "--plt_rcparams",
    type=str,
    default="{}",
    help="Matplotlib rcParams to update, in dictionary format.",
)
parser.add_argument(
    "--plt_rasterized",
    action="store_true",
    help="Save figures with rasterized=True.",
)
parser.add_argument(
    "--save_data",
    action="store_true",
    help="Save the data used to make the figures.",
)
parser.add_argument(
    "--from_file",
    type=str,
    default="",
    help="Path to a .npz file containing the data to plot.",
)


def _get_scenario(
    ctmn_scenario: Literal["constant", "merged", "reset"],
):
    if ctmn_scenario == "constant":
        return "constant_period"
    elif ctmn_scenario == "merged":
        return "merged_interval"
    elif ctmn_scenario == "reset":
        return "reset_interval"
    else:
        raise ValueError(
            f"Invalid contamination scenario: {ctmn_scenario}. "
            "Must be one of: constant, merged, reset."
        )


def _get_ctmn_proc(
    ctmn_population: Literal["constant", "exponential", "uniform"],
    ctmn_scenario: Literal["constant", "merged", "reset"],
    ctmn_rate: float,
    ctmn_population_param: float,
):
    scenario = _get_scenario(ctmn_scenario)
    ctmn_param = ctmn_population_param
    if ctmn_population == "constant":
        return contamprob.ContaminationProcess(
            contamprob.PoissonProcess(ctmn_rate),
            contamprob.SingletonPopulation(ctmn_param),
            scenario=scenario,
        )
    elif ctmn_population == "exponential":
        return contamprob.ContaminationProcess(
            contamprob.PoissonProcess(ctmn_rate),
            contamprob.ExponentialDistribution.from_scale(ctmn_param),
            scenario=scenario,
        )
    elif ctmn_population == "uniform":
        return contamprob.ContaminationProcess(
            contamprob.PoissonProcess(ctmn_rate),
            contamprob.UniformDistribution(ctmn_param),
            scenario=scenario,
        )


def _decide_len_unit(length: float):
    day = 86400
    hour = 3600
    minute = 60
    if length > day / day:
        unit = r"\unit{\day}"
        convert = day / day
    elif length > hour / day:
        unit = r"\unit{\hour}"
        convert = day / hour
    elif length > minute / day:
        unit = r"\unit{\minute}"
        convert = day / minute
    else:
        unit = r"\unit{\second}"
        convert = day
    return unit, convert


def _get_y_label(
    ctmn_population: Literal["constant", "exponential", "uniform"],
):
    if ctmn_population == "constant":
        return r"$\tau$"
    elif ctmn_population == "exponential":
        return r"$\tau_\text{mean}$"
    elif ctmn_population == "uniform":
        return r"$\tau_\text{max}$"
    else:
        raise ValueError(
            f"Invalid contamination population: {ctmn_population}. "
            "Must be one of: constant, exponential, uniform."
        )


class SurvivalFunctionEval:
    def __init__(
        self,
        critical_value: float,
        observation_time: float,
        **kwargs: Unpack[contamprob.ApproxConfig],
    ):
        self.critical_value = critical_value
        self.observation_time = observation_time
        self.kwargs = kwargs

    def __call__(self, ctmn_rate: float, ctmn_period: float) -> float:
        ctmn_proc = _get_ctmn_proc(
            ctmn_population=args.ctmn_population,
            ctmn_scenario=args.ctmn_scenario,
            ctmn_rate=ctmn_rate,
            ctmn_population_param=ctmn_period,
        )
        approx = contamprob.NormalApproximation(ctmn_proc, **self.kwargs)
        return approx(self.observation_time).sf(self.critical_value)


def meshgrid_vectorize(func):
    def wrapper(x, y):
        # Ensure inputs are numpy arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        # Create a meshgrid
        x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

        # Flatten the meshgrid for parallel processing
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()

        # Apply the function in parallel
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(func)(x_val, y_val) for x_val, y_val in zip(x_flat, y_flat)
        )

        # Reshape the results back to the original grid shape
        return np.array(results).reshape(x_grid.shape)

    return wrapper


if __name__ == "__main__":
    contamprob.logger.init_logger(
        "contamprob.approximation", level_console=logging.INFO
    )
    contamprob.logger.init_logger(__name__, level_console=logging.INFO)

    args = parser.parse_args()
    log.info(f"Arguments: {args}")

    set_plt_style(
        lovelyplot_style=ast.literal_eval(args.plt_style),
        rcParams=ast.literal_eval(args.plt_rcparams),
    )

    plt_rasterized = args.plt_rasterized

    save_path = pathlib.Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    sf_eval = SurvivalFunctionEval(
        critical_value=args.critical_value,
        observation_time=args.observation_time,
        max_k=args.max_k,
        self_ctmn=False,
        prob_method="by_hand",
    )
    vec_sf_eval = meshgrid_vectorize(sf_eval)

    name = (
        f"contour_{args.ctmn_population}_{args.ctmn_scenario}"
        + f"_{args.critical_value}"
        + f"_{args.x_lim}_{args.y_lim}"
    )

    rates = np.linspace(0.1, args.x_lim, args.x_length)
    params = (
        np.linspace(100, args.y_lim, args.y_length) / 86400
    )  # convert seconds to days

    if not args.from_file:
        log.info("Calculating contour data...")
        sf_vals = vec_sf_eval(rates, params)
    else:
        log.info(f"Loading contour data from {args.from_file}...")
        data = np.load(args.from_file)
        rates = data["rates"]
        params = data["params"]
        observation_time = data["observation_time"].item()
        critical_value = data["critical_value"].item()
        sf_vals = data["sf_vals"]

    if args.save_data:
        np.savez(
            save_path / f"{name}.npz",
            rates=rates,
            params=params,
            sf_vals=sf_vals,
            observation_time=args.observation_time,
            critical_value=args.critical_value,
        )

    levels = np.linspace(np.min(sf_vals), np.max(sf_vals), 100)

    y_unit, y_convert = _decide_len_unit(np.max(params))
    fig, ax = plt.subplots()
    contour = ax.contourf(
        rates, params * y_convert, sf_vals, levels=levels, cmap="berlin"
    )
    fig.colorbar(
        contour,
        label=rf"Probability of $T_\text{{ctmn}} \geqslant {args.critical_value} \unit{{\day}}$",
    )

    ax.set_xlabel(r"$\lambda$" + r" (\unit{\per\day})")
    ax.set_ylabel(_get_y_label(args.ctmn_population) + f" ({y_unit})")
    ax.set_rasterized(plt_rasterized)

    fig.savefig(save_path / f"{name}.pdf")
