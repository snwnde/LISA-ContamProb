import logging
from typing import Unpack, Literal
import argparse
import pathlib
import matplotlib.pyplot as plt
import lovelyplots  # type: ignore[import]
import numpy as np
import contamprob  # type: ignore[import]

plt.style.use(["paper", "colors5", "use_tex"])
del lovelyplots


log = logging.getLogger(__name__)

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
    "--n_simulations",
    type=int,
    default=20_000,
    help="Number of simulations.",
)
parser.add_argument(
    "--max_k",
    type=int,
    default=-1,
    help="Cutoff for the degree of analytical solution. Default -1 means no cutoff.",
)
parser.add_argument(
    "--event_rate",
    type=float,
    default=1 / 30,
    help="Event rate. In days^-1.",
)
parser.add_argument(
    "--ctmn_rate",
    type=float,
    help="Contamination rate. In days^-1.",
)
parser.add_argument(
    "--ctmn_population",
    type=str,
    help="Contamination population, constant, exponential or uniform.",
)
parser.add_argument(
    "--ctmn_population_param",
    type=float,
    help="Contamination population parameter, "
    "for exponential and uniform distributions. In seconds.",
)
parser.add_argument(
    "--ctmn_scenario",
    type=str,
    help="Contamination scenario, constant, merged or reset.",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="figures",
    help="Path to save the figures.",
)


def _decide_len_unit(length: float):
    day = 86400
    hour = 3600
    minute = 60
    if length > day / day:
        unit = "days"
        convert = day / day
    elif length > hour / day:
        unit = "hours"
        convert = day / hour
    elif length > minute / day:
        unit = "minutes"
        convert = day / minute
    else:
        unit = "seconds"
        convert = day
    return unit, convert


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


def _get_save_name(
    ctmn_population: Literal["constant", "exponential", "uniform"],
    ctmn_scenario: Literal["constant", "merged", "reset"],
):
    scenario = _get_scenario(ctmn_scenario)
    if ctmn_population == "constant":
        return "constant_contamination_period", ""
    elif ctmn_population == "exponential":
        prefix = "exponential_contamination"
    elif ctmn_population == "uniform":
        prefix = "uniform_contamination"
    name1 = f"{prefix}_period_{scenario}"
    name2 = f"{prefix}_interval_length_{scenario}"
    return name1, name2


def _get_ctmn_proc(
    ctmn_population: Literal["constant", "exponential", "uniform"],
    ctmn_scenario: Literal["constant", "merged", "reset"],
    ctmn_rate: float,
    ctmn_population_param: float,
):
    scenario = _get_scenario(ctmn_scenario)
    ctmn_param = ctmn_population_param / 86400  # convert seconds to days
    if ctmn_population == "constant":
        return contamprob.ContaminationProcess(
            contamprob.PoissonProcess(ctmn_rate),
            contamprob.SingletonPopulation(ctmn_param),
            scenario="constant_period",
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


def get_simu_approx(
    ctmn_proc: contamprob.ContaminationProcess,
    event_proc: contamprob.PoissonProcess,
    k_cutoff: int | None = None,
    **approx_config: Unpack[contamprob.ApproxConfig],
):
    simulator = contamprob.Simulator(
        ctmn_proc, event_proc, collect_stats=True, k_cutoff=k_cutoff
    )
    approx = contamprob.NormalApproximation(ctmn_proc, **approx_config)
    return simulator, approx


def compare(
    simulator: contamprob.Simulator,
    approx: contamprob.NormalApproximation,
    observation_time: float,
    n_simulations: int,
):
    reduce_lim = 1_000

    def simulation_results():
        for idx in range(n_simulations):
            result = simulator(observation_time)
            reduced_result = result if idx < reduce_lim else None
            try:
                len_ctmn_intervals = len(result.ctmn_intervals.intervals)
            except AttributeError:
                len_ctmn_intervals = None
            yield (
                reduced_result,
                result.ctmn_length,
                len_ctmn_intervals,
                result.ctmn_int_categories,
            )

    reduced_results, ctmn_times, number_of_ctmn_intervals, ctmn_int_categories = map(
        list, zip(*simulation_results())
    )

    try:
        ctmn_interval_lengths = [
            interval.stop - interval.start
            for rslt in reduced_results[:reduce_lim]
            for interval in rslt.ctmn_intervals.intervals
        ]
    except AttributeError:
        pass
    else:
        # Merge categories from all simulations
        ctmn_stats: dict[int, int] = {}
        for stats in ctmn_int_categories:
            ctmn_stats |= stats

    unit, convert = _decide_len_unit(np.max(ctmn_times))
    fig1, ax1 = plt.subplots()
    density, bins, patches = ax1.hist(
        np.array(ctmn_times) * convert,
        bins="rice",
        alpha=1,
        label="Simulation",
        density=True,
    )
    del density, patches
    x_arr = np.linspace(np.min(bins), np.max(bins), 1000)
    ax1.plot(
        x_arr,
        approx(observation_time).pdf(x_arr / convert) / convert,
        label="Approximation",
    )
    ax1.set_xlabel(f"Contamination time ({unit})")
    ax1.set_ylabel("Probability density function")
    ax1.legend()

    fig2, ax2 = plt.subplots()
    try:
        unit, convert = _decide_len_unit(np.mean(ctmn_interval_lengths))
        density, bins, patches = ax2.hist(
            np.array(ctmn_interval_lengths) * convert,
            bins="rice",
            alpha=1,
            label="Simulation",
            density=True,
        )
    except UnboundLocalError:
        return fig1, None
    else:
        del density, patches
        try:
            ax2.plot(
                bins,
                np.array(approx.get_ctmn_interval_pdf(observation_time)(bins / convert))
                / convert,
                label="Analytical",
            )
        except NotImplementedError:
            pass
        ax2.set_xlabel(f"Contamination interval length ({unit})")
        ax2.set_ylabel("Probability density function")
        ax2.legend()
        log.info(
            f"sample number of ctmn intervals: {np.mean(number_of_ctmn_intervals)}"
        )
        log.info(
            f"ctmn times sample mean: {np.mean(ctmn_times)}, sample variance: {np.var(ctmn_times)}"
        )
        log.info(
            f"ctmn interval length sample mean: {np.mean(ctmn_interval_lengths)}, sample variance: {np.var(ctmn_interval_lengths)}"
        )
        log.info(f"ctmn interval nums per ctmn arrivals {ctmn_stats}")
        return fig1, fig2


if __name__ == "__main__":
    contamprob.logger.init_logger(
        "contamprob.approximation", level_console=logging.INFO
    )
    contamprob.logger.init_logger(__name__, level_console=logging.INFO)

    args = parser.parse_args()
    log.info(f"Arguments: {args}")
    event_proc = contamprob.PoissonProcess(args.event_rate)
    ctmn_proc = _get_ctmn_proc(
        args.ctmn_population,
        args.ctmn_scenario,
        args.ctmn_rate,
        args.ctmn_population_param,
    )
    simulator, approx = get_simu_approx(
        ctmn_proc,
        event_proc,
        prob_method="by_hand",
        self_ctmn=False,
        max_k=args.max_k,
    )

    fig1, fig2 = compare(
        simulator,
        approx,
        observation_time=args.observation_time,
        n_simulations=args.n_simulations,
    )
    save_path = pathlib.Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    name1, name2 = _get_save_name(args.ctmn_population, args.ctmn_scenario)
    fig1.savefig(save_path / f"{name1}.pdf")
    if fig2:
        fig2.savefig(save_path / f"{name2}.pdf")
