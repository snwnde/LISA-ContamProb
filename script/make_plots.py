import ast
import logging
from typing import Unpack, Literal
import argparse
import pathlib
import matplotlib.pyplot as plt
import lovelyplots  # type: ignore[import]
import numpy as np
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
    "for exponential and uniform distributions. In seconds (not self-contamination) "
    "or hours (self-contamination).",
)
parser.add_argument(
    "--ctmn_scenario",
    type=str,
    help="Contamination scenario, constant, merged or reset.",
)
parser.add_argument(
    "--self_ctmn",
    action="store_true",
    help="Use self contamination.",
)
parser.add_argument(
    "--debug_approx",
    action="store_true",
    help="Plot debug approximation.",
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
    self_ctmn: bool,
):
    scenario = _get_scenario(ctmn_scenario)
    if not self_ctmn:
        ctmn_param = ctmn_population_param / 86400  # convert seconds to days
    else:
        ctmn_param = ctmn_population_param / 24  # convert seconds to days
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


def get_self_ctmn_simu_approx(
    ctmn_proc: contamprob.ContaminationProcess,
    **approx_config: Unpack[contamprob.ApproxConfig],
):
    simulator = contamprob.Simulator(ctmn_proc, use_julia=False, with_self_ctmn=True)
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

    sample_results = contamprob.approximation.PDFResults(
        np.mean(ctmn_interval_lengths),
        np.var(ctmn_interval_lengths),
    )

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

    label_prefix = "Approximation"
    label_suffix = rf" ($k\leqslant{args.max_k}$)" if args.max_k > 0 else ""

    ax1.plot(
        x_arr,
        approx(observation_time).pdf(x_arr / convert) / convert,
        label=label_prefix + label_suffix,
    )

    if args.debug_approx:
        debug_approx = contamprob.DebugNormalApproximation(
            simulator.ctmn_proc, sample_results, self_ctmn=False
        )
        debug_approx_dist = debug_approx(observation_time)
        ax1.plot(
            x_arr,
            debug_approx_dist.pdf(x_arr / convert) / convert,
            label="Approximation",
            linestyle="--",
        )

    ax1.set_xlabel(f"Contamination time ({unit})")
    ax1.set_ylabel("Probability density function")
    ax1.legend(loc="upper right")
    ax1.set_rasterized(plt_rasterized)

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
                label="Analytical" + label_suffix,
            )
        except NotImplementedError:
            pass

        ax2.set_xlabel(f"Contamination interval length ({unit})")
        ax2.set_ylabel("Probability density function")
        ax2.legend()
        ax2.set_rasterized(plt_rasterized)
        log.info(
            f"sample number of ctmn intervals: {np.mean(number_of_ctmn_intervals)}"
        )
        log.info(
            f"ctmn times sample mean: {np.mean(ctmn_times)}, sample variance: {np.var(ctmn_times)}"
        )
        log.info(
            f"ctmn interval length sample mean: {sample_results.mean}, sample variance: {sample_results.variance}"
        )
        log.info(f"ctmn interval nums per ctmn arrivals {ctmn_stats}")
        return fig1, fig2


def self_ctmn_compare(
    simulator: contamprob.Simulator,
    approx: contamprob.NormalApproximation,
    observation_time: float,
    n_simulations: int,
):
    def simulation_results():
        for _ in range(n_simulations):
            result = simulator(observation_time)
            victims = len(result.self_ctmn_results.victims)
            affected = len(
                np.unique(
                    np.concatenate(
                        [
                            result.self_ctmn_results.culprits,
                            result.self_ctmn_results.victims,
                        ]
                    )
                )
            )

            def get_empirical():
                for interval in result.ctmn_intervals.intervals:
                    num_arrivals = np.sum(
                        (result.ctmn_arrivals >= interval.start)
                        & (result.ctmn_arrivals < interval.stop)
                    )
                    num_ctmn = num_arrivals - 1
                    len_T = interval.stop - interval.start
                    yield num_ctmn, len_T

            pairs = list(get_empirical())

            yield victims, affected, pairs

    # Use generator expressions to compute statistics
    num_victims, _, list_of_pairs = map(list, zip(*simulation_results()))
    flat_pairs = [item for sublist in list_of_pairs for item in sublist]
    num_array = np.array([pair[0] for pair in flat_pairs])
    len_array = np.array([pair[1] for pair in flat_pairs])
    log.info(
        f"sample mean: {np.mean(num_victims)}, sample variance: {np.var(num_victims)}"
    )
    cov = np.cov(num_array, len_array)

    sample_results = contamprob.approximation.SelfCtmnPDFResults(
        np.mean(num_array),
        cov[0][0],
        cov[0][1],
        np.mean(len_array),
        cov[1][1],
    )

    log.info(
        f"sample num_mean: {sample_results.num_mean}, "
        f"sample num_var: {sample_results.num_variance}, "
        f"sample len_mean: {sample_results.interval_mean}, "
        f"sample len_var: {sample_results.interval_variance}, "
        f"sample len_num_cov: {sample_results.covariance}"
    )

    fig1, ax1 = plt.subplots()

    bins = np.arange(np.min(num_victims) - 1, np.max(num_victims)) + 0.5

    density, bins, patches = ax1.hist(
        num_victims,
        bins=bins,  # type: ignore[arg-type]
        alpha=1,
        label="Simulation",
        density=True,
    )

    del density, patches

    x_arr = np.linspace(np.min(num_victims), np.max(num_victims), 1000)

    approx_dist = approx(observation_time)
    ax1.plot(
        x_arr,
        approx_dist.pdf(x_arr),
        label=rf"Approximation ($k\leqslant{args.max_k}$)",
    )

    if args.debug_approx:
        debug_approx = contamprob.DebugNormalApproximation(
            simulator.ctmn_proc, sample_results, self_ctmn=True
        )
        debug_approx_dist = debug_approx(observation_time)
        ax1.plot(
            x_arr,
            debug_approx_dist.pdf(x_arr),
            label="Approximation",
            linestyle="--",
        )

    ax1.legend()
    ax1.set_xlabel("Number of self-contaminated signals")
    ax1.set_ylabel("Probability mass function")
    ax1.set_rasterized(plt_rasterized)
    return fig1


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

    self_ctmn = True if args.self_ctmn else False

    event_proc = contamprob.PoissonProcess(args.event_rate)
    ctmn_proc = _get_ctmn_proc(
        args.ctmn_population,
        args.ctmn_scenario,
        args.ctmn_rate,
        args.ctmn_population_param,
        self_ctmn=self_ctmn,
    )

    if not self_ctmn:
        simulator, approx = get_simu_approx(
            ctmn_proc,
            event_proc,
            prob_method="by_hand",
            self_ctmn=self_ctmn,
            max_k=args.max_k,
        )

        fig1, fig2 = compare(
            simulator,
            approx,
            observation_time=args.observation_time,
            n_simulations=args.n_simulations,
        )

        name1, name2 = _get_save_name(args.ctmn_population, args.ctmn_scenario)
        fig1.savefig(save_path / f"{name1}.pdf")
        if fig2:
            fig2.savefig(save_path / f"{name2}.pdf")

    else:
        simulator, approx = get_self_ctmn_simu_approx(
            ctmn_proc,
            prob_method="by_hand",
            self_ctmn=self_ctmn,
            max_k=args.max_k,
        )

        fig1 = self_ctmn_compare(
            simulator,
            approx,
            observation_time=args.observation_time,
            n_simulations=args.n_simulations,
        )

        fig1.savefig(
            save_path
            / (
                f"self_contamination_{args.ctmn_population}"
                + f"_obs_{args.observation_time}"
                + f"_rate_{args.ctmn_rate}"
                + f"_param_{args.ctmn_population_param}.pdf"
            )
        )
