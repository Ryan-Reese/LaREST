import json
from enum import IntEnum
from logging import Logger
from pathlib import Path

from larest.exceptions import NoResultsError
from larest.parsers import parse_best_rdkit_conformer


class PipelineStage(IntEnum):
    RDKIT = 1
    CREST_CONFGEN = 2
    CENSO = 3
    CREST_ENTROPY = 4
    FINISH = 5


def restore_results(
    results: dict[str, dict[str, float | None]],
    dir_path: Path,
    logger: Logger,
) -> tuple[dict[str, dict[str, float | None]], PipelineStage]:
    checkpoint_flag: bool = True
    results, checkpoint_flag = _load_rdkit_results(
        results,
        dir_path,
        logger,
    )
    if not checkpoint_flag:
        return results, PipelineStage.RDKIT
    results, checkpoint_flag = _load_crest_confgen_results(
        results,
        dir_path,
        logger,
    )
    if not checkpoint_flag:
        return results, PipelineStage.CREST_CONFGEN
    results, checkpoint_flag = _load_censo_results(
        results,
        dir_path,
        logger,
    )
    if not checkpoint_flag:
        return results, PipelineStage.CENSO
    results, checkpoint_flag = _load_crest_entropy_results(
        results,
        dir_path,
        logger,
    )
    if not checkpoint_flag:
        return results, PipelineStage.CREST_ENTROPY
    results, checkpoint_flag = _load_final_results(
        results,
        dir_path,
        logger,
    )
    return results, PipelineStage.FINISH


def _load_rdkit_results(
    results: dict[str, dict[str, float | None]],
    dir_path: Path,
    logger: Logger,
) -> tuple[dict[str, dict[str, float | None]], bool]:
    xtb_rdkit_results_file: Path = Path(
        dir_path,
        "xtb",
        "rdkit",
        "results.csv",
    )

    if xtb_rdkit_results_file.exists():
        try:
            best_rdkit_conformer_results: dict[str, float | None] = (
                parse_best_rdkit_conformer(
                    xtb_rdkit_results_file,
                )
            )
            del best_rdkit_conformer_results["conformer_id"]
            results |= {
                "rdkit": best_rdkit_conformer_results,
            }
        except Exception as err:
            logger.exception(err)
            logger.exception(
                f"Failed to load pre-existing rdkit results from {xtb_rdkit_results_file}",
            )
            return results, False
        else:
            logger.info(
                f"Loaded pre-existing rdkit results from {xtb_rdkit_results_file}",
            )
            logger.debug(
                f"Pre-existing rdkit results:\n {sorted(best_rdkit_conformer_results.items())}",
            )
            return results, True
    else:
        logger.info(
            f"No pre-existing rdkit results detected in {xtb_rdkit_results_file}",
        )
        logger.info("Continuing by running pipeline...")

        return results, False


def _load_crest_confgen_results(
    results: dict[str, dict[str, float | None]],
    dir_path: Path,
    logger: Logger,
) -> tuple[dict[str, dict[str, float | None]], bool]:
    xtb_crest_confgen_results_file: Path = Path(
        dir_path,
        "xtb",
        "crest",
        "results.json",
    )

    if xtb_crest_confgen_results_file.exists():
        try:
            with open(xtb_crest_confgen_results_file) as fstream:
                xtb_crest_confgen_results: dict[str, float | None] = json.load(
                    fstream,
                )
                results |= {"crest": xtb_crest_confgen_results}

        except Exception as err:
            logger.exception(err)
            logger.exception(
                f"Failed to load pre-existing crest_confgen results from {xtb_crest_confgen_results_file}",
            )
            return results, False
        else:
            logger.info(
                f"Loaded pre-existing crest_confgen results from {xtb_crest_confgen_results_file}",
            )
            logger.debug(
                f"Pre-existing crest_confgen results:\n {sorted(xtb_crest_confgen_results.items())}",
            )
            return results, True
    else:
        logger.info(
            f"No pre-existing crest_confgen results detected in {xtb_crest_confgen_results_file}",
        )
        logger.info("Continuing by running pipeline...")

        return results, False


def _load_censo_results(
    results: dict[str, dict[str, float | None]],
    dir_path: Path,
    logger: Logger,
) -> tuple[dict[str, dict[str, float | None]], bool]:
    censo_results_file: Path = Path(
        dir_path,
        "censo",
        "results.json",
    )

    if censo_results_file.exists():
        try:
            with open(censo_results_file) as fstream:
                censo_results: dict[str, dict[str, float | None]] = json.load(
                    fstream,
                )
                results |= censo_results

        except Exception as err:
            logger.exception(err)
            logger.exception(
                f"Failed to load pre-existing censo results from {censo_results_file}",
            )
            return results, False
        else:
            logger.info(
                f"Loaded pre-existing censo results from {censo_results_file}",
            )
            logger.debug(
                f"Pre-existing censo results:\n {censo_results}",
            )
            return results, True
    else:
        logger.info(
            f"No pre-existing censo results detected in {censo_results_file}",
        )
        logger.info("Continuing by running pipeline...")

        return results, False


def _load_crest_entropy_results(
    results: dict[str, dict[str, float | None]],
    dir_path: Path,
    logger: Logger,
) -> tuple[dict[str, dict[str, float | None]], bool]:
    crest_entropy_results_file: Path = Path(
        dir_path,
        "crest_entropy",
        "results.json",
    )

    if crest_entropy_results_file.exists():
        try:
            with open(crest_entropy_results_file) as fstream:
                crest_entropy_results: dict[str, float | None] = json.load(
                    fstream,
                )
                censo_corrected_results: dict[str, float | None] = results[
                    "3_REFINEMENT"
                ].copy()
                if (censo_corrected_results["S"] is None) or (
                    crest_entropy_results["S_total"] is None
                ):
                    raise NoResultsError(
                        "Failed to apply CREST entropy correction to CENSO results",
                    )
                censo_corrected_results["S"] += crest_entropy_results["S_total"]
                results |= {"censo_corrected": censo_corrected_results}

        except Exception as err:
            logger.exception(err)
            logger.exception(
                f"Failed to load pre-existing crest_entropy results from {crest_entropy_results_file}",
            )
            return results, False
        else:
            logger.info(
                f"Loaded pre-existing crest_entropy results from {crest_entropy_results_file}",
            )
            logger.debug(
                f"Pre-existing crest_entropy results:\n {crest_entropy_results}",
            )
            return results, True
    else:
        logger.info(
            f"No pre-existing crest_entropy results detected in {crest_entropy_results_file}",
        )
        logger.info("Continuing by running pipeline...")

        return results, False


def _load_final_results(
    results: dict[str, dict[str, float | None]],
    dir_path: Path,
    logger: Logger,
) -> tuple[dict[str, dict[str, float | None]], bool]:
    final_results_file: Path = Path(dir_path, "results.json")

    if final_results_file.exists():
        try:
            with open(final_results_file) as fstream:
                final_results: dict[str, dict[str, float | None]] = json.load(fstream)
                results |= final_results
        except Exception as err:
            logger.exception(err)
            logger.exception(
                f"Failed to load pre-existing final results from {final_results_file}, will run LaREST pipeline",
            )
            return results, False
        else:
            logger.info(
                f"Loaded pre-existing final results from {final_results_file}",
            )
            logger.debug(f"Pre-existing final results:\n {results}")
            return results, True
    else:
        logger.info(
            f"No pre-existing final results detected in {final_results_file}, will run LaREST pipeline",
        )
        logger.info("Results will be generated for molecule")

        return results, False
