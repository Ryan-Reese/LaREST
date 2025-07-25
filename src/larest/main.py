import argparse
import logging
from typing import Any

from tqdm import tqdm

from larest.base import Monomer
from larest.parsers import LarestArgumentParser
from larest.setup import get_config, setup_logger


def main(
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    # run pipeline for each monomer
    logger.info("Running pipeline for monomers")
    for monomer_smiles in tqdm(
        config["reaction"]["monomers"],
        desc="Running pipeline for monomers",
    ):
        # run xtb for monomer
        monomer: Monomer = Monomer(
            smiles=monomer_smiles,
            args=args,
            config=config,
        )
        try:
            monomer.run()
        except Exception:
            logger.exception(f"Failed to run pipeline for monomer {monomer_smiles}")
            continue
        else:
            logger.info("Finished running pipeline for monomer")

        # run pipeline for initiator if ROR reaction
        if config["reaction"]["type"] == "ROR":
            logger.info("ROR Reaction detected, running pipeline for initiator")
            try:
                monomer.initiator.run()
            except Exception:
                logger.exception(
                    f"Failed to run pipeline for initiator {monomer.initiator.smiles}",
                )
                continue
            else:
                logger.info("Finished running pipeline for initiator")

        # run pipeline for each polymer length
        for polymer in tqdm(
            monomer.polymers,
            desc="Running pipeline for each polymer length",
        ):
            try:
                # if results already exist for this polymer length
                # prevents repeat computations
                if polymer.load_results():
                    continue
                polymer.run()
            except Exception:
                logger.exception(
                    f"Failed to run pipeline for polymer {monomer_smiles} (length: {polymer.length})",
                )
                continue
            else:
                logger.info("Finished running pipeline for polymer")

        monomer.compile_results()


def entry_point() -> None:
    # parse input arguments to get output and config dirs
    parser: LarestArgumentParser = LarestArgumentParser()
    args: argparse.Namespace = parser.parse_args()

    # load LaREST config
    try:
        config: dict[str, Any] = get_config(args=args)
    except Exception as err:
        raise SystemExit(1) from err

    # load logging config from config file
    try:
        main_logger: logging.Logger = setup_logger(
            name=__name__,
            args=args,
            config=config,
        )
    except Exception as err:
        raise SystemExit(1) from err
    else:
        main_logger.info("LaREST Initialised")
        for config_key, config_value in config.items():
            main_logger.debug(f"{config_key} config:\n{config_value}")

    # TODO: write assertions for config

    main(args=args, config=config, logger=main_logger)


if __name__ == "__main__":
    entry_point()
