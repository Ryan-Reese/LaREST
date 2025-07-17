import argparse
import asyncio
import logging
from typing import Any

from larest.base import Initiator, Monomer, Polymer
from larest.parsers import LarestArgumentParser
from larest.setup import get_config, get_logger


def main(
    args: argparse.Namespace,
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    # run pipeline for initiator if ROR reaction
    if config["reaction"]["type"] == "ROR":
        logger.info("ROR Reaction detected, running pipeline for initiator")
        initiator: Initiator = Initiator(
            smiles=config["reaction"]["initiator"],
            args=args,
            config=config,
        )
        try:
            initiator.run()
        except Exception as err:
            logger.exception("Failed to run xTB for initiator")
            raise SystemExit(1) from err
        else:
            logger.info("Finished running xTB for initiator")


if __name__ == "__main__":
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
        main_logger: logging.Logger = get_logger(
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
