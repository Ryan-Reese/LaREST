import argparse
import logging
import logging.config
import os
import tomllib
from typing import Any


def get_logger(name: str, args: argparse.Namespace) -> logging.Logger:
    logging_config_file = os.path.join(args.config, "logging.toml")
    try:
        with open(logging_config_file, "rb") as fstream:
            log_config = tomllib.load(fstream)
    except Exception as err:
        print(err)
        print(f"Failed to load logging config from {logging_config_file}")
        raise

    try:
        log_config["handlers"]["file"]["filename"] = (
            f"{args.output}/larest.log"  # perhaps can change this in the future to take from the same input file
        )
        logging.config.dictConfig(log_config)
    except Exception as err:
        print(err)
        print(f"Failed to parse logging config: {log_config}")
        raise

    return logging.getLogger(name)


def get_config(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
    config_file = os.path.join(args.config, "step1.toml")
    try:
        with open(config_file, "rb") as fstream:
            config = tomllib.load(fstream)
    except Exception as err:
        logger.exception(err)
        logger.error(f"Failed to load Step 1 config from {config_file}")
        raise
    else:
        return config
