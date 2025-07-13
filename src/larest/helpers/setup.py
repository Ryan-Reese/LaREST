import argparse
import logging
import logging.config
import os
import tomllib
from typing import Any

from larest.helpers.output import create_dir


def get_logger(name: str, args: argparse.Namespace) -> logging.Logger:
    logging_config_file = os.path.join(args.config, "logging.toml")
    try:
        with open(logging_config_file, "rb") as fstream:
            log_config = tomllib.load(fstream)
    except Exception:
        print(f"Failed to load logging config from {logging_config_file}")
        raise

    try:
        log_config["handlers"]["file"]["filename"] = (
            f"{args.output}/larest.log"
            # perhaps can change this in the future to take from the same input file
        )
        logging.config.dictConfig(log_config)
    except Exception:
        print(f"Failed to specify logging output in {log_config}")
        raise

    return logging.getLogger(name)


def get_config(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
    config_file = os.path.join(args.config, "config.toml")
    try:
        with open(config_file, "rb") as fstream:
            config = tomllib.load(fstream)
    except Exception as err:
        logger.exception(err)
        logger.exception(f"Failed to load config from {config_file}")
        raise
    else:
        return config


def create_censorc(args: argparse.Namespace, logger: logging.Logger) -> None:
    config_file = os.path.join(args.config, "config.toml")
    try:
        with open(config_file, "rb") as fstream:
            censo_config = tomllib.load(fstream)["step2"]["censo"]
    except Exception as err:
        logger.exception(err)
        logger.exception(f"Failed to load censo config from {config_file}")
        raise

    temp_config_dir = os.path.join(args.config, "temp")
    create_dir(temp_config_dir, logger)

    censorc_file = os.path.join(temp_config_dir, ".censo2rc")
    try:
        with open(censorc_file, "w") as fstream:
            for header in censo_config:
                fstream.write(f"[{header}]\n")
                fstream.writelines(
                    f"{key} = {value}\n" for key, value in censo_config[header].items()
                )
                fstream.write("\n")
    except Exception as err:
        logger.exception(err)
        logger.exception(f"Failed to create censo config file from {censo_config}")
        raise
    else:
        logger.debug(f"Created censo config file at {censorc_file}")
