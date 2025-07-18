import argparse
import logging
import logging.config
import os
import tomllib
from pathlib import Path
from typing import Any

from larest.output import create_dir


def get_config(args: argparse.Namespace) -> dict[str, Any]:
    config_file: Path = Path(args.config) / "config.toml"
    try:
        with open(config_file, "rb") as fstream:
            config = tomllib.load(fstream)
    except Exception:
        print(f"Failed to load config from {config_file}")
        raise
    else:
        return config


def setup_logger(
    name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> logging.Logger:
    try:
        log_config: dict[str, Any] = config["logging"].copy()  # need to fix logging
        log_config["handlers"]["file"]["filename"] = (
            Path(args.output)
            .joinpath(log_config["handlers"]["file"]["filename"])
            .resolve()
        )
        logging.config.dictConfig(log_config)
    except Exception:
        print(f"Failed to setup logging config from {config}")
        raise

    return logging.getLogger(name)


def create_censorc(args: argparse.Namespace, logger: logging.Logger) -> None:
    config_file: Path = Path(args.config) / "config.toml"
    try:
        with open(config_file, "rb") as fstream:
            censo_config: dict[str, Any] = tomllib.load(fstream)["censo"]
    except Exception as err:
        logger.exception(err)
        logger.exception(f"Failed to load censo config from {config_file}")
        raise

    temp_config_dir: Path = Path(args.config) / "temp"
    create_dir(temp_config_dir, logger)
    censorc_file: Path = temp_config_dir / ".censo2rc"

    try:
        with open(censorc_file, "w") as fstream:
            for header in censo_config.keys():
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
