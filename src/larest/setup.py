import argparse
import logging
import logging.config
import tomllib
from pathlib import Path
from typing import Any

from larest.output import create_dir


def get_config(args: argparse.Namespace) -> dict[str, Any]:
    """Get config and default options for LaREST pipeline run

    Parameters
    ----------
    args : argparse.Namespace
        Input command-line arguments to LaREST, containing location of config dir

    Returns
    -------
    dict[str, Any]
        Final config for LaREST run, obtained by merging config with defaults

    """
    config_file: Path = Path(args.config) / "config.toml"
    defaults_file: Path = Path(args.config) / "default.toml"
    try:
        # load config and defaults
        with open(config_file, "rb") as fstream:
            config = tomllib.load(fstream)
        with open(defaults_file, "rb") as fstream:
            defaults = tomllib.load(fstream)

        # merge config and defaults
        def merge(
            dict1: dict[str, Any],
            dict2: dict[str, Any],
            sub_config: str = "",
        ) -> dict[str, Any]:
            for key, value in dict2.items():
                if key in dict1:
                    if isinstance(dict2[key], dict):
                        merge(dict1[key], dict2[key], sub_config + f"[{key}]")
                    else:
                        pass
                else:
                    dict1[key] = value
            return dict1

        config = merge(config, defaults)

    except Exception:
        print(f"Failed to load config from {config_file}")
        raise
    else:
        return config


def get_logger(
    name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> logging.Logger:
    """Get logger used during LaREST pipeline run

    Parameters
    ----------
    name : str
        Name passed to logger
    args : argparse.Namespace
        Input command-line arguments to LaREST, containing location of output dir
    config : dict[str, Any]
        Config for LaREST run

    Returns
    -------
    logging.Logger
        Logger object used to implement a logging system for LaREST

    """
    try:
        log_config: dict[str, Any] = config["logging"].copy()

        # set logging file location to output dir
        log_config["handlers"]["file"]["filename"] = Path(
            args.output,
            log_config["handlers"]["file"]["filename"],
        ).resolve()

        logging.config.dictConfig(log_config)
    except Exception:
        print(f"Failed to setup logging config from {config}")
        raise

    return logging.getLogger(name)


def create_censorc(
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Create censorc for LaREST run using specified config options

    Parameters
    ----------
    config : dict[str, Any]
        Config for LaREST run
    logger : logging.Logger
        Logger for LaREST run

    """
    try:
        censo_config: dict[str, Any] = config["censo"]
    except Exception as err:
        logger.exception(err)
        logger.exception(f"Failed to load censo config from {config}")
        raise

    censorc_file: Path = config["temp_config_dir"] / ".censo2rc"

    # write new .censo2rc using config options
    try:
        with open(censorc_file, "w") as fstream:
            for header, sub_config in censo_config.items():
                fstream.write(f"[{header}]\n")
                fstream.writelines(
                    f"{key} = {value}\n" for key, value in sub_config.items()
                )
                fstream.write("\n")
    except Exception as err:
        logger.exception(err)
        logger.exception(f"Failed to create censo config file from {censo_config}")
        raise
    else:
        logger.debug(f"Created censo config file at {censorc_file}")
