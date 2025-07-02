import logging
import logging.config
import os
import tomllib

from larest.config.parsers import CRESTParser
from larest.helpers.output import create_dir
from larest.helpers.parsers import parse_monomer_smiles, parse_most_stable_conformer


def run_crest_conformer(smiles, mol_type, dir_name, args, config, logger):
    # Create output dirs
    mol_dir = os.path.join(args.output, "step2", dir_name)
    create_dir(mol_dir, logger)
    pre_dir = os.path.join(mol_dir, "pre")
    create_dir(pre_dir, logger)
    post_dir = os.path.join(mol_dir, "post")
    create_dir(post_dir, logger)

    # find most stable conformer from
    step1_mol_dir = os.path.join(args.output, "step1", "dir_name")
    conformer_id = int(parse_most_stable_conformer(step1_mol_dir)["conformer_id"])
    subprocess.popen()  # cp the .xyz file to step2-pre directory


def main(args, config, logger):
    # get input monomer SMILES strings
    logger.info("Attempting to read input monomer smiles")
    monomer_smiles_list = parse_monomer_smiles(args, logger)
    logger.info("Finished reading input monomer smiles")

    # setup dir for step 2
    step2_dir = os.path.join(args.output, "step2")
    create_dir(step2_dir, logger)

    logger.info("Running CREST for initiator for ROR reaction")
    if config["reaction"]["type"] == "ROR":
        initiator_smiles = config["initiator"]["smiles"]
        initiator_dir_name = os.path.join("initiator", initiator_smiles)
        try:
            run_crest_conformer(
                smiles=initiator_smiles,
                mol_type="initiator",
                dir_name=initiator_dir_name,
                args=args,
                config=config,
                logger=logger,
            )
        except Exception:
            logger.error("Failed to run xTB for initiator")
            raise SystemExit(1)
        else:
            logger.info("Finished running xTB for initiator")


if __name__ == "__main__":
    # parse input arguments to get output and config dirs
    parser = CRESTParser()
    args = parser.parse_args()

    # load logging config from config dir
    logging_config_file = os.path.join(args.config, "logging.toml")
    try:
        with open(logging_config_file, "rb") as fstream:
            log_config = tomllib.load(fstream)
    except Exception as err:
        print(err)
        print(f"Failed to load logging config from {logging_config_file}")
        raise SystemExit(1)
    else:
        log_config["handlers"]["file"]["filename"] = f"{args.output}/larest.log"
        logging.config.dictConfig(log_config)
        logger = logging.getLogger("Step 2")
        logger.info("Logging config loaded")

    # load step 1 config
    step2_config_file = os.path.join(args.config, "step2.toml")

    try:
        with open(step2_config_file, "rb") as fstream:
            config = tomllib.load(fstream)
    except Exception as err:
        logger.exception(err)
        logger.error(f"Failed to load Step 2 config from {step2_config_file}")
        raise SystemExit(1)
    else:
        logger.info("Step 2 config loaded")

    # TODO: write assertions for config

    main(args, config, logger)
