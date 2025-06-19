import os
import logging
import logging.config
import tomllib
from larest.config.parsers import CRESTParser


def find_xtb_lowest(args, config, logger):
    """
    Finds the most stable xyz files from xtb results and saves them in the specified output directory.
    """
    input_dir = os.path.join(args.output, "step1")
    output_dir = os.path.join(args.output, "step2")

    logger.info(f"Finding the most stable conformers")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")

    try:
        os.mkdir(output_dir)
        logger.info(f"Output directory {output_dir} created")
    except FileExistsError:
        logger.warning(f"Output directory {output_dir} already exists")
        os.makedirs(output_dir, exist_ok=True)

    found_directories = []
    # Go through all the input directories
    for dir_name in os.listdir(input_dir):
        current_dir = os.path.join(input_dir, dir_name)
        if os.path.isdir(current_dir):
            logging.info(f"Processing directory: {current_dir}")
            # Check if the target xyz files exist in the current directory
            target_files = [
                "xtbopt_low_init.xyz",
                "xtbopt_low_lactone.xyz",
                "xtbopt_low_polymer.xyz",
            ]
            files_found = [
                f for f in target_files if os.path.isfile(os.path.join(current_dir, f))
            ]

            if len(files_found) == len(target_files):
                found_directories.append(current_dir)
                logging.info(f"All target files found in directory: {current_dir}")

                # Extract the index, SMILES string, repeat units, and initiator from the original directory name
                parts = dir_name.split("_")
                index = parts[0]
                smiles = parts[1]
                conformers_repeatunits = "_".join(parts[2:4])
                initiator = parts[4]  # assuming the initiator is always in the 5th part

                # Create a sub-directory in the output directory with the extracted name
                target_dir = os.path.join(
                    output_dir, f"{index}_{smiles}_{conformers_repeatunits}_{initiator}"
                )
                os.makedirs(target_dir, exist_ok=True)

                # Copy the target files to the target directory
                for file_name in target_files:
                    shutil.copy(
                        os.path.join(current_dir, file_name),
                        os.path.join(target_dir, file_name),
                    )
                    logging.info(f"Copied {file_name} to {target_dir}")

    # Log the number of directories processed
    logging.info(f"Found xyz files in {len(found_directories)} reaction directories.")

    return output_dir


def main(args, config):
    logging.info("Starting Step 2")
    output_dir = find_xtb_lowest(xtb_result)
    run_crest_on_all(output_dir, solvent, threads, rthr, ewin, ethr, opt)

    # Extract data from CREST outputs and compile into DataFrame
    data = []
    for reaction_folder in os.listdir(output_dir):
        reaction_path = os.path.join(output_dir, reaction_folder)

        if not os.path.isdir(reaction_path):
            continue

        # Process only reaction folders directly under crest_output
        xtb_info = xtb_info_extract(reaction_folder)
        if xtb_info is None:
            continue

        for sub_subdir in ["crest_init", "crest_lactone", "crest_polymer"]:
            subdir_path = os.path.join(reaction_path, sub_subdir)
            if os.path.isdir(subdir_path):
                output_script_path = os.path.join(subdir_path, "output_script.out")
                if os.path.exists(output_script_path):
                    crest_info = crest_data_extract(output_script_path)
                    data.append({**xtb_info, **crest_info, "Subfolder": subdir_path})

    if not data:
        logging.error("No valid data extracted from CREST outputs.")
        return

    df = pd.DataFrame(data)

    # Run ROP calculations and generate combined output
    combined_df = rop_calc(df)
    combined_df.to_csv(
        os.path.join(output_dir, "final_combined_output.csv"), index=False
    )

    logging.info("Main process completed")

    # Move the crest.log file to the crest_output directory
    log_file = "crest.log"
    if os.path.exists(log_file):
        shutil.move(log_file, os.path.join(output_dir, log_file))
        logging.info(f"{log_file} has been moved to {output_dir}")


if __name__ == "__main__":

    # parse input arguments to get output and config dirs
    parser = CRESTParser()
    args = parser.parse_args()

    # load logging config from config dir
    try:
        with open(f"{args.config}/logging.toml", "rb") as fstream:
            log_config = tomllib.load(fstream)
            log_config["handlers"]["file"]["filename"] = f"{args.output}/larest.log"
    except Exception as e:
        raise e

    logging.config.dictConfig(log_config)
    logger = logging.getLogger(__name__)
    logger.info("logging config loaded")

    # load step 2 config or fail trying
    try:
        with open(f"{args.config}/step2.toml", "rb") as fstream:
            config = tomllib.load(fstream)
            logger.info("step 2 config loaded")
    except Exception as e:
        logger.exception(e)
        raise e

    main(args, config, logger)
