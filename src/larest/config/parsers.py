import argparse

# TODO: this needs to have a new file name/location


class CRESTParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="Step 2 (CREST)")
        self._setup()

    def _setup(self):
        self.add_argument(
            "-o",
            "--output",
            type=str,
            default="./output",
            help="Output directory for Step 2",
        )
        self.add_argument(
            "-c",
            "--config",
            type=str,
            default="./config",
            help="Config directory for Step 2",
        )
        self.add_argument(
            "-v",
            "--verbose",
            help="increase output verbosity",
            action="store_true",
        )


class XTBParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="Step 1 (xTB)")
        self._setup()

    def _setup(self):
        self.add_argument(
            "-o",
            "--output",
            type=str,
            default="./output",
            help="Output directory for Step 1",
        )
        self.add_argument(
            "-c",
            "--config",
            type=str,
            default="./config",
            help="Config directory for Step 1",
        )
        self.add_argument(
            "-v",
            "--verbose",
            help="increase output verbosity",
            action="store_true",
        )
