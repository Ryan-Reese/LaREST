class PolymerUnitError(Exception):
    def __init__(self, *args):
        super().__init__(args)


class PolymerLengthError(Exception):
    def __init__(self, *args):
        super().__init__(args)


class NoResultsError(Exception):
    def __init__(self, *args):
        super().__init__(args)
