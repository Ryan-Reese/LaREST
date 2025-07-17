class PolymerBuildError(Exception):
    def __init__(self, *args: str) -> None:
        super().__init__(args)


class NoResultsError(Exception):
    def __init__(self, *args: str) -> None:
        super().__init__(args)
