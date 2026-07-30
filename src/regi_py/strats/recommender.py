class RecommenderMixin:
    """Mixin for strategies that can suggest moves to a human player.

    Unified contract: ``getRecommendedMoves(phase, combos)`` returns a list of
    ``Combo`` objects (a ranked subset of ``combos``, best first), NOT strings.
    Callers serialize each ``Combo`` via ``RegiEncoder`` (``combo_to_dict`` ->
    list of card dicts) for the wire / UI. Implemented by
    ``BruteSamplingStrategy``, ``MCTSExplorerStrategy``, the AZ/ADZ explorer and
    direct-net strategies.
    """

    def getRecommendedMoves(self, phase, combos):
        raise NotImplementedError("abstract")
