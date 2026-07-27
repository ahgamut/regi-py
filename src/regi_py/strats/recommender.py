class RecommenderMixin:

    def getRecommendedMoves(self, phase, combos):
        raise NotImplementedError("abstract")
