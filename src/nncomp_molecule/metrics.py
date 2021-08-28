from typing import Sequence

import Levenshtein

import nncomp.registry as R


@R.CriterionRegistry.add
class LevenshteinDistance:
    def __call__(self, input: Sequence[str], target: Sequence[str]):
        distances = [
            Levenshtein.distance(_target, _input)
            for _input, _target in zip(input, target)
        ]
        return sum(distances) / len(distances)
