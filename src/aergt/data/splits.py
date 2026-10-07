from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from aergt.data.samples import GraphSample


def subjects(samples: Sequence[GraphSample]) -> list[str]:
    return sorted({sample.mouse_id for sample in samples})


def subset_by_subjects(samples: Sequence[GraphSample], selected: Sequence[str]) -> list[GraphSample]:
    selected_set = set(selected)
    return [sample for sample in samples if sample.mouse_id in selected_set]


def subject_level_folds(samples: Sequence[GraphSample], n_splits: int) -> list[tuple[int, list[str], list[str]]]:
    unique_subjects = subjects(samples)
    if len(unique_subjects) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique subjects for subject-level CV; "
            f"got {len(unique_subjects)}: {unique_subjects}"
        )

    rows = np.arange(len(unique_subjects))
    groups = np.asarray(unique_subjects)
    splitter = GroupKFold(n_splits=n_splits)

    folds: list[tuple[int, list[str], list[str]]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(rows, groups=groups), start=1):
        train_subjects = [unique_subjects[i] for i in train_idx]
        test_subjects = [unique_subjects[i] for i in test_idx]
        folds.append((fold_idx, train_subjects, test_subjects))
    return folds

