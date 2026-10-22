"""Baseline estimators. Factories return fresh instances (GridSearchCV-safe)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .tree import StabilityTreeClassifier


def make_sklearn_cart(**overrides):
    params = dict(criterion="gini")
    params.update(overrides)
    return DecisionTreeClassifier(**params)


def make_custom_cart(**overrides):
    params = dict(penalty_lambda=0.0, n_bootstrap=10, random_state=0)
    params.update(overrides)
    return StabilityTreeClassifier(**params)


def make_spdt(**overrides):
    params = dict(penalty_lambda=0.1, n_bootstrap=10, random_state=0)
    params.update(overrides)
    return StabilityTreeClassifier(**params)


def make_random_forest(**overrides):
    params = dict(n_estimators=300, random_state=20261021, n_jobs=-1)
    params.update(overrides)
    return RandomForestClassifier(**params)
