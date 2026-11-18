"""Flexible, recomputable transit scoring: station profiles, cell/station tables and a score that can be changed after processing.

    transitlos.flex.scoring    ScoreParams (parameters, curve SHAPES, aggregation) and the numpy functions
    transitlos.flex.shapes     library of alternative shapes for the headway, speed and walk curves
    transitlos.flex.core       compiled core (C, loaded with ctypes): the cap, dominance, the multi-label street search, fast recomputation
    transitlos.flex.profiles   GTFS -> up to P (mode, headway, speed) profiles per parent station
    transitlos.flex.cells      stations of every H3 cell on a street network (uses UrbanAccessAnalyzer.csr_network)
    transitlos.flex.export     files for the browser (+ frontend/score.js, the identical scoring in JavaScript)

See docs/FLEX_SCORING.md.
"""
from .scoring import Curve, ScoreParams, DEFAULT  # noqa: F401
