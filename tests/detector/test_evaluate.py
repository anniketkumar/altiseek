"""Matching and metrics on hand-built views."""

import math

import pytest

from altiseek.detector.evaluate import (
    ViewCounts,
    ViewResult,
    average_precision,
    best_f1_threshold,
    centre_distance,
    iou,
    match,
    size_recall,
    summarise,
    view_counts,
)
from altiseek.detector.infer import Detection
from altiseek.detector.views import Label

GT = Label(0, 10.0, 10.0, 20.0, 20.0)


def det(x0, y0, x1, y1, score, class_id=0):
    return Detection(class_id, score, x0, y0, x1, y1)


def test_iou_and_centre_distance():
    assert iou(GT, det(10, 10, 20, 20, 1.0)) == pytest.approx(1.0)
    assert iou(GT, det(15, 10, 25, 20, 1.0)) == pytest.approx(50 / 150)
    assert iou(GT, det(30, 30, 40, 40, 1.0)) == 0.0
    assert centre_distance(GT, det(13, 14, 23, 24, 1.0)) == pytest.approx(5.0)


def test_match_is_one_to_one_by_score_and_respects_class():
    dets = [det(10, 10, 20, 20, 0.6), det(11, 11, 21, 21, 0.9), det(10, 10, 20, 20, 0.95, 1)]
    scores, hits, used = match([GT], dets, 0)
    assert scores == [0.9, 0.6]
    assert hits == [True, False]  # the duplicate is a false positive
    assert used == [True]
    with pytest.raises(ValueError, match="exactly one"):
        match([GT], dets, 0, iou_min=None)


def test_centre_matching_accepts_tiny_offset_boxes_that_iou_rejects():
    tiny_gt = Label(0, 10.0, 10.0, 12.0, 12.0)
    off = det(11.5, 11.5, 13.5, 13.5, 0.8)  # IoU 1/7 but centres 2.1 px apart
    assert match([tiny_gt], [off], 0)[1] == [False]
    assert match([tiny_gt], [off], 0, iou_min=None, radius_px=3.0)[1] == [True]


def test_average_precision():
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)
    assert average_precision([0.9], [True], 2) == pytest.approx(51 / 101)
    assert average_precision([0.9, 0.8], [False, True], 1) == pytest.approx(0.5)
    assert math.isnan(average_precision([0.5], [False], 0))


def view(truth, detections, gsd=0.05):
    return ViewResult("B", gsd, 1, tuple(truth), tuple(detections))


def test_summarise_counts_recall_precision_and_false_positives():
    views = [
        view([GT], [det(10, 10, 20, 20, 0.9), det(50, 50, 60, 60, 0.7)]),
        view([GT], [det(10, 10, 20, 20, 0.2)]),
        view([], [det(5, 5, 9, 9, 0.8)]),
    ]
    s = summarise(views, 0, tau=0.5)
    assert (s.n_views, s.n_truth) == (3, 2)
    assert s.recall == pytest.approx(0.5)
    assert s.precision == pytest.approx(1 / 3)
    assert s.fp_per_view == pytest.approx(2 / 3)
    assert 0 < s.ap50 < 1


def test_false_positives_count_per_full_view_of_valid_pixels():
    stray = det(50, 50, 60, 60, 0.9)
    views = [
        ViewResult("B", 0.05, 1, (), (stray,), valid_share=0.5),
        ViewResult("B", 0.05, 2, (), (stray,), valid_share=1.0),
    ]
    s = summarise(views, 0, tau=0.5)
    assert s.valid_views == pytest.approx(1.5)
    assert s.fp_per_view == pytest.approx(2 / 1.5)


def test_view_counts_match_by_iou_and_by_centre():
    tiny = Label(0, 10.0, 10.0, 12.0, 12.0)
    off = det(11.5, 11.5, 13.5, 13.5, 0.8)  # IoU 1/7; centres 2.1 px = 0.53 m apart
    v = view([tiny], [off, det(40, 40, 42, 42, 0.1)], gsd=0.25)
    assert view_counts(v, 0, tau=0.5) == ViewCounts(1, tp=0, fp=1, tp_centre=1, fp_centre=0)


def test_size_recall_and_best_threshold():
    small = Label(0, 0.0, 0.0, 3.0, 3.0)
    views = [view([small, GT], [det(10, 10, 20, 20, 0.9), det(0, 0, 3, 3, 0.3)])]
    bins = {label: (n, r) for label, n, r in size_recall(views, 0, tau=0.5)}
    assert bins["0-4 px"] == (1, 0.0)
    assert bins["8-16 px"] == (1, 1.0)
    assert best_f1_threshold(views, 0) == pytest.approx(0.3)
