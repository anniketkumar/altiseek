"""Detector evaluation on views (CLAUDE.md §9): AP50, recall, precision and false positives per
view at an operating threshold, by GSD bin and object size.

Detections are matched to ground truth of the same class one-to-one, greedily by descending
score: by IoU >= 0.5 for AP50, recall and precision, and by centre distance (default 1 m on the
ground) as a second criterion, because IoU is unstable for objects a few pixels wide. AP is the
COCO 101-point interpolated area under the precision-recall curve. Views may be partly fill
(D23), so false positives are counted per full view of valid pixels.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from altiseek.detector.infer import Detection
from altiseek.detector.views import Label

#: Object size bins by sqrt(width x height) in view pixels.
SIZE_EDGES_PX = (0.0, 4.0, 8.0, 16.0, 32.0, math.inf)


@dataclass(frozen=True)
class ViewResult:
    """Ground truth and detections of one evaluation view."""

    split: str
    gsd_mpp: float
    image_id: int
    truth: tuple[Label, ...]
    detections: tuple[Detection, ...]
    #: Share of the view's pixels with valid source; the rest is fill.
    valid_share: float = 1.0


def iou(a: Label | Detection, b: Label | Detection) -> float:
    """Intersection over union of two boxes."""
    w = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    h = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (b.y_max - b.y_min)
    return inter / (union - inter)


def centre_distance(a: Label | Detection, b: Label | Detection) -> float:
    """Distance between box centres, in pixels."""
    return math.hypot(
        (a.x_min + a.x_max - b.x_min - b.x_max) / 2, (a.y_min + a.y_max - b.y_min - b.y_max) / 2
    )


def match(
    truth: Sequence[Label],
    detections: Sequence[Detection],
    class_id: int,
    *,
    iou_min: float | None = 0.5,
    radius_px: float | None = None,
) -> tuple[list[float], list[bool], list[bool]]:
    """Greedy one-to-one matching of one class's detections (by descending score) to its truth.

    Exactly one of ``iou_min`` and ``radius_px`` must be given. Returns the detection scores in
    matching order, whether each detection is a true positive, and whether each truth box
    (in input order, this class only) was matched.
    """
    if (iou_min is None) == (radius_px is None):
        raise ValueError("give exactly one of iou_min and radius_px")
    gts = [t for t in truth if t.class_id == class_id]
    radius = math.inf if radius_px is None else radius_px
    dets = sorted((d for d in detections if d.class_id == class_id), key=lambda d: -d.score)
    used = [False] * len(gts)
    scores, hits = [], []
    for det in dets:
        best, best_k = (-1.0, -1) if iou_min is not None else (math.inf, -1)
        for k, gt in enumerate(gts):
            if used[k]:
                continue
            if iou_min is not None:
                value = iou(det, gt)
                if value >= iou_min and value > best:
                    best, best_k = value, k
            else:
                value = centre_distance(det, gt)
                if value <= radius and value < best:
                    best, best_k = value, k
        if best_k >= 0:
            used[best_k] = True
        scores.append(det.score)
        hits.append(best_k >= 0)
    return scores, hits, used


def average_precision(scores: Sequence[float], hits: Sequence[bool], n_truth: int) -> float:
    """COCO-style 101-point interpolated AP over pooled detections (nan without truth)."""
    if n_truth == 0:
        return math.nan
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    tp = np.asarray(hits, dtype=np.float64)[order]
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1.0 - tp)
    recall = cum_tp / n_truth
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # make it monotone
    levels = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, levels, side="left")
    return float(
        np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0).mean()
    )


@dataclass(frozen=True)
class ViewCounts:
    """One view's counts at threshold ``tau``: truth boxes, and true and false positives by IoU
    and by centre matching."""

    n_truth: int
    tp: int
    fp: int
    tp_centre: int
    fp_centre: int


def view_counts(view: ViewResult, class_id: int, tau: float, radius_m: float = 1.0) -> ViewCounts:
    """Counts of ``class_id`` at ``tau``; centre matching within ``radius_m`` on the ground."""
    above = [d for d in view.detections if d.score >= tau]
    _, hits, _ = match(view.truth, above, class_id)
    radius_px = radius_m / view.gsd_mpp
    _, hits_c, _ = match(view.truth, above, class_id, iou_min=None, radius_px=radius_px)
    return ViewCounts(
        n_truth=sum(t.class_id == class_id for t in view.truth),
        tp=sum(hits),
        fp=len(hits) - sum(hits),
        tp_centre=sum(hits_c),
        fp_centre=len(hits_c) - sum(hits_c),
    )


@dataclass(frozen=True)
class Summary:
    """Swimmer metrics over a set of views at threshold ``tau``. ``valid_views`` sums the views'
    valid shares; false positives are per full view of valid pixels."""

    n_views: int
    valid_views: float
    n_truth: int
    ap50: float
    recall: float
    precision: float
    fp_per_view: float
    recall_centre: float
    precision_centre: float


def summarise(
    views: Sequence[ViewResult], class_id: int, tau: float, radius_m: float = 1.0
) -> Summary:
    """AP50 over all scores; recall, precision and FP per valid view at ``tau`` (IoU and centre
    matching)."""
    scores: list[float] = []
    hits: list[bool] = []
    for v in views:
        s, h, _ = match(v.truth, v.detections, class_id)
        scores += s
        hits += h
    counts = [view_counts(v, class_id, tau, radius_m) for v in views]
    n_truth = sum(c.n_truth for c in counts)
    tp, fp = sum(c.tp for c in counts), sum(c.fp for c in counts)
    tp_c, fp_c = sum(c.tp_centre for c in counts), sum(c.fp_centre for c in counts)
    valid = sum(v.valid_share for v in views)
    return Summary(
        n_views=len(views),
        valid_views=valid,
        n_truth=n_truth,
        ap50=average_precision(scores, hits, n_truth),
        recall=tp / n_truth if n_truth else math.nan,
        precision=tp / (tp + fp) if tp + fp else math.nan,
        fp_per_view=fp / valid if valid else math.nan,
        recall_centre=tp_c / n_truth if n_truth else math.nan,
        precision_centre=tp_c / (tp_c + fp_c) if tp_c + fp_c else math.nan,
    )


def size_recall(
    views: Sequence[ViewResult], class_id: int, tau: float
) -> list[tuple[str, int, float]]:
    """Recall at ``tau`` per object-size bin: (bin label, number of truth boxes, recall)."""
    found = np.zeros(len(SIZE_EDGES_PX) - 1)
    total = np.zeros(len(SIZE_EDGES_PX) - 1)
    for v in views:
        gts = [t for t in v.truth if t.class_id == class_id]
        _, _, used = match(v.truth, [d for d in v.detections if d.score >= tau], class_id)
        for gt, ok in zip(gts, used, strict=True):
            size = math.sqrt((gt.x_max - gt.x_min) * (gt.y_max - gt.y_min))
            k = int(np.searchsorted(SIZE_EDGES_PX, size, side="right")) - 1
            total[k] += 1
            found[k] += ok
    labels = [f"{lo:g}-{hi:g} px" for lo, hi in itertools.pairwise(SIZE_EDGES_PX)]
    return [
        (label, int(n), float(f / n) if n else math.nan)
        for label, n, f in zip(labels, total, found, strict=True)
    ]


def best_f1_threshold(views: Sequence[ViewResult], class_id: int) -> float:
    """The score threshold that maximises F1 (IoU matching) over ``views``."""
    scores, hits = [], []
    n_truth = 0
    for v in views:
        s, h, _ = match(v.truth, v.detections, class_id)
        scores += s
        hits += h
        n_truth += sum(t.class_id == class_id for t in v.truth)
    if not scores or n_truth == 0:
        return 0.25
    order = np.argsort(-np.asarray(scores), kind="stable")
    sorted_scores = np.asarray(scores)[order]
    tp = np.cumsum(np.asarray(hits, dtype=np.float64)[order])
    precision = tp / np.arange(1, len(tp) + 1)
    recall = tp / n_truth
    f1: NDArray[np.float64] = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(sorted_scores[int(np.argmax(f1))])
