"""Per-image pixel statistics: perceptual hashes for the leakage check and blackened-region flags.

ODv2 has no ignore boxes; ignored regions are blackened in the pixels (docs/data_notes.md), so
``has_ignore_regions`` (CLAUDE.md §7 step 1) has to be measured here. Images are analysed at a
quarter of their resolution (JPEG draft decoding), which keeps a streaming pass over the whole
dataset cheap.

The perceptual hash is the usual 64-bit DCT hash: a 32 x 32 luma thumbnail, its 2-D DCT-II, and
one bit per low-frequency coefficient (top-left 8 x 8) above their median. Blackened pixels are
replaced by the mean of the rest first, so the masks do not dominate the hash.
"""

import csv
import io
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from altiseek.data.horizon import BLACK_LUMA, LUMA
from altiseek.data.tables import read_csv

THUMB_PX = 32
HASH_BITS_SIDE = 8
#: Side of a "black block" in full-resolution pixels.
BLACK_BLOCK_PX = 32


@dataclass(frozen=True)
class ImageStats:
    """Pixel statistics of one image. Fractions are of all pixels at the analysis scale."""

    image_id: int
    #: Decoded full-resolution size, to check against the annotations.
    width_px: int
    height_px: int
    #: 64-bit DCT perceptual hash, as 16 hex digits.
    phash: str
    mean_luma: float
    #: Share of pixels with luma below ``BLACK_LUMA``.
    black_fraction: float
    #: Number of fully black ``BLACK_BLOCK_PX`` squares (grid-aligned).
    black_blocks: int

    @property
    def has_ignore_regions(self) -> bool:
        """A fully black 32 x 32 px square: a blackened ignore region (open water never is)."""
        return self.black_blocks > 0


def analyse(image_id: int, data: bytes) -> tuple[ImageStats, NDArray[np.float16]]:
    """Statistics and the masked 32 x 32 luma thumbnail of one encoded image."""
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        image.draft("RGB", (width // 4, height // 4))
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = rgb @ LUMA.astype(np.float32)
    black = luma < BLACK_LUMA
    block = max(1, round(BLACK_BLOCK_PX * luma.shape[1] / width))
    rows, cols = luma.shape[0] // block, luma.shape[1] // block
    blocks = black[: rows * block, : cols * block].reshape(rows, block, cols, block)
    thumb = masked_thumbnail(luma, black)
    stats = ImageStats(
        image_id=image_id,
        width_px=width,
        height_px=height,
        phash=phash_hex(thumb),
        mean_luma=float(luma.mean()),
        black_fraction=float(black.mean()),
        black_blocks=int(blocks.all(axis=(1, 3)).sum()),
    )
    return stats, thumb.astype(np.float16)


def masked_thumbnail(luma: NDArray[np.floating], black: NDArray[np.bool_]) -> NDArray[np.float32]:
    """32 x 32 box-filtered luma, with blackened pixels first set to the mean of the others."""
    filled = luma.astype(np.float32, copy=True)
    if black.any():
        filled[black] = float(luma[~black].mean()) if (~black).any() else 0.0
    image = Image.fromarray(filled)  # float32 -> mode "F"
    return np.asarray(image.resize((THUMB_PX, THUMB_PX), Image.Resampling.BOX), dtype=np.float32)


def phash_hex(thumb: NDArray[np.floating]) -> str:
    """64-bit DCT hash of a 32 x 32 thumbnail, as 16 hex digits."""
    basis = _dct_matrix(thumb.shape[0])
    low = (basis @ thumb @ basis.T)[:HASH_BITS_SIDE, :HASH_BITS_SIDE]
    bits = (low > np.median(low)).ravel()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


_POPCOUNT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def hamming(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.int64]:
    """Bit distance between every hash in ``a`` and every hash in ``b`` (``len(a) x len(b)``)."""
    xor = np.ascontiguousarray(a[:, None] ^ b[None, :])
    return _POPCOUNT16[xor.view(np.uint16)].reshape(*xor.shape, 4).sum(axis=-1, dtype=np.int64)


def hamming_aligned(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.int64]:
    """Bit distance between ``a[i]`` and ``b[i]`` for each ``i``."""
    xor = np.ascontiguousarray(a ^ b)
    return _POPCOUNT16[xor.view(np.uint16)].reshape(-1, 4).sum(axis=1, dtype=np.int64)


def hashes_to_uint64(hashes: Iterable[str]) -> NDArray[np.uint64]:
    return np.array([int(h, 16) for h in hashes], dtype=np.uint64)


def _dct_matrix(n: int) -> NDArray[np.float64]:
    """Orthonormal DCT-II basis: row k is frequency k."""
    k, i = np.mgrid[0:n, 0:n]
    basis = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    basis[0] /= np.sqrt(2.0)
    return basis


def run_scan(
    image_ids: Sequence[int],
    read: Callable[[int], bytes],
    out_dir: Path,
    *,
    workers: int,
    chunk_size: int,
    after_read: Callable[[int, bytes], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Analyse every image not yet in ``out_dir``; resumable. Returns how many were analysed.

    Results are appended to ``out_dir/stats.csv`` and ``out_dir/thumbs_<n>.npz`` one chunk at a
    time, so an interrupted scan loses at most one chunk and skips finished images on restart.
    ``read`` fetches an image's bytes; ``after_read`` (e.g. keeping a local copy) runs on them;
    ``progress(done, total)`` is called after each chunk with this run's counts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = out_dir / "stats.csv"
    names = [f.name for f in fields(ImageStats)]
    done = {s.image_id for s in read_csv(stats_csv, ImageStats)} if stats_csv.exists() else set()
    todo = [i for i in image_ids if i not in done]
    next_chunk = len(list(out_dir.glob("thumbs_*.npz")))

    def work(image_id: int) -> tuple[ImageStats, NDArray[np.float16]]:
        data = read(image_id)
        if after_read is not None:
            after_read(image_id, data)
        return analyse(image_id, data)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(todo), chunk_size):
            results = list(pool.map(work, todo[start : start + chunk_size]))
            np.savez(
                out_dir / f"thumbs_{next_chunk:05d}.npz",
                image_id=np.array([s.image_id for s, _ in results], dtype=np.int64),
                thumb=np.stack([t for _, t in results]),
            )
            next_chunk += 1
            # The CSV row is the commit marker: thumbnails are written first.
            new_file = not stats_csv.exists()
            with stats_csv.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if new_file:
                    writer.writerow(names)
                writer.writerows([getattr(s, n) for n in names] for s, _ in results)
            if progress is not None:
                progress(min(start + chunk_size, len(todo)), len(todo))
    return len(todo)


def load_thumbnails(out_dir: Path) -> dict[int, NDArray[np.float16]]:
    """All thumbnails written by :func:`run_scan`, keyed by image id."""
    thumbs: dict[int, NDArray[np.float16]] = {}
    for path in sorted(out_dir.glob("thumbs_*.npz")):
        with np.load(path) as chunk:
            thumbs.update(zip(chunk["image_id"].tolist(), chunk["thumb"], strict=True))
    return thumbs
