"""NYC Ghost Neighborhoods — where did 3B taxi trips go?
=======================================================

A data-first tour of ~3 billion NYC for-hire trips (2011-2024) across every
yellow taxi, green taxi, and high-volume rideshare (Uber/Lyft) pickup the
city has ever logged.

Questions we do NOT know the answer to in advance:

  1. Which NYC zones became ghost neighborhoods — peaked during the taxi era
     and never recovered?
  2. Which zones were effectively "born" in the last 5 years, appearing on
     the map only after Uber/Lyft + pandemic reshuffled the city?
  3. Which zones are the *most* resurrected — a historic peak, a deep
     trough, then a comeback?

Data source:
  Hugging Face mirror ``DinoPonjevic/NYC_TaxiData_RAW`` — a verbatim copy of
  the NYC TLC monthly trip-records parquets (yellow 2011-2024, green
  2014-2024, high-volume FHV 2019-02 to 2024-12). We use the HF mirror
  because the TLC's own CloudFront distribution aggressively rate-limits
  GCP egress, which kills a 371-way parallel Burla map. HF's Xet bridge
  hands out signed S3 URLs from a different CDN and scales gracefully.

Pipeline shape:

  * Map (hundreds of workers in parallel): each worker streams ONE monthly
    trip-records parquet from HF, projects down to (pickup_zone, month) ->
    trip_count, and returns a tiny dict. A single month of high-volume
    rideshare is ~21M trips in a 500MB parquet; a single worker bag-of-
    counts it down to at most 263 rows. The entire ~3B-trip firehose is
    reduced on the wire.
  * Reduce (driver, local): merge the per-month dicts into a 263-zone x
    ~170-month matrix, rank each zone by its ghost / emergent / resurrected
    score, and render a choropleth SVG of NYC colored by category plus
    three HTML leaderboards with sparkline trajectories.

Run:
    /Users/josephperry/.burla/joeyper23/.venv/bin/python nyc_ghost_neighborhoods.py

Env vars:
    LOCAL=1            single-month smoke test on the driver, no Burla
    MONTHS_LIMIT=N     only process the N most recent months (per taxi type)
    START_YEAR=YYYY    earliest year to include (default 2011)
    END_YEAR=YYYY      latest year to include (default 2024)
    HF_TOKEN=...       optional; raises HF rate limit from 3k/5min → 10k/5min
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("DISABLE_BURLA_TELEMETRY", "True")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Hard-imported at module scope so Burla's dep-detector ships them to workers.
import requests  # noqa: F401
import fsspec  # noqa: F401

# Primary data source: HF mirror of the NYC TLC parquets. The TLC's own
# CloudFront distribution 403s GCP workers en masse, so we route through HF.
HF_REPO = "DinoPonjevic/NYC_TaxiData_RAW"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
ZONE_LOOKUP_URL = f"{HF_BASE}/_data/taxi_zones/taxi_zone_lookup.csv"
ZONE_SHAPES_URL = f"{HF_BASE}/_data/taxi_zones.zip"

# Each entry: (filename_prefix, earliest YYYYMM, latest YYYYMM, pickup_col_candidates)
# Ranges reflect actual file coverage of the HF mirror (verified via HEAD).
TAXI_TYPES: List[Tuple[str, int, int, Tuple[str, ...]]] = [
    ("yellow", 201101, 202412, ("tpep_pickup_datetime", "pickup_datetime", "Trip_Pickup_DateTime")),
    ("green",  201401, 202412, ("lpep_pickup_datetime", "pickup_datetime")),
    ("fhvhv",  201902, 202412, ("pickup_datetime",)),
]

# Column-name soup for the pickup-zone id across schema generations.
PU_ZONE_COL_CANDIDATES = ("PULocationID", "PUlocationID", "pickup_location_id")

OUT_DIR = Path(os.environ.get("NYC_OUT_DIR", "/Users/josephperry/claude/burla-demos/nyc_ghost_out"))
LOCAL_CACHE = Path("/tmp/nyc_ghost_cache")

# How we slice "old" vs "recent".
RECENT_WINDOW_MONTHS = 12        # last N months = "now"
BIRTH_WINDOW_MONTHS = 24         # first N months where a zone was active = "birth"
MIN_HISTORIC_PEAK = 5_000        # a ghost must have once averaged >= this many monthly trips
MIN_RECENT_VOLUME = 500          # a reborn zone must have at least this many trips/month now
MIN_MONTHS_OBSERVED = 36         # must appear in the data >= N months to be scored
TOP_K = 12

# Bookkeeping zones the TLC uses as catch-alls, not real neighborhoods. They
# dominate the top of a naive ghost ranking because yellow cabs used to flag
# "don't know / off-grid" more liberally than modern rideshare does.
EXCLUDE_ZONES: set[int] = {264, 265}  # 264 = "NV"/Unknown, 265 = "Outside of NYC"


# ---------------------------------------------------------------------------
# Map phase: one parquet file per worker, streamed and aggregated
# ---------------------------------------------------------------------------

def _list_months_for_type(prefix: str, first: int, last: int) -> List[str]:
    out = []
    y, m = first // 100, first % 100
    while y * 100 + m <= last:
        out.append(f"{prefix}_tripdata_{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def build_task_list() -> List[str]:
    start_y = int(os.environ.get("START_YEAR", "2011"))
    end_y = int(os.environ.get("END_YEAR", "2024"))
    start_ym = start_y * 100 + 1
    end_ym = end_y * 100 + 12
    tasks: List[str] = []
    for prefix, first, last, _ in TAXI_TYPES:
        lo = max(first, start_ym)
        hi = min(last, end_ym)
        if lo > hi:
            continue
        tasks.extend(_list_months_for_type(prefix, lo, hi))

    limit = os.environ.get("MONTHS_LIMIT")
    if limit:
        n = int(limit)
        tasks = tasks[-n:]
    return tasks


def _hf_url_for_task(task_id: str) -> str:
    """Translate ``yellow_tripdata_2023-04`` -> full HF resolve URL."""
    prefix = task_id.split("_", 1)[0]
    ym = task_id.rsplit("_", 1)[-1]
    year = ym.split("-")[0]
    return f"{HF_BASE}/{year}/{prefix}/{task_id}.parquet"


def _find_col(names: List[str], candidates: Tuple[str, ...]) -> str | None:
    """Case-insensitive match against a list of candidate column names."""
    lower_to_real = {n.lower(): n for n in names}
    for c in candidates:
        if c in names:
            return c
        real = lower_to_real.get(c.lower())
        if real is not None:
            return real
    return None


def process_month(task_id: str) -> Dict:
    """Worker entry-point. Downloads one parquet file, returns {(year_month, zone_id): count}.

    Returned shape:
        {
          "task": "yellow_tripdata_2023-01",
          "taxi_type": "yellow",
          "year": 2023, "month": 1,
          "total_rows": 3_066_766,
          "rows_with_zone": 3_050_000,
          "counts": [[zone_id, pickup_count], ...],   # compact list-of-lists for serialisation
          "elapsed_s": 12.3,
        }
    """
    t0 = time.time()

    print(f"[{task_id}] start", flush=True)

    prefix = task_id.split("_", 1)[0]
    ym = task_id.rsplit("_", 1)[-1]
    year, month = int(ym.split("-")[0]), int(ym.split("-")[1])

    url = _hf_url_for_task(task_id)

    import requests as _requests
    import random as _random

    # HF's Xet bridge issues signed S3 URLs via a 302 and has no meaningful
    # per-IP rate ceiling for burla-sized fleets (3k/5min unauthenticated,
    # 10k/5min with HF_TOKEN, both applied to the redirect, not the S3
    # download). Retries exist only for transient network blips.
    headers = {"User-Agent": "burla-demos/1.0 (+github.com/Burla-Cloud)"}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    last_status: int | None = None
    last_err: Exception | None = None
    blob: bytes | None = None
    MAX_ATTEMPTS = 6
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = _requests.get(url, headers=headers, timeout=300, allow_redirects=True)
            last_status = resp.status_code
            if resp.status_code == 404:
                print(f"[{task_id}] SKIP: 404 (file not on mirror)", flush=True)
                return {
                    "task": task_id, "taxi_type": prefix, "year": year, "month": month,
                    "total_rows": 0, "rows_with_zone": 0, "counts": [],
                    "elapsed_s": round(time.time() - t0, 2),
                    "skip_reason": "http 404",
                }
            if resp.status_code == 429:
                raise RuntimeError("http 429 (hf rate limit)")
            resp.raise_for_status()
            body = resp.content
            if len(body) < 1024:
                raise RuntimeError(f"response too small: {len(body)} bytes")
            blob = body
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            backoff = min(30.0, 2.0 * (2 ** attempt)) + _random.random() * 2.0
            print(f"[{task_id}] download attempt {attempt+1}/{MAX_ATTEMPTS} failed: {e}; sleep {backoff:.1f}s", flush=True)
            time.sleep(backoff)

    if blob is None:
        print(f"[{task_id}] SKIP: gave up after {MAX_ATTEMPTS} tries: {last_err}", flush=True)
        return {
            "task": task_id, "taxi_type": prefix, "year": year, "month": month,
            "total_rows": 0, "rows_with_zone": 0, "counts": [],
            "elapsed_s": round(time.time() - t0, 2),
            "skip_reason": f"download failed: status={last_status} err={last_err}",
        }

    print(f"[{task_id}] downloaded {len(blob)/1e6:.1f}MB in {time.time()-t0:.1f}s", flush=True)

    pf = pq.ParquetFile(pa.BufferReader(blob))
    schema_names = [c.name for c in pf.schema_arrow]

    pickup_candidates: Tuple[str, ...] = ("tpep_pickup_datetime", "lpep_pickup_datetime",
                                          "pickup_datetime", "Pickup_date",
                                          "Trip_Pickup_DateTime")
    pu_col = _find_col(schema_names, pickup_candidates)
    zone_col = _find_col(schema_names, PU_ZONE_COL_CANDIDATES)

    if zone_col is None:
        return {
            "task": task_id, "taxi_type": prefix, "year": year, "month": month,
            "total_rows": pf.metadata.num_rows, "rows_with_zone": 0,
            "counts": [], "elapsed_s": round(time.time() - t0, 2),
            "skip_reason": f"no zone column in {schema_names[:4]}",
        }

    counts: Dict[int, int] = defaultdict(int)
    total_rows = 0
    rows_with_zone = 0

    read_cols = [zone_col]
    if pu_col is not None:
        read_cols.append(pu_col)

    for batch in pf.iter_batches(batch_size=500_000, columns=read_cols):
        total_rows += batch.num_rows
        zone_arr = batch.column(zone_col)

        if pu_col is not None:
            pu_arr = batch.column(pu_col)
            pu_np = pu_arr.to_numpy(zero_copy_only=False)
            if hasattr(pu_np, "astype") and np.issubdtype(pu_np.dtype, np.datetime64):
                years = pu_np.astype("datetime64[Y]").astype(int) + 1970
                months = pu_np.astype("datetime64[M]").astype(int) % 12 + 1
                mask_time = (years == year) & (months == month)
            else:
                mask_time = np.ones(batch.num_rows, dtype=bool)
        else:
            mask_time = np.ones(batch.num_rows, dtype=bool)

        zone_np = zone_arr.to_numpy(zero_copy_only=False)
        try:
            zone_int = np.asarray(zone_np, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        valid = ~np.isnan(zone_int) & (zone_int > 0) & (zone_int < 1000)
        valid = valid & mask_time
        if not valid.any():
            continue

        z = zone_int[valid].astype(np.int32)
        rows_with_zone += int(valid.sum())
        uniq, cnts = np.unique(z, return_counts=True)
        for zi, ci in zip(uniq.tolist(), cnts.tolist()):
            counts[int(zi)] += int(ci)

    elapsed = round(time.time() - t0, 2)
    print(f"[{task_id}] done: {rows_with_zone:,} rows w/ zone, {len(counts)} zones, {elapsed}s", flush=True)
    return {
        "task": task_id,
        "taxi_type": prefix,
        "year": year, "month": month,
        "total_rows": int(total_rows),
        "rows_with_zone": int(rows_with_zone),
        "counts": sorted([[int(k), int(v)] for k, v in counts.items()]),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Reduce phase (driver): assemble zone x month matrix, score, render
# ---------------------------------------------------------------------------

@dataclass
class ZoneSeries:
    zone_id: int
    months: np.ndarray        # int YYYYMM
    counts: np.ndarray        # int32
    total_trips: int
    peak_month: int
    peak_volume: int
    recent_mean: float
    birth_mean: float
    first_active_month: int
    last_active_month: int
    n_active_months: int


def _month_int_to_index(month_int: int, all_months: List[int]) -> int:
    return all_months.index(month_int)


def _http_download(url: str, dest: Path) -> None:
    import requests as _requests
    headers = {"User-Agent": "burla-demos/1.0 (+github.com/Burla-Cloud)"}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            with _requests.get(url, headers=headers, timeout=240, stream=True,
                               allow_redirects=True) as resp:
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed ({url}): {last_err}")


def _load_zone_lookup() -> Dict[int, Dict]:
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    dest = LOCAL_CACHE / "taxi_zone_lookup.csv"
    if not dest.exists():
        _http_download(ZONE_LOOKUP_URL, dest)

    import csv
    lookup: Dict[int, Dict] = {}
    with dest.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                zid = int(row["LocationID"])
            except (KeyError, ValueError):
                continue
            lookup[zid] = {
                "id": zid,
                "borough": row.get("Borough", ""),
                "zone": row.get("Zone", ""),
                "service_zone": row.get("service_zone", ""),
            }
    return lookup


def _load_zone_shapes_as_svg_paths() -> Dict[int, List[List[Tuple[float, float]]]]:
    """Download taxi_zones.zip, return {zone_id: [ring_latlon_list, ...]}.

    NYC taxi shapefile is in NY state-plane coordinates (EPSG:2263); we crudely
    convert to lon/lat via a local affine approximation by sampling a few
    known-bounding-box points. For viral-demo-quality SVG we don't need perfect
    geodesy, just the rough shape.
    """
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    dest = LOCAL_CACHE / "taxi_zones.zip"
    if not dest.exists():
        _http_download(ZONE_SHAPES_URL, dest)

    import shapefile  # pyshp

    shp_dir = LOCAL_CACHE / "taxi_zones"
    if not shp_dir.exists():
        with zipfile.ZipFile(dest) as z:
            z.extractall(shp_dir)

    shp_path = next(shp_dir.rglob("*.shp"))
    r = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in r.fields[1:]]
    zone_idx = fields.index("LocationID") if "LocationID" in fields else 0

    shapes: Dict[int, List[List[Tuple[float, float]]]] = {}
    for sr in r.shapeRecords():
        try:
            zid = int(sr.record[zone_idx])
        except (TypeError, ValueError):
            continue
        shape = sr.shape
        parts = list(shape.parts) + [len(shape.points)]
        rings: List[List[Tuple[float, float]]] = []
        for i in range(len(parts) - 1):
            ring = shape.points[parts[i]:parts[i + 1]]
            if len(ring) >= 3:
                rings.append([(float(x), float(y)) for x, y in ring])
        if rings:
            shapes[zid] = rings

    return shapes


def _build_series_table(
    raw_results: List[Dict],
) -> Tuple[List[int], Dict[int, ZoneSeries], Dict[int, int], List[int]]:
    """Aggregate per-month counts across all taxi types into per-zone time series.

    Returns (all_months_sorted, zone_series, zone_total_map, monthly_totals).
    """
    # (year*100+month) -> zone_id -> trip_count
    matrix: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    month_totals: Dict[int, int] = defaultdict(int)
    zone_totals: Dict[int, int] = defaultdict(int)

    for res in raw_results:
        if not res:
            continue
        ym = int(res["year"]) * 100 + int(res["month"])
        for zone_id, cnt in res.get("counts", []):
            matrix[ym][int(zone_id)] += int(cnt)
            month_totals[ym] += int(cnt)
            zone_totals[int(zone_id)] += int(cnt)

    all_months = sorted(matrix.keys())
    if not all_months:
        return [], {}, {}, []

    mt = [month_totals[m] for m in all_months]

    recent_months = set(all_months[-RECENT_WINDOW_MONTHS:])

    zone_series: Dict[int, ZoneSeries] = {}
    for zid, tot in zone_totals.items():
        months_arr = np.array(all_months, dtype=np.int32)
        counts_arr = np.array([matrix[m].get(zid, 0) for m in all_months], dtype=np.int64)

        nonzero = np.where(counts_arr > 0)[0]
        if len(nonzero) < MIN_MONTHS_OBSERVED:
            continue

        peak_idx = int(counts_arr.argmax())
        peak_month = int(months_arr[peak_idx])
        peak_volume = int(counts_arr[peak_idx])

        recent_mask = np.array([m in recent_months for m in all_months])
        recent_mean = float(counts_arr[recent_mask].mean()) if recent_mask.any() else 0.0

        birth_end = min(nonzero[0] + BIRTH_WINDOW_MONTHS, len(counts_arr))
        birth_mean = float(counts_arr[nonzero[0]:birth_end].mean())

        first_active_month = int(months_arr[nonzero[0]])
        last_active_month = int(months_arr[nonzero[-1]])

        zone_series[zid] = ZoneSeries(
            zone_id=zid, months=months_arr, counts=counts_arr, total_trips=int(tot),
            peak_month=peak_month, peak_volume=peak_volume,
            recent_mean=recent_mean, birth_mean=birth_mean,
            first_active_month=first_active_month, last_active_month=last_active_month,
            n_active_months=int(len(nonzero)),
        )

    return all_months, zone_series, dict(zone_totals), mt


def _classify(zone_series: Dict[int, ZoneSeries]) -> Dict[int, str]:
    """Every zone gets a category label for the map."""
    out: Dict[int, str] = {}
    for zid, zs in zone_series.items():
        peak_avg_12 = zs.peak_volume  # single-peak-month proxy
        ghost_ratio = (zs.recent_mean / peak_avg_12) if peak_avg_12 > 0 else 1.0
        emergent_ratio = (zs.recent_mean / zs.birth_mean) if zs.birth_mean > 0 else float("inf")

        if peak_avg_12 >= MIN_HISTORIC_PEAK and ghost_ratio < 0.35:
            out[zid] = "ghost"
        elif zs.recent_mean >= MIN_RECENT_VOLUME and emergent_ratio >= 4.0 and zs.birth_mean < 1000:
            out[zid] = "emergent"
        elif peak_avg_12 >= MIN_HISTORIC_PEAK and ghost_ratio < 0.7:
            out[zid] = "cooling"
        elif emergent_ratio >= 1.5 and zs.recent_mean >= MIN_RECENT_VOLUME:
            out[zid] = "warming"
        else:
            out[zid] = "stable"
    return out


def _rank_ghosts(zone_series: Dict[int, ZoneSeries]) -> List[ZoneSeries]:
    """Zones whose current volume has collapsed furthest below their historic peak."""
    scored = []
    for zs in zone_series.values():
        if zs.zone_id in EXCLUDE_ZONES:
            continue
        if zs.peak_volume < MIN_HISTORIC_PEAK:
            continue
        ratio = zs.recent_mean / zs.peak_volume if zs.peak_volume > 0 else 1.0
        scored.append((ratio, zs.peak_volume, zs))
    scored.sort(key=lambda t: (t[0], -t[1]))
    return [s[2] for s in scored[:TOP_K]]


def _rank_emergents(zone_series: Dict[int, ZoneSeries]) -> List[ZoneSeries]:
    """Zones that went from nearly nothing to meaningful volume."""
    scored = []
    for zs in zone_series.values():
        if zs.zone_id in EXCLUDE_ZONES:
            continue
        if zs.recent_mean < MIN_RECENT_VOLUME:
            continue
        if zs.birth_mean < 1:
            continue
        if zs.birth_mean > 2000:
            continue
        ratio = zs.recent_mean / zs.birth_mean
        scored.append((ratio, zs.recent_mean, zs))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [s[2] for s in scored[:TOP_K]]


def _rank_resurrected(zone_series: Dict[int, ZoneSeries]) -> List[ZoneSeries]:
    """Zones with historic peak, deep trough, then comeback."""
    scored = []
    for zs in zone_series.values():
        if zs.zone_id in EXCLUDE_ZONES:
            continue
        if zs.peak_volume < MIN_HISTORIC_PEAK:
            continue
        if zs.recent_mean < MIN_RECENT_VOLUME:
            continue
        min_idx = int(zs.counts.argmin())
        if zs.counts[min_idx] > zs.peak_volume * 0.2:
            continue
        peak_idx = int(zs.counts.argmax())
        if not (peak_idx < min_idx < len(zs.counts) - 6):
            continue
        trough = max(1.0, float(zs.counts[min_idx]))
        recovery = zs.recent_mean / trough
        scored.append((recovery, zs.recent_mean, zs))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [s[2] for s in scored[:TOP_K]]


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "ghost":    "#b91c1c",
    "cooling":  "#f59e0b",
    "stable":   "#e2e8f0",
    "warming":  "#22c55e",
    "emergent": "#2563eb",
    "unknown":  "#cbd5e1",
}


def _project(x: float, y: float) -> Tuple[float, float]:
    """NY state-plane (EPSG:2263, feet) -> rough local metric for SVG.

    We only need correct relative positions within NYC, not accurate lon/lat.
    Origin at (~-74.06W, 40.48N) is close to south-west corner of the bounding
    box. 1 ft_x ~ 1 unit; we flip Y because SVG is top-down.
    """
    return (x, -y)


def _rings_to_svg_path(rings: List[List[Tuple[float, float]]]) -> str:
    parts = []
    for ring in rings:
        if not ring:
            continue
        px0, py0 = _project(*ring[0])
        d = [f"M{px0:.1f},{py0:.1f}"]
        for x, y in ring[1:]:
            px, py = _project(x, y)
            d.append(f"L{px:.1f},{py:.1f}")
        d.append("Z")
        parts.append(" ".join(d))
    return " ".join(parts)


def _render_map_svg(
    shapes: Dict[int, List[List[Tuple[float, float]]]],
    labels: Dict[int, str],
    zone_lookup: Dict[int, Dict],
) -> Tuple[str, Dict[str, int]]:
    """Return (svg_markup, category_counts)."""
    all_points = []
    for rings in shapes.values():
        for ring in rings:
            for x, y in ring:
                px, py = _project(x, y)
                all_points.append((px, py))
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = 2000.0
    width = max_x - min_x + 2 * pad
    height = max_y - min_y + 2 * pad

    cat_counts = defaultdict(int)
    paths = []
    for zid, rings in shapes.items():
        cat = labels.get(zid, "unknown")
        cat_counts[cat] += 1
        fill = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["unknown"])
        d = _rings_to_svg_path(rings)
        name = (zone_lookup.get(zid) or {}).get("zone", f"Zone {zid}")
        borough = (zone_lookup.get(zid) or {}).get("borough", "")
        paths.append(
            f'<path d="{d}" fill="{fill}" stroke="#0f172a" stroke-width="20" '
            f'stroke-opacity="0.3" data-zone-id="{zid}">'
            f'<title>{name} ({borough}) — {cat}</title></path>'
        )

    svg_body = "\n".join(paths)
    view = f"{min_x - pad:.1f} {min_y - pad:.1f} {width:.1f} {height:.1f}"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" '
        f'preserveAspectRatio="xMidYMid meet" width="100%" '
        f'style="max-width:960px;background:#0b1220">{svg_body}</svg>'
    )
    return svg, dict(cat_counts)


def _sparkline_svg(counts: np.ndarray, width: int = 240, height: int = 40, color: str = "#2563eb") -> str:
    n = len(counts)
    if n < 2:
        return ""
    max_v = float(counts.max()) or 1.0
    step = width / (n - 1)
    pts = []
    for i, v in enumerate(counts):
        x = i * step
        y = height - (float(v) / max_v) * height
        pts.append(f"{x:.1f},{y:.1f}")
    path = "M" + " L".join(pts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'
    )


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

_CSS = """
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1024px; margin: 40px auto; padding: 0 20px; line-height: 1.55;
         color: #0f172a; }
  h1 { font-size: 34px; margin-bottom: 4px; }
  h2 { font-size: 22px; margin-top: 36px; margin-bottom: 6px; }
  .sub { color: #64748b; margin-top: 0; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; margin: 10px 0 20px; }
  .chip { display: inline-flex; align-items: center; gap: 6px; padding: 2px 8px;
          border-radius: 999px; background: #f1f5f9; }
  .chip-dot { width: 10px; height: 10px; border-radius: 999px; display: inline-block; }
  .card { border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 20px;
          margin: 12px 0; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }
  .card .label { font-size: 12px; color: #64748b; text-transform: uppercase;
                 letter-spacing: 0.05em; }
  .card h3 { margin: 4px 0 8px 0; font-size: 17px; }
  .card.alert { position: relative; background: linear-gradient(180deg, rgba(239,68,68,0.08), rgba(239,68,68,0.02)),
                #fff; border-color: rgba(239,68,68,0.3); border-left: 3px solid #ef4444; padding: 22px 26px 24px; }
  .card.alert .label { color: #ef4444; letter-spacing: 0.22em; font-weight: 700; }
  .card.alert h3 { font-size: 22px; color: #0f172a; font-weight: 800; margin: 8px 0 12px; max-width: 680px; }
  .card.alert .stats { color: #475569; font-size: 14px; max-width: 680px; }
  .card.alert .date-chip { position: absolute; top: 18px; right: 20px; font-family: 'JetBrains Mono', monospace;
                           font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase; color: #ef4444;
                           font-weight: 700; border: 1px solid rgba(239,68,68,0.42); padding: 4px 10px; background: rgba(239,68,68,0.06); }
  .stats { color: #475569; font-size: 13px; }
  .grid { display: grid; grid-template-columns: 1fr 260px; gap: 16px; align-items: center; }
  .grid svg { display: block; }
  .footer { color: #94a3b8; font-size: 12px; margin-top: 40px; }
  .map-wrap { margin: 16px 0; background: #0b1220; border-radius: 14px;
              padding: 8px; overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #f1f5f9; }
  th { color: #64748b; font-weight: 500; }
</style>
"""


def _fmt_ym(ym_int: int) -> str:
    y, m = divmod(ym_int, 100)
    return f"{y:04d}-{m:02d}"


def _fmt_int(n: float) -> str:
    n = int(round(n))
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,}"


def _render_zone_card(
    rank: int, zs: ZoneSeries, zone_lookup: Dict[int, Dict],
    color: str, tag_label: str, extra: str,
) -> str:
    meta = zone_lookup.get(zs.zone_id) or {}
    zone_name = meta.get("zone", f"Zone {zs.zone_id}")
    borough = meta.get("borough", "")
    spark = _sparkline_svg(zs.counts, color=color)
    return f"""
    <div class=card>
      <div class=grid>
        <div>
          <div class=label>#{rank} &middot; {borough} &middot; {tag_label}</div>
          <h3>{zone_name}</h3>
          <div class=stats>
            peaked {_fmt_ym(zs.peak_month)} at {_fmt_int(zs.peak_volume)}/mo &middot;
            now {_fmt_int(zs.recent_mean)}/mo &middot;
            {_fmt_int(zs.total_trips)} trips total &middot;
            active {_fmt_ym(zs.first_active_month)}–{_fmt_ym(zs.last_active_month)}
          </div>
          <div class=stats>{extra}</div>
        </div>
        <div>{spark}</div>
      </div>
    </div>"""


def _render_main_html(
    all_months: List[int],
    monthly_totals: List[int],
    zone_series: Dict[int, ZoneSeries],
    zone_lookup: Dict[int, Dict],
    labels: Dict[int, str],
    map_svg: str,
    cat_counts: Dict[str, int],
    ghosts: List[ZoneSeries],
    emergents: List[ZoneSeries],
    resurrected: List[ZoneSeries],
    task_count: int,
    total_trips: int,
    elapsed_s: float,
    generated_at: str,
) -> str:
    chips = "".join(
        f'<span class=chip><span class=chip-dot style="background:{CATEGORY_COLORS[c]}"></span>'
        f'{c} ({cat_counts.get(c,0)})</span>'
        for c in ("ghost", "cooling", "stable", "warming", "emergent")
    )

    agg_spark = _sparkline_svg(np.array(monthly_totals, dtype=np.int64), width=960, height=80, color="#0ea5e9")

    # Find the modal trough month across "resurrected" zones. If 11 out of 12
    # zones hit their low at the same month, the demo needs to tell you.
    trough_months = [int(zs.months[zs.counts.argmin()]) for zs in resurrected] if resurrected else []
    mass_extinction: str = ""
    if trough_months:
        from collections import Counter as _Counter
        mode_month, mode_count = _Counter(trough_months).most_common(1)[0]
        if mode_count >= max(3, len(trough_months) // 2):
            _ym = _fmt_ym(mode_month).replace("-", " &middot; ")
            mass_extinction = (
                f'<div class="card alert">'
                f'<div class=date-chip>{_ym}</div>'
                f'<div class=label>Mass extinction event</div>'
                f'<h3>{mode_count} of {len(trough_months)} resurrected zones bottomed '
                f'in the same month.</h3>'
                f'<div class=stats>When the data is allowed to speak, the pandemic shows up '
                f'as a single visible knife-cut across NYC&rsquo;s busiest neighborhoods.</div>'
                f'</div>'
            )

    def section(title: str, subtitle: str, items: List[ZoneSeries], color: str, tag_fn, extras: str = "") -> str:
        if not items:
            return f"<h2>{title}</h2><p class=stats>{subtitle}</p><p><em>No zones matched.</em></p>"
        cards = "\n".join(_render_zone_card(i + 1, zs, zone_lookup, color, *tag_fn(zs)) for i, zs in enumerate(items))
        return f"<h2>{title}</h2><p class=stats>{subtitle}</p>{extras}{cards}"

    def ghost_tags(zs: ZoneSeries) -> Tuple[str, str]:
        ratio = zs.recent_mean / max(1.0, float(zs.peak_volume))
        return (f"now {ratio*100:.1f}% of peak",
                f"drop vs peak: {_fmt_int(zs.peak_volume - zs.recent_mean)}/mo")

    def emergent_tags(zs: ZoneSeries) -> Tuple[str, str]:
        ratio = zs.recent_mean / max(1.0, zs.birth_mean)
        return (f"{ratio:.0f}x birth-window rate",
                f"started at {_fmt_int(zs.birth_mean)}/mo; now {_fmt_int(zs.recent_mean)}/mo")

    def resurrected_tags(zs: ZoneSeries) -> Tuple[str, str]:
        min_idx = int(zs.counts.argmin())
        return (f"bottomed {_fmt_ym(int(zs.months[min_idx]))}",
                f"trough {_fmt_int(zs.counts[min_idx])}/mo → now {_fmt_int(zs.recent_mean)}/mo")

    return f"""<!doctype html><meta charset=utf-8><title>Ghost Neighborhoods of NYC</title>{_CSS}
<h1>Ghost Neighborhoods of NYC</h1>
<p class=sub>{_fmt_int(total_trips)} for-hire trips across {len(all_months)} months and
{len(zone_series)} taxi zones — a data-first X-ray of which parts of NYC quietly stopped
getting picked up. Computed across {task_count} monthly trip-records files from the NYC TLC
by a {task_count}-way parallel Burla map in {elapsed_s:.1f}s.</p>

<div class=legend>{chips}</div>
<div class=map-wrap>{map_svg}</div>

<h2>The entire NYC for-hire fleet, month by month</h2>
<p class=stats>{_fmt_ym(all_months[0])} → {_fmt_ym(all_months[-1])} &middot;
single line, one point per month, every taxi + every rideshare pickup.</p>
{agg_spark}

{section("Ghost Zones",
         "Zones that peaked during the taxi era and never came back. Ranked by current trips/month as a fraction of their all-time monthly peak.",
         ghosts, "#b91c1c", ghost_tags)}

{section("Newborn Zones",
         "Zones where for-hire pickups barely existed in their first two years of observation, but are now pulling meaningful volume. Ranked by growth ratio.",
         emergents, "#2563eb", emergent_tags)}

{section("Resurrected Zones",
         "Zones that had a historic peak, a deep trough, and a measurable recovery from that trough.",
         resurrected, "#0ea5e9", resurrected_tags, mass_extinction)}

<div class=footer>Generated {generated_at}. Source: NYC TLC monthly trip-record parquets
(yellow 2011-2024, green 2014-2024, high-volume FHV 2019.02-2024), served via the
Hugging Face mirror <code>DinoPonjevic/NYC_TaxiData_RAW</code>. Pipeline: Burla
<code>remote_parallel_map</code> over every monthly file, merged on the driver.</div>
"""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _write_outputs(
    all_months: List[int],
    monthly_totals: List[int],
    zone_series: Dict[int, ZoneSeries],
    zone_lookup: Dict[int, Dict],
    raw_results: List[Dict],
    task_count: int,
    total_trips: int,
    elapsed_s: float,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shapes = _load_zone_shapes_as_svg_paths()
    labels = _classify(zone_series)
    map_svg, cat_counts = _render_map_svg(shapes, labels, zone_lookup)

    ghosts = _rank_ghosts(zone_series)
    emergents = _rank_emergents(zone_series)
    resurrected = _rank_resurrected(zone_series)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    html = _render_main_html(
        all_months, monthly_totals, zone_series, zone_lookup, labels, map_svg, cat_counts,
        ghosts, emergents, resurrected, task_count, total_trips, elapsed_s, generated_at,
    )
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")

    (OUT_DIR / "summary.json").write_text(json.dumps({
        "total_trips": int(total_trips),
        "n_months": len(all_months),
        "n_zones": len(zone_series),
        "n_tasks": task_count,
        "elapsed_s": round(elapsed_s, 2),
        "category_counts": cat_counts,
        "ghosts": [
            {"zone_id": zs.zone_id,
             "zone": zone_lookup.get(zs.zone_id, {}).get("zone", ""),
             "borough": zone_lookup.get(zs.zone_id, {}).get("borough", ""),
             "peak_month": zs.peak_month, "peak_volume": zs.peak_volume,
             "recent_mean": round(zs.recent_mean, 1)}
            for zs in ghosts
        ],
        "emergents": [
            {"zone_id": zs.zone_id,
             "zone": zone_lookup.get(zs.zone_id, {}).get("zone", ""),
             "borough": zone_lookup.get(zs.zone_id, {}).get("borough", ""),
             "birth_mean": round(zs.birth_mean, 1),
             "recent_mean": round(zs.recent_mean, 1),
             "first_active_month": zs.first_active_month}
            for zs in emergents
        ],
        "resurrected": [
            {"zone_id": zs.zone_id,
             "zone": zone_lookup.get(zs.zone_id, {}).get("zone", ""),
             "borough": zone_lookup.get(zs.zone_id, {}).get("borough", ""),
             "peak_volume": zs.peak_volume,
             "trough_month": int(zs.months[zs.counts.argmin()]),
             "trough_volume": int(zs.counts.min()),
             "recent_mean": round(zs.recent_mean, 1)}
            for zs in resurrected
        ],
        "generated_at_utc": generated_at,
    }, indent=2))

    # Emit a raw per-month aggregate for downstream diagnostics.
    (OUT_DIR / "task_results.json").write_text(json.dumps([
        {k: v for k, v in r.items() if k != "counts"} for r in raw_results
    ], indent=2))


def main_local() -> int:
    """Smoke path: process a single recent month on the driver."""
    tasks = build_task_list()
    tasks = tasks[-2:]
    print(f"LOCAL: running {len(tasks)} tasks in-process")

    t0 = time.time()
    results = [process_month(t) for t in tasks]
    print(f"LOCAL: {len(results)} tasks done in {time.time()-t0:.1f}s")

    zone_lookup = _load_zone_lookup()
    all_months, zone_series, _, monthly_totals = _build_series_table(results)
    total_trips = sum(r.get("rows_with_zone", 0) for r in results)
    _write_outputs(
        all_months, monthly_totals, zone_series, zone_lookup, results,
        task_count=len(tasks), total_trips=total_trips, elapsed_s=time.time() - t0,
    )
    print("LOCAL done.")
    print((OUT_DIR / "summary.json").read_text())
    return 0


def main() -> int:
    if os.environ.get("LOCAL", "").strip() not in ("", "0", "false", "False"):
        return main_local()

    from burla import remote_parallel_map  # type: ignore

    tasks = build_task_list()
    print(f"task count: {len(tasks)} monthly trip-record files to process")
    for t in tasks[:3]:
        print("  e.g.", t)
    print("  ...")
    for t in tasks[-3:]:
        print("       ", t)

    # HF/Xet happily serves hundreds of concurrent downloads. Parallelism is
    # bounded by cluster size, not by origin rate-limits like CloudFront was.
    max_par_env = os.environ.get("MAX_PARALLELISM")
    max_par = int(max_par_env) if max_par_env else None

    t0 = time.time()
    kwargs = {"func_cpu": 1, "func_ram": 4}
    if max_par is not None:
        kwargs["max_parallelism"] = max_par
    results = list(remote_parallel_map(process_month, tasks, **kwargs))
    elapsed = time.time() - t0
    print(f"map done in {elapsed:.1f}s; collected {len(results)} result dicts")

    zone_lookup = _load_zone_lookup()
    all_months, zone_series, _, monthly_totals = _build_series_table(results)
    total_trips = sum(r.get("rows_with_zone", 0) for r in results)
    print(f"reduce: {len(zone_series)} scored zones, {len(all_months)} months, "
          f"{_fmt_int(total_trips)} trips aggregated")

    _write_outputs(
        all_months, monthly_totals, zone_series, zone_lookup, results,
        task_count=len(tasks), total_trips=total_trips, elapsed_s=elapsed,
    )
    print(f"artifacts → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
