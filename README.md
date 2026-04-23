# NYC Ghost Neighborhoods — a Burla demo

**Live site: <https://burla-cloud.github.io/nyc-ghost-neighborhoods/>**

## Try it in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Burla-Cloud/nyc-ghost-neighborhoods/blob/main/Burla_NYCGhostNeighborhoods_Demo.ipynb)

Follow along in a notebook - scan 36 months of NYC taxi parquet across 12 cloud workers and find emergent / declining neighborhoods. About 3 minutes. No prior Burla knowledge needed.

Process **every single NYC taxi & for-hire-vehicle trip ever released by the
TLC** — 2,758,715,765 rows across 168 monthly parquet files, January 2011 →
December 2024 — and let the *geography itself* tell you which neighborhoods
died, which ones were born, and which ones went dark in the pandemic and then
came back.

No pre-picked zones, no "interesting neighborhood" list. Every one of the 264
TLC taxi zones is scored by its own time series and sorted into one of six
categories (`stable`, `emergent`, `cooling`, `warming`, `ghost`, `resurrected`).

## Headline

> **2.76 billion taxi trips, processed in ~15 seconds of wall-clock on Burla.**
> That is the entire public TLC trip archive (yellow + green + FHV + HVFHS,
> 2011 → present), scanned parquet-file-by-parquet-file in parallel.

|  |  |
|---|---:|
| Trips scanned | **2,758,715,765** |
| Months covered | 168 (2011-01 → 2024-12) |
| Taxi zones | 264 |
| Monthly parquet files processed | 371 (yellow + green + FHV + HVFHS) |
| Total on-disk corpus | ~50 GB parquet |
| Serial equivalent compute | **~30–60 min** (bandwidth-bound single laptop) |
| **Burla wall-clock (map + reduce)** | **~15 s** |
| Peak parallel workers | up to 168 (one per month) |

## Top ghost neighborhoods (peaked, still active, but a shadow of themselves)

Ranked by `recent_mean / peak_volume`. These are zones that once had heavy
yellow-cab traffic and now don't — classic pre-Uber Manhattan corridors that
the FHV data doesn't fully backfill.

| # | Zone | Borough | Peak month | Peak trips/mo | Recent mean | % of peak |
|:---:|:---|:---|:---:|---:|---:|---:|
| 1 | **Central Park** | Manhattan | 2011-05 | 219,763 | 86,952 | **40 %** |
| 2 | Battery Park | Manhattan | 2011-07 | 11,483 | 3,694 | 32 % |
| 3 | Randalls Island | Manhattan | 2019-05 | 10,330 | 3,593 | 35 % |
| 4 | Penn Station / Madison Sq W | Manhattan | 2012-03 | 478,490 | 241,276 | 50 % |
| 5 | Midtown East | Manhattan | 2011-03 | 536,481 | 276,741 | 52 % |
| 6 | Gramercy | Manhattan | 2012-03 | 436,977 | 226,530 | 52 % |
| 7 | Flushing Meadows-Corona Pk | Queens | 2022-09 | 45,111 | 18,212 | 40 % |

The pattern is almost entirely **Manhattan, pre-Uber**: rides that used to be
~100% yellow-cab in 2011 are now distributed across Uber/Lyft FHV trips, so
the *yellow-cab* line for these zones has collapsed even though the
neighborhoods are still alive.

## Top emergent neighborhoods (born essentially from zero)

Ranked by the ratio `recent_mean / birth_mean`. These are outer-borough zones
that yellow cabs barely served in 2011 but which the FHV (Uber/Lyft) data
reveals as real ride markets today.

| # | Zone | Borough | Birth mean/mo | Recent mean/mo | Growth |
|:---:|:---|:---|---:|---:|---:|
| 1 | **Far Rockaway** | Queens | 43.9 | **42,053.8** | **958×** |
| 2 | Pelham Bay | Bronx | 30.1 | 25,531.0 | 848× |
| 3 | Eastchester | Bronx | 64.4 | 54,723.8 | 849× |
| 4 | Woodlawn / Wakefield | Bronx | 85.8 | 64,601.8 | 753× |
| 5 | Brownsville | Brooklyn | 156.4 | 120,262.8 | 769× |
| 6 | East New York / Penn Ave | Brooklyn | 83.0 | 58,412.8 | 704× |
| 7 | Williamsbridge / Olinville | Bronx | 141.6 | 95,214.6 | 672× |

Every single one of the top 12 emergents is in the Bronx, deep Brooklyn,
Queens, or Staten Island — the outer-borough "transit deserts" that yellow
cabs historically refused to serve. The ride-share era effectively built a
second taxi system where yellow-cab coverage had been zero.

## Resurrected neighborhoods (pandemic crash → full recovery)

Same story, sharper shape: every one of these zones dropped >90 % in April 2020
and then clawed its way back.

| Zone | Borough | Peak | April-2020 trough | Recent mean | Recovery |
|:---|:---|---:|---:|---:|---:|
| **LaGuardia Airport** | Queens | 609,926 | 10,717 | **502,846** | 82 % |
| JFK Airport | Queens | 603,431 | 13,689 | 526,789 | 87 % |
| West Chelsea / Hudson Yards | Manhattan | 374,949 | 12,803 | 274,427 | 73 % |
| Little Italy / NoLiTa | Manhattan | 273,363 | 7,821 | 180,902 | 66 % |
| Greenwich Village South | Manhattan | 270,940 | 6,981 | 178,987 | 66 % |
| Midtown Center | Manhattan | 617,692 | 16,103 | 393,324 | 64 % |
| West Village | Manhattan | 379,452 | 12,376 | 266,027 | 70 % |

Midtown's peak→recent ratio (64–66 %) vs the airports' (82–87 %) is the
remote-work story in one number: airport travel is fully back; office-worker
corridors aren't.

## Why the TLC trip archive

- It's the largest public ground-truth mobility dataset in any city — the TLC
  releases *every trip*, not a sample.
- Monthly parquet files are the right unit of work for `remote_parallel_map`:
  one file per task, fan out to 168 tasks in one call.
- Pickup-zone ID + datetime is enough for the geography-of-mobility question;
  we don't need to join against lat/long or PII.

**Limitations we don't hide:**

1. The "ghost" and "emergent" patterns are partly a **reporting artifact**:
   TLC extended its FHV/HVFHS coverage over time, so some "emergent" zones
   are zones that weren't even in the feed in 2011, not zones that were
   empty. The metric still cleanly separates "yellow-cab-dominant" from
   "ride-share-dominant" zones though.
2. Trips are scored by pickup zone only — a yellow cab picking up in Midtown
   and dropping in the Bronx doesn't count as Bronx activity. Both zones
   would need symmetric treatment for a true mobility analysis.
3. Recent mean uses the last 12 months. Smoothing with a 36-month window
   softens the pandemic distortion but also softens the signal.

## Data source

- NYC TLC: `https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page`
- Actual parquet URLs: `https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{YYYY}-{MM}.parquet`
- Zone metadata: `taxi_zone_lookup.csv` (264 rows) — one row per `LocationID`
- Row schema varies by taxi_type; we only use `pickup_datetime` + `PULocationID`

## How it works

Two stages, one script:

- **Map — `process_month`** (one task per (taxi_type, year, month), func_cpu=1,
  func_ram=4). Streams the monthly parquet straight from CloudFront (no local
  cache), counts trips per `PULocationID`, and returns a tiny dict of
  `(zone_id → count)` plus some diagnostics. Each task is ~0.3–0.5 s of
  real work; the whole fan-out finishes before a single-threaded run has
  finished downloading one file.
- **Reduce — done on the client** after `remote_parallel_map` returns.
  Aggregates month-level zone counts into a (zone × month) matrix, joins
  the zone-lookup CSV for borough/name, classifies each zone into a
  category, and writes `index.html` + `summary.json`.

## How to run

```bash
# Full Burla run (~15 s wall-clock for the map; few seconds for the reduce)
python nyc_ghost_neighborhoods.py

# Constrain to yellow-cab only for a faster dry-run
NYC_TAXI_TYPES=yellow python nyc_ghost_neighborhoods.py

# Earlier years only
NYC_START_YEAR=2011 NYC_END_YEAR=2015 python nyc_ghost_neighborhoods.py
```

View the report: `open nyc_ghost_out/index.html` (~2 MB single-file).

## Artifacts (in `nyc_ghost_out/`)

| File | Contents |
|---|---|
| `index.html` | One-page report: all 264 zones + 6 categories + spark-line sparks |
| `summary.json` | Every emergent / ghost / resurrected zone + counts per category |
| `task_results.json` | Per-task timings and row counts for the 371 monthly jobs |

## Files

```
nyc_ghost_neighborhoods.py   map + classify + render (all core logic)
nyc_ghost_out/               artifacts from the latest run
requirements.txt             burla + pyarrow + requests + fsspec
```

---

*Source: NYC TLC trip record data · Orchestration: Burla `remote_parallel_map`.*
