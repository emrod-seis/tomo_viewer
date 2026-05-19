# TOMO VIEWER

Interactive 3D visualisation of seismic tomography models and associated datasets, built with Plotly Dash.

---

## Requirements

```bash
pip install dash plotly numpy pandas scipy h5py lxml xlrd geopandas shapely matplotlib
```
```or
conda install dash plotly numpy pandas scipy h5py lxml xlrd geopandas shapely matplotlib
```
---

## Quick start

```bash
python app.py
```

Then open `http://127.0.0.1:8050` in your browser. Closing the browser window automatically shuts down the server.

---

## Auto-loaded files

Place the following files in the same directory as `app.py` and they will load at startup:

| File | Description |
|------|-------------|
| `SAM5_P_2019.tomo` | P-wave tomography model (CSV format) |
| `nazca_SAM5_P_2019_clip.grd` | Nazca slab geometry (GMT/NetCDF4 `.grd`) |
| `GVP_Volcano_List_Holocene_202605111003.xls` | GVP Holocene volcano catalogue |
| `eqs_NEIC_mag3.csv` | NEIC earthquake catalogue (M ≥ 3) |

If any file is missing the app starts without it and shows a warning in the relevant panel.

---

## Input file formats

### Tomography `.tomo` / `.csv`
Comma-separated with a header row. Required columns:

| Column | Description |
|--------|-------------|
| `lat` | Latitude (°) |
| `lon` | Longitude (°) |
| `dep` | Depth (km) |
| `%dVp` or `%dVs` | Velocity perturbation |
| `hq` | Hit quality / weight (optional, defaults to 1) |

Alternative column names accepted: `latitude`, `longitude`, `depth`, `dvp`, `dvs`, `dv`, `velocity`, `quality`, `hit_quality`, `weight`.

### Slab `.grd`
GMT/Slab2 NetCDF4 grid file with `x` (longitude), `y` (latitude), `z` (depth km) variables. Longitudes in 0–360 convention are automatically converted to −180–180.

### Volcanoes `.xls`
GVP (Global Volcanism Program) Holocene volcano list in SpreadsheetML format, downloadable from [volcano.si.edu](https://volcano.si.edu).

### Earthquakes `.csv`
Comma-separated with a header row. Required columns: `lat`, `lon`, `depth` (km). Optional: `magnitude` (also accepts `mag`, `ml`, `mw`, `ms`, `mb`). Alternative column names `latitude`, `longitude`, `depth_km`, `dep` are also accepted.

### XYZ anomaly `.xyz` / `.csv` / `.txt`
Whitespace- or comma-delimited with columns `x` (lon), `y` (lat), `z` (value/depth). Header row is optional; column order `x, y, z` is assumed if no recognised header is found.

---

## Controls

### Domain
Sets the geographic bounding box applied to all layers — tomography, slab, borders, volcanoes, and earthquakes.

| Control | Description |
|---------|-------------|
| LAT MIN / MAX | Latitude limits (°) |
| LON MIN / MAX | Longitude limits (°) |

### Tomography options

| Control | Description |
|---------|-------------|
| Hit quality | Minimum hit-quality threshold for displaying nodes |
| Depth range | Slider to select the depth extent of displayed layers; defaults to inner layers `[1:-1]` to avoid edge artefacts |
| ISO range MIN / MAX | Velocity perturbation thresholds for the isosurface |
| Surface count | Number of isosurfaces to render |
| Opacity | Isosurface transparency |
| Grid resolution | Interpolation grid size (higher = sharper but slower) |
| Colorscale | Colormap for the isosurface |

### Slab
Display the subducting slab either as a surface or as depth contour lines. Opacity is adjustable. Clipped to the domain bounding box.

### Volcanoes
GVP Holocene volcano catalogue rendered as markers at the surface. Clipped to the domain bounding box.

### Earthquakes

| Control | Description |
|---------|-------------|
| Show earthquakes | Toggle visibility |
| Scale by magnitude | Scale marker size by magnitude |
| Min magnitude | Filter events below this magnitude |
| Depth range MIN / MAX | Filter events outside this depth range (km) |

Earthquakes are filtered by both the domain lat/lon limits and the depth range inputs.

### Extras
Country/coastline borders clipped to the domain bounding box (requires Natural Earth shapefile, downloaded automatically on first run).

### XYZ anomaly surface
Renders an arbitrary 3D point cloud as a smooth isosurface blob. Useful for visualising anomaly boundaries or other volumetric datasets.

---

## Notes

- Natural Earth country borders are downloaded automatically on first run (~500 KB zip) and cached locally as `ne_110m_admin_0_countries_lakes.shp`.
- When uploading a new tomography file the domain lat/lon inputs reset to the new file's extents.
