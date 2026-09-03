# -*- coding: utf-8 -*-

"""
TOMOVIEWER - plot 3d datasets

Run:
    pip install dash plotly numpy pandas scipy h5py lxml xlrd
    python app.py
"""

import base64
import io
import tempfile
import os

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import h5py
import lxml.etree as _lxml_ET

app = dash.Dash(__name__)
server = app.server

# ── Shutdown endpoint ─────────────────────────────────────────────────────────
import signal, threading

@server.route('/shutdown', methods=['POST'])
def shutdown():
    """Called by the browser's beforeunload event to kill the server."""
    def _kill():
        import time; time.sleep(0.3)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_kill, daemon=True).start()
    return 'bye', 200

# ── Colorscales ───────────────────────────────────────────────────────────────
SEISMIC_SCALE = [
    [0.00, 'rgb(160,0,0)'],
    [0.20, 'rgb(255,80,0)'],
    [0.45, 'rgb(255,220,200)'],
    [0.50, 'rgb(255,255,255)'],
    [0.55, 'rgb(200,220,255)'],
    [0.80, 'rgb(30,144,255)'],
    [1.00, 'rgb(0,0,160)'],
]
COLORSCALE_OPTIONS = {
    'Seismic (blue-white-red)': SEISMIC_SCALE,
    'RdBu': 'RdBu', 'Viridis': 'Viridis',
    'Plasma': 'Plasma', 'Turbo': 'Turbo', 'Hot': 'Hot',
}

# ── Slab .grd reader ──────────────────────────────────────────────────────────
def read_grd(path_or_bytes):
    """
    Read a GMT/Slab2 .grd (HDF5/NetCDF4) file.
    Returns (lon_2d, lat_2d, depth_2d) as 2-D numpy arrays.
    depth_2d values are in km, NaN where no slab.
    """
    if isinstance(path_or_bytes, (str, os.PathLike)):
        f = h5py.File(path_or_bytes, 'r')
    else:
        # bytes from upload — write to temp file
        tmp = tempfile.NamedTemporaryFile(suffix='.grd', delete=False)
        tmp.write(path_or_bytes)
        tmp.close()
        f = h5py.File(tmp.name, 'r')

    x = f['x'][()]   # longitude in 0-360 convention
    y = f['y'][()]   # latitude
    z = f['z'][()]   # depth (km), shape (len(y), len(x))
    f.close()

    # Convert 0-360 → -180-180 if needed
    if x.max() > 180:
        x = x - 360.0

    lon_2d, lat_2d = np.meshgrid(x, y)
    return lon_2d, lat_2d, z.astype(float)


def slab_trace(lon_2d, lat_2d, depth_2d, downsample=1, opacity=0.6):
    """
    Build a go.Surface trace for the slab geometry.
    depth_2d values are positive km → plotted as z = -depth (down).
    Data is already downsampled at encode time; downsample arg kept for API compat.
    """
    d = downsample
    lo = lon_2d[::d, ::d]
    la = lat_2d[::d, ::d]
    de = np.array(depth_2d[::d, ::d], dtype=np.float64)
    z_plot = -de

    return go.Surface(
        x=lo,
        y=la,
        z=z_plot,
        surfacecolor=de,          # colour by depth
        colorscale='Viridis',
        reversescale=True,
        showscale=True,
        opacity=opacity,
        name='Nazca Slab',
        colorbar=dict(
            title=dict(text='Slab depth (km)', side='right',
                       font=dict(color='#aaa', size=11, family='monospace')),
            x=1.08,
            thickness=14, len=0.55,
            tickfont=dict(color='#aaa', family='monospace'),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
        ),
        hovertemplate=(
            'Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>'
            'Slab depth: %{surfacecolor:.0f} km<extra>Nazca Slab</extra>'
        ),
    )
def slab_contour_trace(lon_2d, lat_2d, depth_2d, n_contours=15, depth_range=None):
    """
    Depth contour lines of the slab drawn as Scatter3d at z=0 (surface projection).
    Extracts iso-depth contours from the 2D grid using matplotlib's contour engine.

    depth_range: optional (min_km, max_km) — same depth-slider range used for the
    tomography/earthquake filters. When given, grid points whose slab depth falls
    outside the range are masked out (so contour lines don't extend past the
    selected band) and the contour levels themselves are spaced across that
    range instead of the slab's full depth extent.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lo = lon_2d
    la = lat_2d
    de = -np.array(depth_2d, dtype=np.float64)

    if depth_range is not None:
        depth_lo, depth_hi = float(depth_range[0]), float(depth_range[1])
        out_of_range = (np.array(depth_2d, dtype=np.float64) < depth_lo) | \
                        (np.array(depth_2d, dtype=np.float64) > depth_hi)
        de = np.where(out_of_range, np.nan, de)

    valid = de[~np.isnan(de)]
    if len(valid) == 0:
        return None
    if depth_range is not None:
        levels = np.linspace(-depth_hi, -depth_lo, n_contours + 2)[1:-1]
    else:
        levels = np.linspace(valid.min(), valid.max(), n_contours + 2)[1:-1]

    fig_mpl, ax = plt.subplots()
    cs = ax.contour(lo, la, de, levels=levels)
    plt.close(fig_mpl)

    xs, ys, zs, texts = [], [], [], []
    for i, level in enumerate(cs.levels):
        for seg in cs.allsegs[i]:
            if len(seg) < 2:
                continue
            xs.extend(seg[:, 0].tolist() + [None])
            ys.extend(seg[:, 1].tolist() + [None])
            # use the contour level as the z coordinate
            zs.extend([float(level)] * len(seg) + [None])
            texts.extend([f'{level:.0f} km'] * len(seg) + [None])
    color_vals = np.array(zs, dtype=float)
    if not xs:
        return None
    return go.Scatter3d(
    	x=xs,
    	y=ys,
    	z=zs,

    	mode='lines',
    	name='Slab contours',
	line=dict(
        	color=color_vals,
        	colorscale='Viridis',
        	cmin=np.nanmin(color_vals),
        	cmax=np.nanmax(color_vals),
        	width=1.5,

        	colorbar=dict(
            	title=dict(
                	text='Slab depth (km)',
                	side='right',
                	font=dict(color='#aaa', size=11, family='monospace')
            	),
            	x=1.08,
            	thickness=14,
            	len=0.55,
            	tickfont=dict(color='#aaa', family='monospace'),
            	bgcolor='rgba(0,0,0,0)',
            	bordercolor='rgba(0,0,0,0)',
        	),
    	),
    	hovertext=texts,
    	hovertemplate='Slab depth: %{hovertext}<extra>Slab contour</extra>',
    	opacity=0.9,
	)

# ── Country borders ───────────────────────────────────────────────────────────
_BORDERS_SHP = "ne_110m_admin_0_countries_lakes.shp"

def _ensure_shapefile():
    """Download Natural Earth 110m countries shapefile if not present."""
    import os, zipfile, io, urllib.request
    if os.path.exists(_BORDERS_SHP) and os.path.getsize(_BORDERS_SHP) > 1000:
        return True
    ne_urls = [
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries_lakes.zip",
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries_lakes.zip",
    ]
    for url in ne_urls:
        try:
            print(f"[borders] Downloading from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                zdata = r.read()
            with zipfile.ZipFile(io.BytesIO(zdata)) as z:
                z.extractall(".")
            if os.path.exists(_BORDERS_SHP) and os.path.getsize(_BORDERS_SHP) > 1000:
                print("[borders] Shapefile downloaded successfully.")
                return True
        except Exception as e:
            print(f"[borders] Download failed ({url}): {e}")
    return False

def _load_borders():
    _ensure_shapefile()
    try:
        import geopandas as gpd
        world = gpd.read_file(_BORDERS_SHP)

    except Exception as e:
        print("Border load failed:", e)
        return None, None

    xs, ys = [], []

    for geom in world.geometry:
        if geom is None:
            continue

        polys = (
            list(geom.geoms)
            if geom.geom_type == "MultiPolygon"
            else [geom]
        )

        for poly in polys:
            coords = np.asarray(poly.exterior.coords)

            xs.extend(coords[:, 0].tolist() + [None])
            ys.extend(coords[:, 1].tolist() + [None])

    return xs, ys

def _check_borders_available():
    _ensure_shapefile()
    return os.path.exists(_BORDERS_SHP) and os.path.getsize(_BORDERS_SHP) > 1000

_BORDER_XS = True if _check_borders_available() else None
_BORDER_YS = _BORDER_XS

from shapely.geometry import box
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go

def borders_trace(lat_min, lat_max, lon_min, lon_max):
    import os
    if not os.path.exists(_BORDERS_SHP) or os.path.getsize(_BORDERS_SHP) < 1000:
        print('[borders] shapefile missing')
        return None
    try:
        world = gpd.read_file(_BORDERS_SHP)
    except Exception as e:
        print(f'[borders] read failed: {e}')
        return None

    bbox = box(lon_min, lat_min, lon_max, lat_max)

    # Keep only intersecting geometries
    world = world[world.intersects(bbox)]

    xs, ys = [], []

    for geom in world.geometry:

        if geom is None:
            continue

        polys = (
            geom.geoms
            if geom.geom_type == "MultiPolygon"
            else [geom]
        )

        for poly in polys:

            clipped = poly.intersection(bbox)

            if clipped.is_empty:
                continue

            clipped_polys = (
                clipped.geoms
                if clipped.geom_type == "MultiPolygon"
                else [clipped]
            )

            for cp in clipped_polys:

                if not hasattr(cp, "exterior"):
                    continue

                coords = np.asarray(cp.exterior.coords)

                xs.extend(coords[:, 0].tolist() + [None])
                ys.extend(coords[:, 1].tolist() + [None])

    if not xs:
        return None

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=[0.1 if x is not None else None for x in xs],

        mode='lines',

        name='Borders',

        line=dict(
            color='rgba(220,220,220,0.8)',
            width=3,
        ),

        hoverinfo='skip',
        showlegend=False,
    )

def read_volcanoes(path_or_bytes):

    """
    Parse GVP Holocene volcano list (SpreadsheetML .xls).
    Returns a DataFrame with columns:
      name, country, lat, lon, elevation_m, type, last_eruption
    """
    if isinstance(path_or_bytes, (str, os.PathLike)):
        with open(path_or_bytes, 'rb') as f:
            content = f.read()
    else:
        content = path_or_bytes

    parser = _lxml_ET.XMLParser(recover=True)
    root = _lxml_ET.fromstring(content, parser)
    ns = 'urn:schemas-microsoft-com:office:spreadsheet'

    rows_data = []
    for row in root.iter(f'{{{ns}}}Row'):
        cells = []
        for cell in row.iter(f'{{{ns}}}Cell'):
            datas = list(cell.iter(f'{{{ns}}}Data'))
            cells.append(datas[0].text if datas else '')
        rows_data.append(cells)

    if len(rows_data) < 3:
        return None, 'Could not parse rows from file'

    # Row 0: title/metadata, Row 1: column headers, Row 2+: data
    header = rows_data[1]
    df = pd.DataFrame(rows_data[2:], columns=header)
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        'Volcano Name': 'name',
        'Country': 'country',
        'Latitude': 'lat',
        'Longitude': 'lon',
        'Elevation (m)': 'elevation_m',
        'Primary Volcano Type': 'type',
        'Last Known Eruption': 'last_eruption',
    })

    for col in ['lat', 'lon', 'elevation_m']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['lat', 'lon'])
    return df[['name', 'country', 'lat', 'lon',
               'elevation_m', 'type', 'last_eruption']], None


def filter_volcanoes_to_region(vol_df, lat_min, lat_max, lon_min, lon_max):
    """Clip volcano list to the current tomo bounding box."""
    return vol_df[
        (vol_df['lat'] >= lat_min) & (vol_df['lat'] <= lat_max) &
        (vol_df['lon'] >= lon_min) & (vol_df['lon'] <= lon_max)
    ].copy()


def volcano_trace(vol_df):
    """
    go.Scatter3d — volcanoes plotted at z=0 (surface) as red triangles.
    """
    if vol_df is None or len(vol_df) == 0:
        return None

    elev_m = pd.to_numeric(vol_df['elevation_m'], errors='coerce').fillna(0)

    hover = (
        vol_df['name'].fillna('') + '<br>' +
        vol_df['type'].fillna('') + '<br>' +
        'Elev: ' + elev_m.astype(int).astype(str) + ' m<br>' +
        'Last eruption: ' + vol_df['last_eruption'].fillna('unknown')
    )

    return go.Scatter3d(
        x=vol_df['lon'].tolist(),
        y=vol_df['lat'].tolist(),
        z=[0.0] * len(vol_df),   # plotted at surface
        mode='markers',
        name='Volcanoes',
        marker=dict(
            symbol='diamond',
            size=3,
            color='rgb(255,80,20)',
            line=dict(color='rgb(255,200,100)', width=1),
            opacity=0.95,
        ),
        hovertext=hover.tolist(),
        hovertemplate='%{hovertext}<extra>Volcano</extra>',
    )


def parse_earthquakes(raw_bytes):
    """
    Parse earthquake CSV. Required columns: lat, lon, depth (km).
    Optional: magnitude (or mag / ml / mw / ms / mb).
    Returns (df, has_magnitude, error_str).
    """
    try:
        text = raw_bytes.decode('utf-8')
    except Exception:
        return None, False, 'Cannot decode as UTF-8.'
    try:
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip().lower() for c in df.columns]
    except Exception as e:
        return None, False, str(e)

    # Normalise column names
    df = df.rename(columns={
        'latitude': 'lat', 'longitude': 'lon',
        'depth_km': 'depth', 'dep': 'depth',
        'mag': 'magnitude', 'ml': 'magnitude', 'mw': 'magnitude',
        'ms': 'magnitude', 'mb': 'magnitude',
    })
    missing = {'lat', 'lon', 'depth'} - set(df.columns)
    if missing:
        return None, False, f'Missing columns: {missing}'
    for c in ['lat', 'lon', 'depth']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['lat', 'lon', 'depth'])
    has_mag = 'magnitude' in df.columns
    if has_mag:
        df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
    return df, has_mag, None


def parse_pha(raw_bytes):
    """
    Parse a NonLinLoc/HypoDD-style .pha phase file.

    Event (hypocenter) header lines start with '#':
        # YYYY MM DD HH MM SS.ss  LAT  LON  DEPTH  MAG  ERR_H ERR_V ERR_T  NPHASE
    All other lines are per-station phase-pick lines and are ignored — only
    the hypocenter location on each '#' line is extracted.

    Returns (df, has_magnitude, error_str) with columns: lat, lon, depth,
    magnitude, time (string 'YYYY-MM-DD HH:MM:SS.ss').
    """
    try:
        text = raw_bytes.decode('utf-8', errors='replace')
    except Exception:
        return None, False, 'Cannot decode as UTF-8.'

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('#'):
            continue
        parts = line[1:].split()
        if len(parts) < 9:
            continue
        try:
            year, month, day, hour, minute = parts[0:5]
            sec   = float(parts[5])
            lat   = float(parts[6])
            lon   = float(parts[7])
            depth = float(parts[8])
            mag   = float(parts[9]) if len(parts) > 9 else np.nan
        except (ValueError, IndexError):
            continue
        rows.append({
            'lat': lat, 'lon': lon, 'depth': depth, 'magnitude': mag,
            'time': f'{year}-{int(month):02d}-{int(day):02d} '
                    f'{int(hour):02d}:{int(minute):02d}:{sec:05.2f}',
        })

    if not rows:
        return None, False, 'No event header lines (starting with "#") found.'

    df = pd.DataFrame(rows)
    has_mag = bool(df['magnitude'].notna().any())
    return df, has_mag, None


def parse_earthquake_file(raw_bytes, filename=''):
    """
    Dispatch earthquake-file parsing by extension:
      .pha / .phs → hypocenter header lines (parse_pha)
      everything else → CSV (parse_earthquakes)
    """
    ext = os.path.splitext(filename or '')[1].lower()
    if ext in ('.pha', '.phs'):
        return parse_pha(raw_bytes)
    return parse_earthquakes(raw_bytes)


def earthquake_trace(eq_df, scale_by_mag=False, cmin=None, cmax=None):
    """go.Scatter3d — earthquakes coloured by depth using Viridis (same as slab)."""
    if eq_df is None or len(eq_df) == 0:
        return None
    z = (-eq_df['depth']).tolist()
    if scale_by_mag and 'magnitude' in eq_df.columns:
        mag = eq_df['magnitude'].fillna(1.0)
        sizes = (mag ** 2 / 8).clip(0.5, 20).tolist()
        hover = ('Lat: ' + eq_df['lat'].round(2).astype(str) + '<br>' +
                 'Lon: ' + eq_df['lon'].round(2).astype(str) + '<br>' +
                 'Depth: ' + eq_df['depth'].round(1).astype(str) + ' km<br>' +
                 'M: ' + mag.round(1).astype(str))
    else:
        sizes = 1
        hover = ('Lat: ' + eq_df['lat'].round(2).astype(str) + '<br>' +
                 'Lon: ' + eq_df['lon'].round(2).astype(str) + '<br>' +
                 'Depth: ' + eq_df['depth'].round(1).astype(str) + ' km')
    return go.Scatter3d(
        x=eq_df['lon'].tolist(), y=eq_df['lat'].tolist(), z=z,
        mode='markers',
        name='Earthquakes',
        marker=dict(
            size=sizes,
            color=eq_df['depth'].tolist(),
            colorscale='Viridis',
            reversescale=True,
            cmin=cmin,
            cmax=cmax,
            opacity=0.80,
            line=dict(width=0),
            showscale=False,   # shares Viridis scale with slab
        ),
        hovertext=hover.tolist(),
        hovertemplate='%{hovertext}<extra>Earthquake</extra>',
    )




# ── XYZ anomaly surface ───────────────────────────────────────────────────────
def parse_xyz(raw_bytes):
    """
    Parse a whitespace- or comma-delimited XYZ file.
    Accepts columns in any order as long as the header contains x/lon, y/lat, z/depth (or similar).
    Falls back to positional (col 0=x, 1=y, 2=z) if no header is recognised.
    Returns (df with columns x, y, z, error_str).
    """
    try:
        text = raw_bytes.decode('utf-8')
    except Exception:
        return None, 'Cannot decode as UTF-8.'
    import io as _io

    df = None
    # Try each combination of separator and header treatment
    for sep, hdr in [(',', 0), (r'\s+', 0), (',', None), (r'\s+', None)]:
        try:
            candidate = pd.read_csv(
                _io.StringIO(text), sep=sep, header=hdr,
                engine='python', on_bad_lines='skip',
            )
            if candidate.shape[1] >= 3 and len(candidate) >= 4:
                df = candidate
                break
        except Exception:
            continue

    if df is None:
        return None, 'Could not parse file — need at least 3 columns.'

    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map recognised header names
    X_NAMES = ['x', 'lon', 'longitude', 'easting']
    Y_NAMES = ['y', 'lat', 'latitude',  'northing']
    Z_NAMES = ['z', 'depth', 'value', 'val', 'anomaly', 'elev', 'elevation']

    def _pick(names):
        for n in names:
            if n in df.columns:
                return n
        return None

    xc, yc, zc = _pick(X_NAMES), _pick(Y_NAMES), _pick(Z_NAMES)

    # Positional fallback — use first three columns
    cols = list(df.columns)
    if xc is None: xc = cols[0]
    if yc is None: yc = cols[1]
    if zc is None: zc = cols[2]

    out = pd.DataFrame({'x': pd.to_numeric(df[xc], errors='coerce'),
                        'y': pd.to_numeric(df[yc], errors='coerce'),
                        'z': pd.to_numeric(df[zc], errors='coerce')})
    out = out.dropna()
    if len(out) < 4:
        return None, 'Too few valid rows after parsing.'
    return out, None


def xyz_surface_trace(xyz_df, ngrid=30, opacity=0.7, colorscale='Plasma'):
    """
    Render the XYZ boundary points as a smooth 3D volume using go.Isosurface.

    Strategy:
      - Build a 3D grid (lon × lat × depth).
      - At each depth layer, compute a 2D proximity/presence field: 1 near
        the boundary points, decaying to 0 away from them.
      - Stack into a 3D scalar field and render the 0.5 isosurface, giving a
        smooth closed blob that follows the point cloud in all three dimensions.
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter

    x, y, z = xyz_df['x'].values, xyz_df['y'].values, xyz_df['z'].values
    depths = np.unique(z)

    # Pad depth range slightly so the isosurface closes top and bottom
    z_pad   = (depths[-1] - depths[0]) * 0.15
    zi      = np.linspace(depths[0] - z_pad, depths[-1] + z_pad, max(len(depths) * 4, 24))

    xi = np.linspace(x.min(), x.max(), ngrid)
    yi = np.linspace(y.min(), y.max(), ngrid)

    # 3D grid arrays
    gx, gy, gz_dep = np.meshgrid(xi, yi, zi)   # each (ngrid, ngrid, nz)
    nz = len(zi)

    # Overall point cloud scale — use as the decay radius
    span = max(x.max()-x.min(), y.max()-y.min())
    radius = span / max(ngrid * 0.5, 1)

    # Build scalar field: at each (lon, lat, depth) node, interpolate
    # the presence of boundary points.  For each depth slice in zi, blend
    # contributions from nearby real depth layers weighted by depth distance.
    field = np.zeros((ngrid, ngrid, nz))

    for dep in depths:
        mask = z == dep
        lx, ly = x[mask], y[mask]
        if len(lx) < 2:
            continue
        tree = cKDTree(np.column_stack([lx, ly]))

        for k, zi_val in enumerate(zi):
            # Depth weight: Gaussian falloff from this real layer
            dz_sigma = (depths[-1] - depths[0]) / max(len(depths) - 1, 1) * 1.2
            depth_w = np.exp(-0.5 * ((zi_val - dep) / dz_sigma) ** 2)
            if depth_w < 0.01:
                continue

            pts2d = np.column_stack([gx[:, :, k].ravel(), gy[:, :, k].ravel()])
            dists, _ = tree.query(pts2d)
            # Spatial presence: 1 close to points, decaying outward
            presence = np.exp(-0.5 * (dists / radius) ** 2).reshape(ngrid, ngrid)
            field[:, :, k] += presence * depth_w

    # Normalise to [0, 1]
    fmax = field.max()
    if fmax < 1e-9:
        return None
    field /= fmax

    # Smooth the field for a clean isosurface
    sigma = max(1.0, ngrid / 15)
    field = gaussian_filter(field, sigma=[sigma, sigma, sigma * 0.5])

    # Flatten for go.Isosurface
    fx = gx.ravel().tolist()
    fy = gy.ravel().tolist()
    fz = (-gz_dep).ravel().tolist()   # negate: depth → negative z in scene
    fv = field.ravel().tolist()

    iso_thresh = float(field.max() * 0.5)

    return go.Isosurface(
        x=fx, y=fy, z=fz,
        value=fv,
        isomin=iso_thresh * 0.9, isomax=iso_thresh * 1.1,
        surface_count=1,
        colorscale=colorscale,
        showscale=False,
        opacity=opacity,
        name='XYZ Anomaly',
        caps=dict(x_show=False, y_show=False, z_show=False),
        hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Depth: %{z:.0f} km<extra>XYZ Anomaly</extra>',
    )
def _blockmean(lon, lat, val, lon_min, lon_max, lat_min, lat_max, ncells):
    """
    Average scattered points into ncells x ncells cells (gmt blockmean equivalent).
    Returns (lon_centres, lat_centres, mean_values) for non-empty cells only.
    """
    lon_edges = np.linspace(lon_min, lon_max, ncells + 1)
    lat_edges = np.linspace(lat_min, lat_max, ncells + 1)
    col_idx = np.searchsorted(lon_edges[1:], lon, side='left').clip(0, ncells - 1)
    row_idx = np.searchsorted(lat_edges[1:], lat, side='left').clip(0, ncells - 1)

    sums   = np.zeros((ncells, ncells))
    counts = np.zeros((ncells, ncells))
    np.add.at(sums,   (row_idx, col_idx), val)
    np.add.at(counts, (row_idx, col_idx), 1)

    mask = counts > 0
    row, col = np.where(mask)
    lon_c = 0.5 * (lon_edges[col] + lon_edges[col + 1])
    lat_c = 0.5 * (lat_edges[row] + lat_edges[row + 1])
    return lon_c, lat_c, sums[mask] / counts[mask]


def interpolate_to_grid(df, hq_min, depth_range, ngrid=50, lat_range=None, lon_range=None):
    """
    For each depth layer:
      1. blockmean  — average scattered points into cells  (gmt blockmean)
      2. RBFInterpolator thin-plate spline — smooth minimum-curvature surface
         (equivalent to gmt surface -T0)
      3. HQ mask — grid cells with no qualifying data point within one cell-width
         are set to 0 to suppress extrapolation artefacts
    Returns flat arrays (x, y, z, v) ready for go.Isosurface.
    """
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree

    mask = (
        (df['hq'] >= hq_min) &
        (df['dep'] >= depth_range[0]) &
        (df['dep'] <= depth_range[1])
    )
    if lat_range is not None:
        mask &= (df['lat'] >= lat_range[0]) & (df['lat'] <= lat_range[1])
    if lon_range is not None:
        mask &= (df['lon'] >= lon_range[0]) & (df['lon'] <= lon_range[1])
    sub = df[mask]
    if len(sub) < 10:
        return None

    depths = sorted(sub['dep'].unique())
    lat_min, lat_max = sub['lat'].min(), sub['lat'].max()
    lon_min, lon_max = sub['lon'].min(), sub['lon'].max()

    grid_lat = np.linspace(lat_min, lat_max, ngrid)
    grid_lon = np.linspace(lon_min, lon_max, ngrid)
    glon, glat = np.meshgrid(grid_lon, grid_lat)
    query_pts = np.column_stack([glon.ravel(), glat.ravel()])

    # Max distance a grid node can be from a real data point before being masked
    cell_w = (lon_max - lon_min) / ngrid
    cell_h = (lat_max - lat_min) / ngrid
    max_dist = np.sqrt((cell_w * 2) ** 2 + (cell_h * 2) ** 2)

    all_x, all_y, all_z, all_v = [], [], [], []

    for dep in depths:
        layer = sub[sub['dep'] == dep]
        if len(layer) < 4:
            continue

        # Step 1 — blockmean
        blon, blat, bval = _blockmean(
            layer['lon'].values, layer['lat'].values, layer['%dvp'].values,
            lon_min, lon_max, lat_min, lat_max, ngrid * 2,
        )
        if len(blon) < 4:
            continue

        # Step 2 — thin-plate spline
        try:
            rbf = RBFInterpolator(
                np.column_stack([blon, blat]), bval,
                kernel='thin_plate_spline',
                smoothing=0.0,
            )
            gv = rbf(query_pts).reshape(ngrid, ngrid)
        except Exception:
            gv = griddata((blon, blat), bval, (glon, glat), method='linear')
            gv = np.nan_to_num(gv, nan=0.0)

        # Step 3 — HQ proximity mask: zero out grid nodes too far from any
        # real data point (prevents RBF from extrapolating into empty regions)
        tree = cKDTree(np.column_stack([layer['lon'].values, layer['lat'].values]))
        dists, _ = tree.query(query_pts, workers=-1)
        outside = (dists > max_dist).reshape(ngrid, ngrid)
        gv[outside] = 0.0

        all_x.append(glon.ravel())
        all_y.append(glat.ravel())
        all_z.append(np.full(ngrid * ngrid, -float(dep)))
        all_v.append(gv.ravel())

    if not all_x:
        return None
    return (np.concatenate(all_x), np.concatenate(all_y),
            np.concatenate(all_z), np.concatenate(all_v))


# ── Figure builder ────────────────────────────────────────────────────────────
def build_figure(df, hq_min, iso_min, iso_max, n_surfaces, opacity,
                 cs_name, depth_range, ngrid,
                 lat_range=None, lon_range=None,
                 slab_data=None, show_slab=True, slab_opacity=0.6, slab_mode='surface',
                 vol_df=None, show_volcanoes=True, vel_label='%dVp',
                 show_borders=True, eq_df=None, show_earthquakes=False,
                 eq_scale_mag=False,
                 show_tomo=True,
                 xyz_df=None, show_xyz=False, xyz_opacity=0.7,
                 xyz_colorscale='Plasma', xyz_ngrid=60,
                 cscale_min=None, cscale_max=None):

    traces = []

    # ── Tomo isosurface ───────────────────────────────────────────────────────
    result = interpolate_to_grid(df, hq_min, depth_range, ngrid,
                                 lat_range=lat_range, lon_range=lon_range) if show_tomo else None
    if result is not None:
        x, y, z, v = result
        cs = COLORSCALE_OPTIONS.get(cs_name, SEISMIC_SCALE)
        vmin, vmax = float(v.min()), float(v.max())
        iso_min_c = max(iso_min, vmin)
        iso_max_c = min(iso_max, vmax)
        if iso_min_c >= iso_max_c:
            iso_min_c, iso_max_c = vmin * 0.8, vmax * 0.8
        eps = (vmax - vmin) * 0.02
        iso_min_c = max(iso_min_c, vmin + eps)
        iso_max_c = min(iso_max_c, vmax - eps)

        # Color scale bounds: use explicit overrides if provided, else match iso range
        cs_cmin = cscale_min if cscale_min is not None else iso_min_c
        cs_cmax = cscale_max if cscale_max is not None else iso_max_c

        # For seismic-style scales, zero-centre the colorscale so that data value 0
        # always maps to white (midpoint), negative → red, positive → blue.
        def _zero_centre_scale(scale, cmin, cmax):
            """Remap a symmetric colorscale so its midpoint sits at data value 0."""
            if not isinstance(scale, list):
                return scale, False   # named scale, cannot remap; keep reversescale
            total = cmax - cmin
            if total <= 0:
                return scale, False
            zero_frac = (0.0 - cmin) / total   # where 0 falls in [cmin, cmax]
            zero_frac = max(0.0, min(1.0, zero_frac))
            # Remap each stop: stops < 0.5 map to [0, zero_frac],
            # stops >= 0.5 map to [zero_frac, 1]
            new_scale = []
            for pos, color in scale:
                if pos <= 0.5:
                    new_pos = pos * 2.0 * zero_frac
                else:
                    new_pos = zero_frac + (pos - 0.5) * 2.0 * (1.0 - zero_frac)
                new_scale.append([round(new_pos, 6), color])
            return new_scale, False   # reversescale=False; mapping already correct

        if isinstance(cs, list):
            cs_mapped, do_reverse = _zero_centre_scale(cs, cs_cmin, cs_cmax)
        else:
            cs_mapped, do_reverse = cs, True   # non-seismic named scales keep original behaviour

        # Volume rendering with a graded opacity ramp: values in the middle of
        # the [isomin, isomax] window fade toward transparent, while the
        # strongest anomalies at the extremes become opaque. This shows the
        # whole field continuously instead of a handful of overlapping solid
        # shells, which made weaker anomalies hard to pick out.
        # surface_count controls how finely the volume is sampled — bump the
        # floor up so the opacity ramp looks smooth even if the "surfaces"
        # slider is set low (that control was tuned for discrete isosurfaces).
        _vol_surface_count = max(int(n_surfaces), 15)

        traces.append(go.Volume(
            x=x.tolist(), y=y.tolist(), z=z.tolist(),
            value=v.tolist(),
            isomin=iso_min_c, isomax=iso_max_c,
            surface_count=_vol_surface_count,
            opacityscale='extremes',
            colorscale=cs_mapped, reversescale=do_reverse,
            cmin=cs_cmin, cmax=cs_cmax,
            showscale=True,
            opacity=opacity,
            name=vel_label,
            colorbar=dict(
                title=dict(text=vel_label, side='right',
                           font=dict(color='#aaa', size=13, family='monospace')),
                x=0.95,
                thickness=16, len=0.70,
                tickfont=dict(color='#aaa', family='monospace'),
                bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
                tickformat='.3f',
            )
        ))

    # ── Slab surface / contours ───────────────────────────────────────────────
    if show_slab and slab_data is not None:
        lon_2d, lat_2d, depth_2d = slab_data
        if lat_range is not None or lon_range is not None:
            la0, la1 = (lat_range if lat_range else [lat_2d.min(), lat_2d.max()])
            lo0, lo1 = (lon_range if lon_range else [lon_2d.min(), lon_2d.max()])
            mask = (lat_2d >= la0) & (lat_2d <= la1) & (lon_2d >= lo0) & (lon_2d <= lo1)
            depth_clipped = np.where(mask, depth_2d, np.nan)
            # Trim rows/cols that are entirely NaN so the surface doesn't expand the scene
            row_ok = np.any(~np.isnan(depth_clipped), axis=1)
            col_ok = np.any(~np.isnan(depth_clipped), axis=0)
            lon_2d    = lon_2d[np.ix_(row_ok, col_ok)]
            lat_2d    = lat_2d[np.ix_(row_ok, col_ok)]
            depth_clipped = depth_clipped[np.ix_(row_ok, col_ok)]
        else:
            depth_clipped = depth_2d
        if slab_mode == 'contours':
            tr = slab_contour_trace(lon_2d, lat_2d, depth_clipped, depth_range=depth_range)
            if tr is not None:
                traces.append(tr)
        else:
            traces.append(slab_trace(lon_2d, lat_2d, depth_clipped, opacity=slab_opacity))

    # Shared region bounds for volcanoes / borders
    _lat_min = lat_range[0] if lat_range else df['lat'].min()
    _lat_max = lat_range[1] if lat_range else df['lat'].max()
    _lon_min = lon_range[0] if lon_range else df['lon'].min()
    _lon_max = lon_range[1] if lon_range else df['lon'].max()

    # ── Volcanoes ─────────────────────────────────────────────────────────────
    if show_volcanoes and vol_df is not None and len(vol_df) > 0:
        clipped = filter_volcanoes_to_region(
            vol_df,
            lat_min=_lat_min, lat_max=_lat_max,
            lon_min=_lon_min, lon_max=_lon_max,
        )
        tr = volcano_trace(clipped)
        if tr is not None:
            traces.append(tr)

    # ── Country borders ───────────────────────────────────────────────────────
    # Borders are drawn at the surface (z≈0), so only show them when the
    # selected depth window actually includes the surface (depth min == 0).
    if show_borders and float(depth_range[0]) <= 0:
        tr = borders_trace(
            lat_min=_lat_min, lat_max=_lat_max,
            lon_min=_lon_min, lon_max=_lon_max,
        )
        if tr is not None:
            traces.append(tr)

    # ── Earthquakes ───────────────────────────────────────────────────────────
    if show_earthquakes and eq_df is not None and len(eq_df) > 0:
        # Match the slab contours' colour range: same depth window as the
        # depth-range slider, so earthquake markers and slab contour lines
        # sit on the exact same Viridis colour scale.
        eq_cmin = float(depth_range[0])
        eq_cmax = float(depth_range[1])
        tr = earthquake_trace(eq_df, scale_by_mag=eq_scale_mag,
                              cmin=eq_cmin, cmax=eq_cmax)
        if tr is not None:
            traces.append(tr)

    # ── XYZ anomaly surface ───────────────────────────────────────────────────
    if show_xyz and xyz_df is not None and len(xyz_df) >= 4:
        tr = xyz_surface_trace(xyz_df, ngrid=int(xyz_ngrid),
                               opacity=xyz_opacity, colorscale=xyz_colorscale)
        if tr is not None:
            traces.append(tr)

    if not traces:
        fig = go.Figure()
        fig.add_annotation(
            text='No data to display — check filters',
            xref='paper', yref='paper', x=0.5, y=0.5,
            showarrow=False, font=dict(color='#aaa', size=14))
        fig.update_layout(**_dark_layout())
        return fig

    fig = go.Figure(data=traces)

    # Force the scene to the tomo domain by adding invisible corner markers
    _lo0 = lon_range[0] if lon_range else float(df['lon'].min())
    _lo1 = lon_range[1] if lon_range else float(df['lon'].max())
    _la0 = lat_range[0] if lat_range else float(df['lat'].min())
    _la1 = lat_range[1] if lat_range else float(df['lat'].max())
    _z0  = -float(depth_range[1])
    _z1  = -float(depth_range[0])
    fig.add_trace(go.Scatter3d(
        x=[_lo0, _lo1, _lo0, _lo1, _lo0, _lo1, _lo0, _lo1],
        y=[_la0, _la0, _la1, _la1, _la0, _la0, _la1, _la1],
        z=[_z0,  _z0,  _z0,  _z0,  _z1,  _z1,  _z1,  _z1],
        mode='markers',
        marker=dict(size=0.001, opacity=0),
        showlegend=False,
        hoverinfo='skip',
        name='_bounds',
    ))

    ax = dict(
        backgroundcolor='rgb(8,8,18)', gridcolor='rgb(50,50,70)',
        showbackground=True, zerolinecolor='rgb(80,80,100)',
        tickfont=dict(color='#c8d8f0', size=10, family='monospace'),
        title=dict(font=dict(color='#c8d8f0', size=11, family='monospace')),
    )
    depths_in_range = sorted(
        d for d in df['dep'].unique()
        if depth_range[0] <= d <= depth_range[1]
    )
    z_ticks  = [-d for d in depths_in_range]
    z_labels = [f'{int(d)} km' for d in depths_in_range]

    # Compute aspect ratio: scale x (lon) by cos(mean_lat) so 1° lon = 1° lat at mean lat
    _mean_lat = np.radians(0.5 * (_la0 + _la1))
    _lon_span = (_lo1 - _lo0) * np.cos(_mean_lat)
    _lat_span = _la1 - _la0
    _dep_span = float(depth_range[1] - depth_range[0])
    # Normalise so the largest horizontal dimension = 2
    _h_max = max(_lon_span, _lat_span, 1e-6)
    _ax = round(2.0 * _lon_span / _h_max, 3)
    _ay = round(2.0 * _lat_span / _h_max, 3)
    _az = round(2.0 * (_dep_span / 111.0) / _h_max, 3)  # convert km→degrees equivalent
    _az = max(0.3, min(_az, 1.5))  # clamp z so it doesn't get silly

    fig.update_layout(
        **_dark_layout(),
        scene=dict(
            xaxis={**ax, 'title': 'Longitude'},
            yaxis={**ax, 'title': 'Latitude'},
            zaxis={**ax, 'title': 'Depth',
                       'tickvals': z_ticks, 'ticktext': z_labels,
                       'range': [_z0, _z1]},
            bgcolor='rgb(8,8,18)',
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
            aspectmode='manual',
            aspectratio=dict(x=_ax, y=_ay, z=_az),
        ),
    )
    return fig


def _dark_layout():
    return dict(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgb(8,8,18)',
        font=dict(color='white',
                  family='"JetBrains Mono","Fira Code",monospace'),
        showlegend=True,
        legend=dict(
            x=0.01, y=0.98,
            xanchor='left', yanchor='top',
            bgcolor='rgba(10,10,28,0.75)',
            bordercolor='rgba(80,80,120,0.6)',
            borderwidth=1,
            font=dict(color='#b0c4e8', size=11, family='monospace'),
            itemsizing='constant',
            tracegroupgap=4,
        ),
    )


# ── File parsers ──────────────────────────────────────────────────────────────
def parse_tomo(raw_bytes, wave_type='vp'):
    """
    wave_type: 'vp' or 'vs' — which column to use when the file has both.
    Returns (df, vel_label, error_str).
    """
    try:
        text = raw_bytes.decode('utf-8')
    except Exception:
        return None, '%dVp', 'Cannot decode as UTF-8.'
    try:
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip().lower() for c in df.columns]
    except Exception as e:
        return None, '%dVp', str(e)

    VP_COLS = ['%dvp', 'dvp', 'dv', 'velocity', 'v']
    VS_COLS = ['%dvs', 'dvs']

    has_vp = any(c in df.columns for c in VP_COLS)
    has_vs = any(c in df.columns for c in VS_COLS)

    # Pick column based on wave_type preference, fall back to whatever exists
    if wave_type == 'vs' and has_vs:
        vel_label = '%dVs'
        # rename vs col → %dvp (internal unified name), drop vp if present
        for c in VS_COLS:
            if c in df.columns:
                df = df.rename(columns={c: '%dvp'})
                break
        for c in VP_COLS:
            if c in df.columns and c != '%dvp':
                df = df.drop(columns=[c], errors='ignore')
    elif has_vp:
        vel_label = '%dVp'
        for c in VP_COLS:
            if c in df.columns:
                df = df.rename(columns={c: '%dvp'})
                break
        for c in VS_COLS:
            if c in df.columns and c != '%dvp':
                df = df.drop(columns=[c], errors='ignore')
    elif has_vs:
        # Only vs available even though vp was requested
        vel_label = '%dVs'
        for c in VS_COLS:
            if c in df.columns:
                df = df.rename(columns={c: '%dvp'})
                break
    else:
        return None, '%dVp', f'No velocity column found. Columns: {list(df.columns)}'

    df = df.rename(columns={
        'latitude': 'lat', 'longitude': 'lon', 'depth': 'dep',
        'quality': 'hq', 'hit_quality': 'hq', 'weight': 'hq',
    })
    missing = {'lat', 'lon', 'dep', '%dvp'} - set(df.columns)
    if missing:
        return None, vel_label, f'Missing columns: {missing}'
    if 'hq' not in df.columns:
        df['hq'] = 1.0
    df = df[['lat', 'lon', 'dep', '%dvp', 'hq']].dropna()
    return df, vel_label, None


def parse_grd(raw_bytes):
    try:
        return read_grd(raw_bytes), None
    except Exception as e:
        return None, str(e)


# ── Slab store helpers ────────────────────────────────────────────────────────
def _encode_slab(slab_tuple, downsample=4):
    """Encode (lon_2d, lat_2d, depth_2d) into JSON-safe dict, downsampled."""
    lo, la, de = slab_tuple
    d = downsample
    lo, la, de = lo[::d, ::d], la[::d, ::d], de[::d, ::d]
    return dict(lon=lo.tolist(), lat=la.tolist(), dep=de.tolist(),
                shape=list(lo.shape))

def _decode_slab(d):
    """Decode JSON dict back to (lon_2d, lat_2d, depth_2d) as float64."""
    if d is None:
        return None
    shape = d['shape']
    return (
        np.array(d['lon'], dtype=np.float64).reshape(shape),
        np.array(d['lat'], dtype=np.float64).reshape(shape),
        np.array(d['dep'], dtype=np.float64).reshape(shape),  # None→nan via float64
    )


# ── Load startup data ─────────────────────────────────────────────────────────
# Tomo
try:
    with open('SAM5_P_2019.tomo', 'rb') as _f:
        _df, _vel_label, _err = parse_tomo(_f.read(), wave_type='vp')
    if _err:
        raise ValueError(_err)
    _tomo_msg = f'SAM5_P_2019.tomo  ({len(_df):,} pts)  [{_vel_label}]'
    print(f'[startup] Loaded tomo: {_tomo_msg}')
except Exception as e:
    print(f'[startup] Tomo load failed: {e} — using synthetic data')
    _vel_label = '%dVp'
    _la, _lo, _de = np.meshgrid(np.linspace(-60, 10, 15),
                                 np.linspace(-85, -45, 15),
                                 [60, 200, 410, 660, 1000])
    _v = 0.03*np.sin(np.radians(_la.ravel()))*np.cos(np.radians(_lo.ravel()))
    _df = pd.DataFrame({'lat': _la.ravel(), 'lon': _lo.ravel(),
                        'dep': _de.ravel(), '%dvp': _v, 'hq': 1.0})
    _tomo_msg = 'Synthetic demo data'

# Slab
try:
    _slab = read_grd('nazca_SAM5_P_2019_clip.grd')
    _slab_msg = 'nazca_SAM5_P_2019_clip.grd  loaded'
    print(f'[startup] Loaded slab: {_slab_msg}')
except Exception as e:
    print(f'[startup] Slab load failed: {e}')
    _slab = None
    _slab_msg = 'No slab file loaded'

# Volcanoes
_VOL_PATH = 'GVP_Volcano_List_Holocene_202605111003.xls'
try:
    _vol_df, _vol_err = read_volcanoes(_VOL_PATH)
    if _vol_err:
        raise ValueError(_vol_err)
    _vol_msg = f'{_VOL_PATH}  ({len(_vol_df):,} volcanoes global)'
    print(f'[startup] Loaded volcanoes: {_vol_msg}')
except Exception as e:
    print(f'[startup] Volcano load failed: {e}')
    _vol_df = None
    _vol_msg = 'No volcano file loaded'

# Earthquakes
_EQ_PATH = 'eqs_NEIC_mag3.csv'
try:
    with open(_EQ_PATH, 'rb') as _f:
        _eq_df, _eq_has_mag, _eq_err = parse_earthquake_file(_f.read(), _EQ_PATH)
    if _eq_err:
        raise ValueError(_eq_err)
    _eq_msg = f'{_EQ_PATH}  ({len(_eq_df):,} events)'
    if _eq_has_mag:
        _eq_msg += '  [mag col found]'
    print(f'[startup] Loaded earthquakes: {_eq_msg}')
except Exception as e:
    print(f'[startup] Earthquake load failed: {e}')
    _eq_df = None
    _eq_has_mag = False
    _eq_msg = 'No earthquake file loaded'

_depths  = sorted(_df['dep'].unique())
_vmin_g  = float(_df['%dvp'].min())
_vmax_g  = float(_df['%dvp'].max())
_dep_min = _depths[0]
_dep_max = _depths[-1]
_iso_init = [_vmax_g * 0.3, _vmax_g * 0.8]
# Default depth display range: skip outermost layer on each end [1:-1]
_dep_default_min = _depths[2]  if len(_depths) > 2 else _dep_min
_dep_default_max = _depths[-3] if len(_depths) > 2 else _dep_max
# Default lat/lon limits: use slab extents if loaded, else tomo data extents
if _slab is not None:
    _lo, _la, _de = _slab
    _valid = ~np.isnan(_de)
    _lat_min_g = float(_la[_valid].min())
    _lat_max_g = float(_la[_valid].max())
    _lon_min_g = float(_lo[_valid].min())
    _lon_max_g = float(_lo[_valid].max())
else:
    _lat_min_g = float(_df['lat'].min())
    _lat_max_g = float(_df['lat'].max())
    _lon_min_g = float(_df['lon'].min())
    _lon_max_g = float(_df['lon'].max())

print('[startup] Building initial figure...')
_fig0 = build_figure(
    _df,
    iso_min=_iso_init[0], iso_max=_iso_init[1],
    hq_min=0.1,
    n_surfaces=5, opacity=0.6,
    cs_name='Seismic (blue-white-red)',
    depth_range=[_dep_default_min, _dep_default_max], ngrid=50,
    lat_range=[_lat_min_g, _lat_max_g],
    lon_range=[_lon_min_g, _lon_max_g],
    show_tomo=False,
    slab_data=_slab, show_slab=(_slab is not None), slab_opacity=0.6, slab_mode='surface',
    vol_df=_vol_df, show_volcanoes=(_vol_df is not None),
    show_borders=(_BORDER_XS is not None),
)
print('[startup] Ready.')

# ── UI helpers ────────────────────────────────────────────────────────────────
PANEL = dict(background='rgb(14,14,28)', border='1px solid rgb(40,40,60)',
             borderRadius='8px', padding='16px', marginBottom='12px')
LBL   = dict(color='#b0c4e8', fontSize='10px', letterSpacing='2px',
             textTransform='uppercase', marginBottom='6px', display='block')
SUB   = dict(color='#8fa8cc', fontSize='9px', letterSpacing='2px',
             textTransform='uppercase', marginTop='12px', marginBottom='5px',
             display='block', paddingTop='10px',
             borderTop='1px solid rgb(35,35,58)')
VAL   = dict(fontSize='18px', color='#4af', marginBottom='6px', fontWeight='300')

def _smarks(lo, hi, n=4):
    return {float(v): dict(label=f'{v:.3f}',
                           style=dict(color='#b0c4e8', fontSize='10px'))
            for v in np.linspace(lo, hi, n)}


def _status_div(msg, ok=True):
    return html.Div(msg, style=dict(
        marginTop='6px', fontSize='11px', letterSpacing='1px',
        color='#4af' if ok else '#f64'))

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style=dict(
        background='rgb(6,6,14)', minHeight='100vh',
        fontFamily='"JetBrains Mono","Fira Code",monospace',
        color='white', display='flex', flexDirection='column',
        padding='20px', boxSizing='border-box',
    ),
    children=[
        # ── Inject CSS ────────────────────────────────────────────────────────
        dcc.Markdown('''
<style>
.radio-group label {
    display: inline-block;
    padding: 4px 11px;
    margin: 2px 3px 2px 0;
    background: rgb(28,28,52);
    border: 1px solid rgb(70,70,110);
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    color: #dfe8ff;
    font-family: monospace;
    transition: all 0.15s;
}

.radio-group input[type="radio"] {
    display: none;
}

.radio-group input[type="radio"]:checked + label {
    background: rgb(15,65,155);
    border-color: rgb(80,145,255);
    color: white;
    font-weight: 600;
    box-shadow: 0 0 8px rgba(80,145,255,0.5);
}
.radio-label:hover {
    border-color: rgb(90,140,255);
    color: white;
}

.radio-label:active {
    background: rgb(15,65,155);
    border-color: rgb(80,145,255);
    color: white;
    box-shadow: 0 0 8px rgba(80,145,255,0.5);
}


/* Dropdown dark override */
.Select-control { background: rgb(12,12,26) !important; border-color: rgb(55,55,95) !important; }
.Select-menu-outer { background: rgb(18,18,34) !important; }
.Select-option { color: #ccd8ee !important; }
.Select-option.is-focused { background: rgb(25,50,110) !important; }
.Select-value-label { color: #ccd8ee !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgb(45,45,85); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgb(65,65,115); }

/* Remove number input spinners */
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
}
input[type=number] {
    -moz-appearance: textfield;
    appearance: textfield;
}
</style>
<script>
window.addEventListener('beforeunload', function() {
    navigator.sendBeacon('/shutdown');
});
</script>''', dangerously_allow_html=True),

        html.Div(style=dict(marginBottom='20px'), children=[
            html.H1('TOMO VIEWER',
                    style=dict(fontSize='17px', letterSpacing='4px',
                               color='#4af', margin='0 0 4px 0', fontWeight='400')),
            html.P('3D data viewer',
                   style=dict(color='#7a90a8', fontSize='11px',
                              margin=0, letterSpacing='1.5px')),
        ]),

        html.Div(style=dict(display='flex', gap='20px', flex='1',
                             alignItems='flex-start'), children=[

            # ── Sidebar ───────────────────────────────────────────────────────
            html.Div(style=dict(
                width='300px', flexShrink='0',
                height='calc(100vh - 80px)',
                overflowY='auto', overflowX='hidden',
                paddingRight='6px',
                scrollbarWidth='thin',
                scrollbarColor='rgb(45,45,85) transparent',
            ), children=[

                # ── Domain ───────────────────────────────────────────────────
                html.Div(style=PANEL, children=[
                    html.Span('DOMAIN', style=LBL),

                    html.Span('LAT', style=SUB),
                    html.Div(style=dict(display='flex', gap='8px', alignItems='center'), children=[
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MIN', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='lat-min', type='text',
                                      value=str(round(_lat_min_g, 2)),
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                        html.Span('→', style=dict(color='#445566', fontSize='14px',
                                                   marginTop='16px')),
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MAX', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='lat-max', type='text',
                                      value=str(round(_lat_max_g, 2)),
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                    ]),

                    html.Span('LON', style=SUB),
                    html.Div(style=dict(display='flex', gap='8px', alignItems='center'), children=[
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MIN', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='lon-min', type='text',
                                      value=str(round(_lon_min_g, 2)),
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                        html.Span('→', style=dict(color='#445566', fontSize='14px',
                                                   marginTop='16px')),
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MAX', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='lon-max', type='text',
                                      value=str(round(_lon_max_g, 2)),
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                    ]),
                ]),

                # ── Tomo upload ───────────────────────────────────────────────
                html.Div(style=PANEL, children=[
                    html.Span('TOMO FILE  (.tomo / .csv)', style=LBL),
                    dcc.Upload(id='upload-tomo',
                               children=html.Div([
                                   html.Span('Drop .tomo / CSV ',
                                             style=dict(color='#6ab4ff')),
                                   html.Span('or click',
                                             style=dict(color='#8fa8cc')),
                               ]),
                               style=dict(border='1px dashed rgb(55,75,110)',
                                          borderRadius='6px', padding='10px',
                                          textAlign='center', cursor='pointer',
                                          fontSize='12px', background='rgb(10,10,22)'),
                               multiple=False),
                    html.Div(id='tomo-status', children=_status_div(_tomo_msg)),
                    dcc.Checklist(
                        id='show-tomo',
                        options=[{'label': ' Show tomography', 'value': 'show'}],
                        value=[],
                        inline=True,
                        inputClassName='radio-input',
                        labelClassName='radio-label',
                        labelStyle={'color': '#6ab4ff'},
                        style=dict(marginTop='6px'),
                    ),
                ]),

                # ── Consolidated TOMOGRAPHY OPTIONS ───────────────────────────
                html.Div(style=PANEL, children=[
                    html.Span('TOMOGRAPHY OPTIONS', style=LBL),

                    html.Span('HIT QUALITY  >=', style=SUB),
                    dcc.RadioItems(
                        id='hq-slider',
                        className='radio-group',
                        inputClassName='radio-input',
                        labelClassName='radio-label',
                        options=[{'label': str(v), 'value': v}
                                 for v in [0.1, 0.2, 0.3, 0.4, 0.5]],                        

                        value=0.1, inline=True, 
                        labelStyle={
                            'color': '#6ab4ff'
                        }
                    ),

                    html.Span('DEPTH RANGE  (km)', style=SUB),
                    html.Div(style=dict(display='flex', gap='8px', alignItems='center'), children=[
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MIN', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='depth-min', type='text',
                                      value=str(int(_dep_default_min)),
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                        html.Span('→', style=dict(color='#445566', fontSize='14px',
                                                   marginTop='16px')),
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MAX', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='depth-max', type='text',
                                      value=str(int(_dep_default_max)),
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                    ]),

                    html.Span(id='iso-range-header',
                              children=f'{_vel_label}  ISO RANGE', style=SUB),
                    html.Div(id='iso-range-label',
                             style=dict(fontSize='10px', color='#8fa8cc',
                                        marginBottom='6px', letterSpacing='1px')),
                    html.Div(style=dict(display='flex', gap='8px',
                                        alignItems='center'), children=[
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MIN', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='iso-min', type='text',
                                      value=str(round(_iso_init[0], 4)),
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                        html.Span('→', style=dict(color='#445566', fontSize='14px',
                                                   marginTop='16px')),
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MAX', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='iso-max', type='text',
                                      value=str(round(_iso_init[1], 4)),
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                    ]),
                    html.Div(id='iso-data-range',
                             style=dict(marginTop='5px', fontSize='10px',
                                        color='#7a90a8', letterSpacing='1px'),
                             children=f'data: {_vmin_g:.4f} → {_vmax_g:.4f}'),

                    html.Span('VOLUME SMOOTHNESS', style=SUB),
                    dcc.Input(id='n-surfaces', type='number', value=15,
                              min=1, max=30, step=1, debounce=True,
                              style=dict(width='100%', boxSizing='border-box',
                                         background='rgb(10,10,22)',
                                         border='1px solid rgb(55,55,95)',
                                         borderRadius='4px', color='#6ab4ff',
                                         padding='5px 8px', fontSize='14px',
                                         fontFamily='monospace')),

                    html.Span('OPACITY', style=SUB),
                    html.Div(id='opacity-label', style=VAL),
                    dcc.Slider(id='opacity-slider', min=0.1, max=1.0,
                               step=0.05, value=0.6,
                               marks={v: dict(label=str(v),
                                              style=dict(color='#b0c4e8', fontSize='10px'))
                                      for v in [0.1, 0.5, 1.0]},
                               tooltip={'always_visible': False}),

                    html.Span('GRID RESOLUTION', style=SUB),
                    html.Div(id='ngrid-label', style=VAL),
                    html.P('Higher = sharper but slower',
                           style=dict(color='#7a90a8', fontSize='10px',
                                      margin='0 0 4px 0')),
                    dcc.Slider(id='ngrid-slider', min=20, max=100, step=10, value=50,
                               marks={v: dict(label=str(v),
                                              style=dict(color='#b0c4e8', fontSize='10px'))
                                      for v in [20, 50, 80, 100]},
                               tooltip={'always_visible': False}),

                    html.Span('COLORSCALE', style=SUB),
                    dcc.Dropdown(id='cs-picker',
                                 options=[{'label': k, 'value': k}
                                          for k in COLORSCALE_OPTIONS],
                                 value='Seismic (blue-white-red)',
                                 clearable=False,
                                 style=dict(background='rgb(12,12,26)',
                                            color='#ccd8ee', fontSize='12px',
                                            border='1px solid rgb(55,55,95)',
                                            borderRadius='4px')),

                    html.Span('COLOR SCALE RANGE', style=SUB),
                    html.Div(id='cscale-range-label',
                             style=dict(fontSize='10px', color='#8fa8cc',
                                        marginBottom='6px', letterSpacing='1px')),
                    html.Div(style=dict(display='flex', gap='8px',
                                        alignItems='center'), children=[
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MIN', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='cscale-min', type='text',
                                      value='',
                                      placeholder='auto',
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                        html.Span('→', style=dict(color='#445566', fontSize='14px',
                                                   marginTop='16px')),
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MAX', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='cscale-max', type='text',
                                      value='',
                                      placeholder='auto',
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                    ]),
                    html.P('Leave blank to auto-fit to iso range',
                           style=dict(color='#7a90a8', fontSize='10px',
                                      margin='5px 0 0 0')),
                ]),

                # ── Slab upload ───────────────────────────────────────────────
                html.Div(style=PANEL, children=[
                    html.Span('SLAB FILE  (.grd)', style=LBL),
                    dcc.Upload(id='upload-slab',
                               children=html.Div([
                                   html.Span('Drop .grd ',
                                             style=dict(color='#7de8ff')),
                                   html.Span('or click',
                                             style=dict(color='#8fa8cc')),
                               ]),
                               style=dict(border='1px dashed rgb(40,80,100)',
                                          borderRadius='6px', padding='10px',
                                          textAlign='center', cursor='pointer',
                                          fontSize='12px', background='rgb(10,10,22)'),
                               multiple=False),
                    html.Div(id='slab-status',
                             children=_status_div(_slab_msg, ok=(_slab is not None))),
                    dcc.Checklist(
                        id='show-slab',
                        options=[{'label': ' Show slab', 'value': True}],
                        value=[True] if _slab is not None else [],
                        inline=True,
                        inputClassName='radio-input',
                        labelClassName='radio-label',
                        labelStyle={'color': '#6ab4ff'},
                        style=dict(marginTop='10px'),
                    ),
                    html.Span('DISPLAY MODE',
                              style={**SUB, 'marginTop': '10px'}),
                    dcc.RadioItems(
                        id='slab-mode',
                        className='radio-group',
                        inputClassName='radio-input',
                        labelClassName='radio-label',
                        options=[{'label': 'Surface', 'value': 'surface'},
                                 {'label': 'Contours', 'value': 'contours'}],
                        value='contours', inline=True,  
                        labelStyle={
                            'color': '#6ab4ff'
                        }
                    ),
                    html.Span('OPACITY', style={**SUB, 'marginTop': '8px'}),
                    dcc.Slider(id='slab-opacity', min=0.1, max=1.0,
                               step=0.05, value=0.6,
                               marks={v: dict(label=str(v),
                                              style=dict(color='#b0c4e8', fontSize='10px'))
                                      for v in [0.1, 0.5, 1.0]},
                               tooltip={'always_visible': False}),
                ]),

                # ── Volcanoes ─────────────────────────────────────────────────
                html.Div(style=PANEL, children=[
                    html.Span('PLOT VOLCANOES  (.xls GVP)', style=LBL),
                    dcc.Upload(id='upload-vol',
                               children=html.Div([
                                   html.Span('Drop GVP .xls ',
                                             style=dict(color='#ffbb66')),
                                   html.Span('or click',
                                             style=dict(color='#8fa8cc')),
                               ]),
                               style=dict(border='1px dashed rgb(90,65,35)',
                                          borderRadius='6px', padding='10px',
                                          textAlign='center', cursor='pointer',
                                          fontSize='12px', background='rgb(10,10,22)'),
                               multiple=False),
                    html.Div(id='vol-status',
                             children=_status_div(_vol_msg, ok=(_vol_df is not None))),
                    dcc.Checklist(
                        id='show-vol',
                        options=[{'label': ' Show volcanoes', 'value': 'show'}],
                        value=['show'] if _vol_df is not None else [],
                        inline=True,
                        inputClassName='radio-input',
                        labelClassName='radio-label',
                        labelStyle={'color': '#6ab4ff'},
                    ),
                ]),

                # ── Earthquakes ───────────────────────────────────────────────
                html.Div(style=PANEL, children=[
                    html.Span('EARTHQUAKES  (.csv / .pha)', style=LBL),

                    dcc.Upload(
                        id='upload-eq',
                        children=html.Div([
                            html.Span('Drop CSV or PHA ', style=dict(color='#ffdd88')),
                            html.Span('(lat, lon, depth)', style=dict(color='#8fa8cc')),
                        ]),
                        style=dict(
                            border='1px dashed rgb(90,75,30)',
                            borderRadius='6px',
                            padding='10px',
                            textAlign='center',
                            cursor='pointer',
                            fontSize='12px',
                            background='rgb(10,10,22)'
                        ),
                        multiple=False
                    ),

                    html.Div(
                        id='eq-status',
                        children=_status_div(_eq_msg, ok=(_eq_df is not None))
                    ),

                    dcc.Checklist(
                        id='show-eq',
                        options=[{'label': ' Show earthquakes', 'value': 'show'}],
                        value=['show'] if _eq_df is not None else [],
                        inline=True,
                        inputClassName='radio-input',
                        labelClassName='radio-label',
                        labelStyle={'color': '#6ab4ff'},
                    ),

                    # ── Magnitude scaling toggle ────────────────────────────────
                    html.Div(
                        id='eq-mag-row',
                        style=dict(marginTop='6px', display='block') if _eq_has_mag else dict(marginTop='6px', display='none'),
                        children=[

                        dcc.Checklist(
                            id='eq-scale-mag',
                            options=[{'label': ' Scale by magnitude', 'value': 'scale'}],
                            value=[],
                            inline=True,
                            inputClassName='radio-input',
                            labelClassName='radio-label',
                            labelStyle={'color': '#6ab4ff'},
                        ),

                    html.Span('MIN MAGNITUDE', style={**SUB, 'marginTop': '8px'}),

                    dcc.Input(
                        id='eq-min-mag',
                        type='text',
                        value='0',
                        debounce=True,
                        style=dict(
                            width='100%',
                            boxSizing='border-box',
                            background='rgb(10,10,22)',
                            border='1px solid rgb(55,55,95)',
                            borderRadius='4px',
                            color='#6ab4ff',
                            padding='6px 8px',
                            fontSize='12px',
                            fontFamily='monospace',
                            MozAppearance='textfield',
                            appearance='textfield',
                        ),
                    ),

                    html.Span('DEPTH RANGE  (km)', style={**SUB, 'marginTop': '8px'}),
                    html.Div(style=dict(display='flex', gap='8px', alignItems='center'), children=[
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MIN', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='eq-dep-min', type='text',
                                      value=str(int(_dep_min)),
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                        html.Span('→', style=dict(color='#445566', fontSize='14px',
                                                   marginTop='16px')),
                        html.Div(style=dict(flex='1'), children=[
                            html.Span('MAX', style=dict(color='#b0c4e8', fontSize='9px',
                                                         letterSpacing='2px', display='block',
                                                         marginBottom='3px')),
                            dcc.Input(id='eq-dep-max', type='text',
                                      value=str(int(_dep_max)),
                                      debounce=True,
                                      style=dict(width='100%', boxSizing='border-box',
                                                 background='rgb(10,10,22)',
                                                 border='1px solid rgb(55,55,95)',
                                                 borderRadius='4px', color='#6ab4ff',
                                                 padding='5px 8px', fontSize='12px',
                                                 fontFamily='monospace',
                                                 MozAppearance='textfield',
                                                 appearance='textfield')),
                        ]),
                    ]),
                ]
            ),
        ]),
                # ── Extras ────────────────────────────────────────────────────
                html.Div(style=PANEL, children=[
                    html.Span('EXTRAS', style=LBL),
                    dcc.Checklist(
                        id='show-borders',
                        options=[{'label': ' Geographical borders', 'value': 'show'}],
                        value=['show'] if _BORDER_XS is not None else [],
                        inline=True,
                        inputClassName='radio-input',
                        labelClassName='radio-label',
                        labelStyle={'color': '#6ab4ff'},
                    ),
                ]),

                # ── XYZ Anomaly ───────────────────────────────────────────────
                html.Div(style=PANEL, children=[
                    html.Span('XYZ ANOMALY SURFACE  (.xyz / .csv / .txt)', style=LBL),
                    dcc.Upload(id='upload-xyz',
                               children=html.Div([
                                   html.Span('Drop XYZ file ', style=dict(color='#d488ff')),
                                   html.Span('or click',       style=dict(color='#8fa8cc')),
                               ]),
                               style=dict(border='1px dashed rgb(80,40,110)',
                                          borderRadius='6px', padding='10px',
                                          textAlign='center', cursor='pointer',
                                          fontSize='12px', background='rgb(10,10,22)'),
                               multiple=False),
                    html.Div(id='xyz-status',
                             children=_status_div('No XYZ file loaded', ok=False)),
                    dcc.Checklist(
                        id='show-xyz',
                        options=[{'label': ' Show anomaly surface', 'value': 'show'}],
                        value=[],
                        inline=True,
                        inputClassName='radio-input',
                        labelClassName='radio-label',
                        labelStyle={'color': '#d488ff'},
                        style=dict(marginTop='6px'),
                    ),
                    html.Span('OPACITY', style={**SUB, 'marginTop': '8px'}),
                    dcc.Slider(id='xyz-opacity', min=0.1, max=1.0, step=0.05, value=0.7,
                               marks={v: dict(label=str(v),
                                              style=dict(color='#b0c4e8', fontSize='10px'))
                                      for v in [0.1, 0.5, 1.0]},
                               tooltip={'always_visible': False}),
                    html.Span('COLORSCALE', style={**SUB, 'marginTop': '4px'}),
                    dcc.Dropdown(id='xyz-colorscale',
                                 options=[{'label': c, 'value': c}
                                          for c in ['Plasma','Viridis','Hot','Turbo','RdBu','Magma']],
                                 value='Plasma', clearable=False,
                                 style=dict(background='rgb(12,12,26)', color='#ccd8ee',
                                            fontSize='12px', border='1px solid rgb(55,55,95)',
                                            borderRadius='4px')),
                    html.Span('GRID RESOLUTION', style={**SUB, 'marginTop': '4px'}),
                    dcc.Slider(id='xyz-ngrid', min=15, max=50, step=5, value=30,
                               marks={v: dict(label=str(v),
                                              style=dict(color='#b0c4e8', fontSize='10px'))
                                      for v in [20, 60, 120]},
                               tooltip={'always_visible': False}),
                ]),

                # ── Stats ─────────────────────────────────────────────────────
                html.Div(id='stats-panel',
                         style=dict(**PANEL, fontSize='11px',
                                    lineHeight='2', color='#b0c4e8')),
            ]),

            # ── Graph ─────────────────────────────────────────────────────────
            html.Div(style=dict(flex='1', minWidth='0',
                                 position='sticky', top='0'), children=[
                dcc.Loading(id='loading', type='circle', color='#4af',
                            children=dcc.Graph(
                                id='seismic-graph', figure=_fig0,
                                config={'scrollZoom': True, 'displayModeBar': True,
                                        'modeBarButtonsToRemove': ['select2d', 'lasso2d']},
                                style=dict(height='750px', width='100%'),
                            )),
            ]),
        ]),

        dcc.Store(id='tomo-store', data=_df.to_dict('list')),
        dcc.Store(id='vel-label',  data=_vel_label),
        dcc.Store(id='slab-store', data=_encode_slab(_slab) if _slab is not None else None),
        dcc.Store(id='vol-store',  data=_vol_df.to_dict('list') if _vol_df is not None else None),
        dcc.Store(id='eq-store',   data=_eq_df.to_dict('list') if _eq_df is not None else None),
        dcc.Store(id='xyz-store',  data=None),
    ]
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('tomo-store',  'data'),
    Output('vel-label',   'data'),
    Output('tomo-status', 'children'),
    Output('iso-min',     'value'),
    Output('iso-max',     'value'),
    Output('iso-data-range', 'children'),
    Output('depth-min',   'value'),
    Output('depth-max',   'value'),
    Output('lat-min',     'value'),
    Output('lat-max',     'value'),
    Output('lon-min',     'value'),
    Output('lon-max',     'value'),
    Input('upload-tomo',  'contents'),
    State('upload-tomo',  'filename'),
    prevent_initial_call=True,
)
def load_tomo(contents, filename):
    if not contents:
        raise dash.exceptions.PreventUpdate
    _, b64 = contents.split(',', 1)
    df, vel_label, err = parse_tomo(base64.b64decode(b64), wave_type='vp')
    if err:
        return ([dash.no_update] * 2 + [_status_div(f'Error: {err}', ok=False)] +
                [dash.no_update] * 9)
    vmin, vmax = float(df['%dvp'].min()), float(df['%dvp'].max())
    depths = sorted(df['dep'].unique())
    dep_val_min = depths[1]  if len(depths) > 2 else depths[0]
    dep_val_max = depths[-2] if len(depths) > 2 else depths[-1]
    iso_min_val = str(round(vmax * 0.3, 4))
    iso_max_val = str(round(vmax * 0.8, 4))
    data_range_txt = f'data: {vmin:.4f} → {vmax:.4f}'
    lat_min_v = str(round(float(df['lat'].min()), 2))
    lat_max_v = str(round(float(df['lat'].max()), 2))
    lon_min_v = str(round(float(df['lon'].min()), 2))
    lon_max_v = str(round(float(df['lon'].max()), 2))
    return (
        df.to_dict('list'),
        vel_label,
        _status_div(f'{filename}  ({len(df):,} pts)  [{vel_label}]'),
        iso_min_val, iso_max_val, data_range_txt,
        str(int(dep_val_min)), str(int(dep_val_max)),
        lat_min_v, lat_max_v, lon_min_v, lon_max_v,
    )


@app.callback(
    Output('slab-store',  'data'),
    Output('slab-status', 'children'),
    Input('upload-slab',  'contents'),
    State('upload-slab',  'filename'),
    prevent_initial_call=True,
)
def load_slab(contents, filename):
    if not contents:
        raise dash.exceptions.PreventUpdate
    _, b64 = contents.split(',', 1)
    slab, err = parse_grd(base64.b64decode(b64))
    if err:
        return dash.no_update, _status_div(f'Error: {err}', ok=False)
    lo, la, de = slab
    msg = (f'{filename}  '
           f'lon [{lo.min():.1f}, {lo.max():.1f}]  '
           f'lat [{la.min():.1f}, {la.max():.1f}]')
    return _encode_slab(slab), _status_div(msg)


@app.callback(
    Output('vol-store',  'data'),
    Output('vol-status', 'children'),
    Input('upload-vol',  'contents'),
    Input('show-vol', 'value'),
    State('upload-vol',  'filename'),
    prevent_initial_call=True,
)
def load_volcanoes(contents, show_vol, filename):
    if not contents:
        raise dash.exceptions.PreventUpdate
    _, b64 = contents.split(',', 1)
    vol_df, err = read_volcanoes(base64.b64decode(b64))
    if err:
        return dash.no_update, _status_div(f'Error: {err}', ok=False)
    return (
        vol_df.to_dict('list'),
        _status_div(f'{filename}  ({len(vol_df):,} volcanoes)'),
    )


@app.callback(
    Output('eq-store',  'data'),
    Output('eq-status', 'children'),
    Output('eq-mag-row', 'style'),
    Input('upload-eq',  'contents'),
    State('upload-eq',  'filename'),
    prevent_initial_call=True,
)
def load_earthquakes(contents, filename):
    if not contents:
        raise dash.exceptions.PreventUpdate
    _, b64 = contents.split(',', 1)
    eq_df, has_mag, err = parse_earthquake_file(base64.b64decode(b64), filename)
    if err:
        return dash.no_update, _status_div(f'Error: {err}', ok=False), {'display': 'none'}
    mag_style = dict(marginTop='6px', display='block') if has_mag else {'display': 'none'}
    msg = f'{filename}  ({len(eq_df):,} events)'
    if has_mag:
        msg += '  [mag col found]'
    return eq_df.to_dict('list'), _status_div(msg), mag_style





@app.callback(
    Output('xyz-store',  'data'),
    Output('xyz-status', 'children'),
    Input('upload-xyz',  'contents'),
    State('upload-xyz',  'filename'),
    prevent_initial_call=True,
)
def load_xyz(contents, filename):
    if not contents:
        raise dash.exceptions.PreventUpdate
    _, b64 = contents.split(',', 1)
    xyz_df, err = parse_xyz(base64.b64decode(b64))
    if err:
        return dash.no_update, _status_div(f'Error: {err}', ok=False)
    msg = (f'{filename}  ({len(xyz_df):,} pts)  '
           f'X[{xyz_df["x"].min():.2f},{xyz_df["x"].max():.2f}]  '
           f'Z[{xyz_df["z"].min():.3f},{xyz_df["z"].max():.3f}]')
    return xyz_df.to_dict('list'), _status_div(msg)


@app.callback(
    Output('seismic-graph',     'figure'),
    Output('stats-panel',       'children'),
    Output('opacity-label',     'children'),
    Output('iso-range-label',   'children'),
    Output('ngrid-label',       'children'),
    Output('iso-range-header',  'children'),
    Output('cscale-range-label','children'),
    Input('tomo-store',    'data'),
    Input('vel-label',     'data'),
    Input('slab-store',    'data'),
    Input('vol-store',     'data'),
    Input('hq-slider',     'value'),
    Input('iso-min',       'value'),
    Input('iso-max',       'value'),
    Input('n-surfaces',    'value'),
    Input('opacity-slider','value'),
    Input('cs-picker',     'value'),
    Input('depth-min',      'value'),
    Input('depth-max',      'value'),
    Input('ngrid-slider',  'value'),
    Input('slab-opacity',  'value'),
    Input('slab-mode',     'value'),
    Input('eq-store',      'data'),
    Input('eq-min-mag',    'value'),
    Input('show-vol', 'value'),
    Input('show-eq', 'value'),
    Input('eq-scale-mag', 'value'),
    Input('show-slab', 'value'),
    Input('show-borders', 'value'),
    Input('show-tomo',   'value'),
    Input('xyz-store',    'data'),
    Input('show-xyz',     'value'),
    Input('xyz-opacity',  'value'),
    Input('xyz-colorscale', 'value'),
    Input('xyz-ngrid',    'value'),
    Input('lat-min',      'value'),
    Input('lat-max',      'value'),
    Input('lon-min',      'value'),
    Input('lon-max',      'value'),
    Input('eq-dep-min',   'value'),
    Input('eq-dep-max',   'value'),
    Input('cscale-min',   'value'),
    Input('cscale-max',   'value'),
)
def update_figure(tomo_store, vel_label, slab_store, vol_store, hq_min, iso_min, iso_max,
                  n_surf, opacity, cs_name, depth_min_val, depth_max_val, ngrid,
                  slab_op, slab_mode, eq_store, eq_min_mag,
                  show_vol_val, show_eq_val, eq_scale_val, show_slab_val, show_borders_val,
                  show_tomo_val,
                  xyz_store, show_xyz_val, xyz_opacity, xyz_colorscale, xyz_ngrid,
                  lat_min_val, lat_max_val, lon_min_val, lon_max_val,
                  eq_dep_min_val, eq_dep_max_val,
                  cscale_min_val, cscale_max_val):
    df = pd.DataFrame(tomo_store)
    vel_label = vel_label or '%dVp'
    slab = _decode_slab(slab_store)
    vol_df = pd.DataFrame(vol_store) if vol_store else None
    show_slab = bool(show_slab_val)
    show_vol = bool(show_vol_val)
    show_borders = bool(show_borders_val)
    show_tomo    = bool(show_tomo_val)
    eq_df        = pd.DataFrame(eq_store) if eq_store else None
    if eq_df is not None:
        try:
            min_mag = float(eq_min_mag) if eq_min_mag not in (None, '') else 0.0
        except (TypeError, ValueError):
            min_mag = 0.0
        if 'magnitude' in eq_df.columns:
            eq_df = eq_df[eq_df['magnitude'].fillna(0) >= min_mag]
    show_eq      = bool(show_eq_val)
    eq_scale_mag = bool(eq_scale_val)
    xyz_df       = pd.DataFrame(xyz_store) if xyz_store else None
    show_xyz     = bool(show_xyz_val)
    xyz_opacity  = xyz_opacity  if xyz_opacity  is not None else 0.7
    xyz_colorscale = xyz_colorscale or 'Plasma'
    xyz_ngrid    = xyz_ngrid    if xyz_ngrid    is not None else 60

    # Guard against None if inputs not yet set — inputs are now type='text', parse as float
    def _parse_float(val, fallback):
        try:
            return float(val)
        except (TypeError, ValueError):
            return fallback

    iso_min = _parse_float(iso_min, _vmin_g)
    iso_max = _parse_float(iso_max, _vmax_g)

    depth_range = [
        _parse_float(depth_min_val, float(_dep_min)),
        _parse_float(depth_max_val, float(_dep_max)),
    ]

    # Lat/lon limits (fall back to data extents if not set)
    lat_range = [
        _parse_float(lat_min_val, float(df['lat'].min())),
        _parse_float(lat_max_val, float(df['lat'].max())),
    ]
    lon_range = [
        _parse_float(lon_min_val, float(df['lon'].min())),
        _parse_float(lon_max_val, float(df['lon'].max())),
    ]
    eq_dep_min = _parse_float(eq_dep_min_val, float(_dep_min))
    eq_dep_max = _parse_float(eq_dep_max_val, float(_dep_max))

    # Color scale bounds (None = auto, i.e. match iso range)
    cscale_min = _parse_float(cscale_min_val, None) if cscale_min_val not in (None, '') else None
    cscale_max = _parse_float(cscale_max_val, None) if cscale_max_val not in (None, '') else None

    # Apply lat/lon/depth limits to earthquakes
    if eq_df is not None and len(eq_df):
        eq_df = eq_df[
            (eq_df['lat']   >= lat_range[0]) & (eq_df['lat']   <= lat_range[1]) &
            (eq_df['lon']   >= lon_range[0]) & (eq_df['lon']   <= lon_range[1]) &
            (eq_df['depth'] >= eq_dep_min)   & (eq_df['depth'] <= eq_dep_max)
        ]

    fig = build_figure(
        df, hq_min, iso_min, iso_max,
        n_surf, opacity, cs_name, depth_range, ngrid,
        lat_range=lat_range, lon_range=lon_range,
        show_tomo=show_tomo,
        slab_data=slab, show_slab=show_slab, slab_opacity=slab_op, slab_mode=slab_mode or 'surface',
        vol_df=vol_df, show_volcanoes=show_vol,
        vel_label=vel_label, show_borders=show_borders,
        eq_df=eq_df, show_earthquakes=show_eq, eq_scale_mag=eq_scale_mag,
        xyz_df=xyz_df, show_xyz=show_xyz, xyz_opacity=xyz_opacity,
        xyz_colorscale=xyz_colorscale, xyz_ngrid=xyz_ngrid,
        cscale_min=cscale_min, cscale_max=cscale_max,
    )

    sub = df[
        (df['hq'] >= hq_min) &
        (df['dep'] >= depth_range[0]) &
        (df['dep'] <= depth_range[1]) &
        (df['lat'] >= lat_range[0]) &
        (df['lat'] <= lat_range[1]) &
        (df['lon'] >= lon_range[0]) &
        (df['lon'] <= lon_range[1])
    ]
    v = sub['%dvp'].values if len(sub) else np.array([0.])
    stats = [
        html.Div(f'TOMO PTS   {len(sub):,}', style=dict(color='#4af')),
        html.Div(f'{vel_label} MIN   {v.min():.4f}'),
        html.Div(f'{vel_label} MAX   {v.max():.4f}'),
        html.Div(f'{vel_label} MEAN  {v.mean():.4f}'),
        html.Div(f'{vel_label} STD   {v.std():.4f}'),
        html.Div(f'DEPTH      {int(depth_range[0])}-{int(depth_range[1])} km' if depth_range else 'DEPTH  —'),
        html.Div(f'LAT        {lat_range[0]:.2f} to {lat_range[1]:.2f}'),
        html.Div(f'LON        {lon_range[0]:.2f} to {lon_range[1]:.2f}'),
    ]
    if slab is not None:
        lo, la, de = slab
        valid = de[~np.isnan(de)]
        stats += [
            html.Div(f'SLAB DEPTH {valid.min():.0f}-{valid.max():.0f} km',
                     style=dict(color='#7df')),
        ]
    if vol_df is not None and show_vol:
        clipped = filter_volcanoes_to_region(
            vol_df,
            lat_min=lat_range[0], lat_max=lat_range[1],
            lon_min=lon_range[0], lon_max=lon_range[1],
        )
        stats += [
            html.Div(f'VOLCANOES  {len(clipped)} in region',
                     style=dict(color='#fa6')),
        ]
    if eq_df is not None and show_eq:
        stats += [
            html.Div(f'EARTHQUAKES  {len(eq_df):,} events  '
                     f'dep {eq_dep_min:.0f}-{eq_dep_max:.0f} km',
                     style=dict(color='#ffdd88')),
        ]
    if xyz_df is not None and show_xyz:
        stats += [
            html.Div(f'XYZ ANOMALY  {len(xyz_df):,} pts  '
                     f'Z[{xyz_df["z"].min():.3f}, {xyz_df["z"].max():.3f}]',
                     style=dict(color='#d488ff')),
        ]

    if cscale_min is not None and cscale_max is not None:
        cscale_label = f'{cscale_min:.4f}  →  {cscale_max:.4f}'
    elif cscale_min is not None:
        cscale_label = f'{cscale_min:.4f}  →  auto'
    elif cscale_max is not None:
        cscale_label = f'auto  →  {cscale_max:.4f}'
    else:
        cscale_label = 'auto (follows iso range)'

    return (
        fig, stats,
        f'{opacity:.0%}',
        f'{vel_label}  {iso_min:.4f}  →  {iso_max:.4f}',
        f'{ngrid} x {ngrid}',
        f'{vel_label}  ISO RANGE',
        cscale_label,
    )


if __name__ == '__main__':
    app.run(debug=True)
