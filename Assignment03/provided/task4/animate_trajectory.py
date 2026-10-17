"""Task 4 · Reconstruct the Flight: an interactive 3D animation of the lander.
PROVIDED: you do not need to change this file. Not graded.

The optional last cell of Task 4.4 in task4.ipynb runs it, after the
comparison (it reuses the tracks the comparison saves in outputs/task4/compare/,
so it takes seconds). From a terminal:

    python -m provided.task4.animate_trajectory

It writes outputs/task4/trajectory_animation.html. Open it in any web
browser (it works offline). Press Play or drag the slider to fly the descent
frame by frame; drag the scene to rotate it and scroll to zoom; click a
tracker in the legend to hide it; the buttons switch between preset views and
between a linear and a log altitude axis (log spreads out the last hundred
metres, where the lander hovers and lands).

All three trackers of Task 3 fly in the same scene, one colour each, so you
can see what each tracker's tracks turn into. The pyramid under the
SIFT-guided lander is the camera's view: the patch of flat ground the
straight-down camera model says the image covers at that altitude.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from provided.io import load_tracks
from provided.runner import CONSTANTS, ROOT, output_dir
from provided.task4.compare_trackers import fill_gaps
from provided.task4.run import TRACKER_NAMES, fit_all_pairs, fly, motion_arrays

try:
    import plotly.graph_objects as go
except ImportError:          # the rest of the assignment does not need plotly
    go = None

COLOURS = {"one-step": "#ff9f43", "iterative": "#54a0ff", "sift-guided": "#2ee6a6"}
LINE_WIDTH = {"one-step": 4, "iterative": 4, "sift-guided": 7}
BG, WALL, FLOOR, GRID, INK, MUTED = ("#0a0e16", "#0f1522", "#161e2d", "#243047", "#e6ebf2",
                                     "#8b97aa")
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def load_flights(compare_dir, constants):
    """Re-fit and fly every tracker from the tracks cached by task4.compare_trackers.

    Returns {tracker: flight} (see task4.run.fly) with two extra entries per
    flight: "headings" (radians, one per frame, the lander's turn about Z) and
    "image" (width, height, scale of the tracked frames). None if the cache is
    missing.
    """
    flights = {}
    for name in TRACKER_NAMES:
        path = compare_dir / f"tracks_{name}.npz"
        if not path.exists():
            print(f"no saved tracks for {name}: run the Task 4.4 comparison first "
                  f"(the compare cell of task4.ipynb, or python -m provided.task4.compare_trackers)")
            return None
        tracks, info = load_tracks(path)
        centre = np.array([(info["width"] - 1) / 2, (info["height"] - 1) / 2])
        rows, _ = fit_all_pairs(tracks, centre, verbose=False)
        first, last = min(tracks)[0], max(tracks)[1]
        rows, _ = fill_gaps(rows, first, last)
        arrays = motion_arrays(rows)
        flight = fly(*arrays, constants, float(info["scale"]), verbose=False)
        flight["headings"] = np.r_[0.0, np.cumsum(arrays[2])]
        flight["image"] = (float(info["width"]), float(info["height"]), float(info["scale"]))
        flights[name] = flight
    return flights


def _rgba(hex_colour, alpha):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _footprint(position, heading, image, focal_length):
    """The four ground corners the image covers, from the straight-down camera model.

    A ground point G appears at x = (f / Z) D R(-psi)(G - C) (provided/task4/trajectory.py),
    so the image corners x = (+-w/2, +-h/2) sit at G = C + (Z / f) R(psi) D x.
    """
    width, height, scale = image
    f = focal_length * scale
    c, s = np.cos(heading), np.sin(heading)
    rot_flip = np.array([[c, -s], [s, c]]) @ np.diag([1.0, -1.0])
    corners_px = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [width / 2, height / 2]
    return position[:2] + (position[2] / f) * corners_px @ rot_flip.T


def _frustum(position, corners, ground):
    """One polyline: the footprint rectangle and the four edges up to the camera."""
    xs, ys, zs = [], [], []
    for cx, cy in [*corners, corners[0]]:
        xs.append(cx), ys.append(cy), zs.append(ground)
    for cx, cy in corners:
        xs += [None, position[0], cx]
        ys += [None, position[1], cy]
        zs += [None, position[2], ground]
    return xs, ys, zs


def build_figure(flights, trackers, constants, frame_ms=90, camera_tracker="sift-guided"):
    names = [t for t in trackers if t in flights]
    hero = camera_tracker if camera_tracker in names else None
    positions = {t: np.asarray(flights[t]["positions"], dtype=float) for t in names}
    frame_numbers = [int(f) for f in flights[names[0]]["frames"]]
    n = min(len(p) for p in positions.values())
    unit = flights[names[0]]["length_unit"]
    u = " m" if unit == "m" else ""
    focal = constants.get("focal_length_px")
    show_camera = hero is not None and focal and flights[hero]["horizontal"]

    everything = np.vstack(list(positions.values()))
    z_top = float(everything[:, 2].max())
    ground = 0.5 * float(everything[:, 2].min())       # the floor, kept > 0 for the log axis
    corners = {}
    if show_camera:
        corners = [_footprint(positions[hero][k], flights[hero]["headings"][k],
                              flights[hero]["image"], float(focal)) for k in range(n)]
        everything = np.vstack([everything, np.c_[np.vstack(corners), np.zeros(4 * n)]])
    # a square ground plane, so the ground track keeps its true shape
    lo, hi = everything[:, :2].min(axis=0), everything[:, :2].max(axis=0)
    mid, half = (lo + hi) / 2, 0.53 * max(float((hi - lo).max()), 1.0)
    x_range, y_range = [mid[0] - half, mid[0] + half], [mid[1] - half, mid[1] + half]
    z_linear = [0.0, 1.06 * z_top]
    z_log = [float(np.log10(ground)), float(np.log10(1.25 * z_top))]
    z_aspect = float(np.clip(z_linear[1] / (2 * half), 0.55, 1.25))

    fig = go.Figure()

    # static: the full path of every tracker, faint, so the whole flight is always visible
    for t in names:
        p = positions[t]
        fig.add_trace(go.Scatter3d(
            x=p[:n, 0], y=p[:n, 1], z=p[:n, 2], mode="lines", legendgroup=t, showlegend=False,
            line=dict(color=_rgba(COLOURS[t], 0.32), width=2), hoverinfo="skip"))
    n_static = len(fig.data)

    def animated(k):
        """Every trace that moves, for frame index k."""
        out = []
        for t in names:
            p, colour = positions[t], COLOURS[t]
            trail = p[:k + 1]
            custom = frame_numbers[:k + 1]
            out.append(go.Scatter3d(                                    # trail so far
                x=trail[:, 0], y=trail[:, 1], z=trail[:, 2], customdata=custom,
                mode="lines", name=t, legendgroup=t, showlegend=True,
                line=dict(color=colour, width=LINE_WIDTH[t]),
                hovertemplate=(f"<b>{t}</b><br>frame %{{customdata}}<br>X %{{x:.0f}}{u}, "
                               f"Y %{{y:.0f}}{u}<br>altitude %{{z:.1f}}{u}<extra></extra>")))
            out.append(go.Scatter3d(                                    # shadow on the ground
                x=trail[:, 0], y=trail[:, 1], z=np.full(len(trail), ground), mode="lines",
                legendgroup=t, showlegend=False, hoverinfo="skip",
                line=dict(color=_rgba(colour, 0.45), width=2)))
            x, y, z = p[k]
            out.append(go.Scatter3d(                                    # drop line
                x=[x, x], y=[y, y], z=[z, ground], mode="lines", legendgroup=t,
                showlegend=False, hoverinfo="skip",
                line=dict(color=_rgba(colour, 0.55), width=2, dash="dot")))
            out.append(go.Scatter3d(                                    # the lander
                x=[x], y=[y], z=[z], mode="markers", legendgroup=t, showlegend=False,
                customdata=[frame_numbers[k]],
                marker=dict(symbol="diamond", size=11 if t == hero else 8, color=colour,
                            line=dict(color="#ffffff", width=1.5)),
                hovertemplate=(f"<b>{t}</b><br>frame %{{customdata}}<br>altitude "
                               f"%{{z:.1f}}{u}<extra></extra>")))
        if show_camera:
            fx, fy, fz = _frustum(positions[hero][k], corners[k], ground)
            out.append(go.Scatter3d(
                x=fx, y=fy, z=fz, mode="lines", legendgroup=hero, showlegend=False,
                hoverinfo="skip", line=dict(color=_rgba(COLOURS[hero], 0.5), width=2)))
            c = corners[k]
            out.append(go.Mesh3d(
                x=c[:, 0], y=c[:, 1], z=np.full(4, ground), i=[0, 0], j=[1, 2], k=[2, 3],
                color=COLOURS[hero], opacity=0.16, legendgroup=hero, showlegend=False,
                hoverinfo="skip", flatshading=True))
        return out

    def readout(k):
        parts = [f"<b>Frame {frame_numbers[k]}</b>"]
        for t in sorted(names, key=lambda t: t != hero):
            parts.append(f"<span style='color:{COLOURS[t]}'>{t}</span>  "
                         f"{positions[t][k, 2]:,.0f}{u}")
        return "      ".join(parts)

    for trace in animated(0):
        fig.add_trace(trace)
    moving = list(range(n_static, len(fig.data)))
    fig.frames = [go.Frame(name=str(frame_numbers[k]), data=animated(k), traces=moving,
                           layout=dict(title_text=readout(k)))
                  for k in range(n)]

    # labels in the scene: where the flight starts and ends, and the one altitude reading
    notes = []
    ref = positions[hero] if hero else positions[names[0]]
    notes.append(dict(x=ref[0, 0], y=ref[0, 1], z=ref[0, 2], ax=50, ay=-30,
                      text=f"start, frame {frame_numbers[0]}"))
    notes.append(dict(x=ref[n - 1, 0], y=ref[n - 1, 1], z=ref[n - 1, 2], ax=70, ay=40,
                      text=f"last frame {frame_numbers[n - 1]}"))
    anchor = constants.get("anchor_frame")
    if anchor in frame_numbers and unit == "m":
        k = frame_numbers.index(anchor)
        notes.append(dict(x=ref[k, 0], y=ref[k, 1], z=ref[k, 2], ax=90, ay=-45,
                          text=f"{constants.get('anchor_altitude_m')} m reading, frame {anchor}"))
    for note in notes:
        note.update(showarrow=True, arrowcolor=MUTED, arrowwidth=1, arrowhead=0,
                    font=dict(color=MUTED, size=12, family=FONT), bgcolor="rgba(10,14,22,0.7)")
    # on a log axis, scene annotations are placed in log10 units
    notes_log = [dict(note, z=float(np.log10(note["z"]))) for note in notes]

    axis = dict(backgroundcolor=WALL, showbackground=True, gridcolor=GRID, zerolinecolor=GRID,
                color=MUTED, title_font=dict(size=13, color=INK, family=FONT),
                tickfont=dict(size=11, color=MUTED, family=FONT))
    up = dict(x=0, y=0, z=1)
    perspective, flat = dict(type="perspective"), dict(type="orthographic")
    low = dict(x=0, y=0, z=-0.12)          # look a little below the middle of the box
    views = {   # the flat views are orthographic, so they read like 2D plots
        "Oblique": dict(eye=dict(x=1.5, y=-1.35, z=0.55), center=low, up=up,
                        projection=perspective),
        "Descent profile": dict(eye=dict(x=1.2, y=0.0, z=0.0), center=dict(x=0, y=0, z=0),
                                up=up, projection=flat),
        "From behind": dict(eye=dict(x=0.3, y=-1.95, z=0.45), center=low, up=up,
                            projection=perspective),
        # straight down, a hair to the south: an eye exactly on the Z axis leaves the
        # screen orientation undefined, this keeps X to the right and Y up
        "Top down": dict(eye=dict(x=0.0, y=-0.01, z=1.15), center=dict(x=0, y=0, z=0),
                         up=up, projection=flat),
    }
    x_title = f"X{u.strip() and ' (m)'}, right in frame {frame_numbers[0]}"
    y_title = f"Y{u.strip() and ' (m)'}, up in frame {frame_numbers[0]}"
    z_title = f"altitude Z{u.strip() and ' (m)'}"
    end_on = {"Descent profile": "x", "Top down": "z"}   # the axis that points at you
    aspect = dict(x=1, y=1, z=z_aspect)

    def view_args(name):
        hidden = end_on.get(name)
        # an orthographic camera does not zoom with distance, so the flat views
        # enlarge the box instead
        grow = 1.7 if views[name]["projection"]["type"] == "orthographic" else 1.0
        args = {"scene.camera": views[name],
                "scene.aspectratio": {k: v * grow for k, v in aspect.items()}}
        for axis_name, title in (("x", x_title), ("y", y_title), ("z", z_title)):
            args[f"scene.{axis_name}axis.showticklabels"] = axis_name != hidden
            args[f"scene.{axis_name}axis.title.text"] = "" if axis_name == hidden else title
        return args
    play = dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True,
                transition=dict(duration=0), mode="immediate")
    pause = dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                 mode="immediate")
    button_style = dict(bgcolor=FLOOR, bordercolor=GRID, font=dict(color=INK, size=12, family=FONT),
                        showactive=False, pad=dict(r=6, t=6, b=6, l=6))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color=INK),
        title=dict(text=readout(0), x=0.02, xanchor="left", y=0.965,
                   font=dict(size=16, color=INK, family=FONT)),
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.99, xanchor="right", y=0.93, bgcolor="rgba(15,21,34,0.85)",
                    bordercolor=GRID, borderwidth=1, font=dict(size=13, color=INK),
                    title=dict(text="tracker (click to hide)", font=dict(size=11, color=MUTED))),
        scene=dict(
            xaxis=dict(axis, title=x_title, range=x_range),
            yaxis=dict(axis, title=y_title, range=y_range),
            zaxis=dict(axis, backgroundcolor=FLOOR, title=z_title, range=z_linear,
                       type="linear"),
            domain=dict(x=[0.09, 1.0], y=[0.08, 0.93]),
            aspectmode="manual", aspectratio=aspect,
            camera=views["Oblique"], dragmode="turntable", annotations=notes),
        updatemenus=[
            dict(type="buttons", direction="left", x=0.02, xanchor="left", y=0.035,
                 yanchor="bottom", buttons=[
                     dict(label="▶  Play", method="animate", args=[None, play]),
                     dict(label="❚❚  Pause", method="animate", args=[[None], pause]),
                     dict(label="⟲  Rewind", method="animate",
                          args=[[str(frame_numbers[0])], pause])], **button_style),
            dict(type="buttons", direction="down", x=0.01, xanchor="left", y=0.86,
                 yanchor="top", buttons=[dict(label=name, method="relayout",
                                              args=[view_args(name)])
                                         for name in views], **button_style),
            dict(type="buttons", direction="down", x=0.01, xanchor="left", y=0.52,
                 yanchor="top", buttons=[
                     dict(label="Linear altitude", method="relayout",
                          args=[{"scene.zaxis.type": "linear", "scene.zaxis.range": z_linear,
                                 "scene.annotations": notes}]),
                     dict(label="Log altitude", method="relayout",
                          args=[{"scene.zaxis.type": "log", "scene.zaxis.range": z_log,
                                 "scene.annotations": notes_log}])],
                 **button_style)],
        sliders=[dict(
            active=0, x=0.22, len=0.76, y=0.02, yanchor="bottom", pad=dict(t=0, b=6),
            bgcolor=FLOOR, bordercolor=GRID, activebgcolor=COLOURS.get(hero, INK),
            tickcolor=GRID, font=dict(color="rgba(0,0,0,0)", size=1),
            currentvalue=dict(prefix="Frame ", font=dict(color=INK, size=13, family=FONT),
                              xanchor="right"),
            steps=[dict(label=str(f), method="animate", args=[[str(f)], pause])
                   for f in frame_numbers[:n]])],
        annotations=[dict(
            text="Drag to rotate · scroll to zoom · hover for values", x=0.99, xanchor="right",
            y=0.965, yanchor="middle", xref="paper", yref="paper", showarrow=False,
            font=dict(size=12, color=MUTED, family=FONT))],
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Task 4: interactive 3D animation of the lander")
    parser.add_argument("--compare-dir", type=Path, default=ROOT / "outputs" / "task4" / "compare")
    parser.add_argument("--constants", type=Path, default=CONSTANTS)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "task4")
    parser.add_argument("--frame-ms", type=int, default=90, help="milliseconds per frame on Play")
    parser.add_argument("--camera", choices=[*TRACKER_NAMES, "none"], default="sift-guided",
                        help="whose camera view pyramid to draw")
    parser.add_argument("--cdn", action="store_true",
                        help="load plotly.js from the internet: a much smaller file, but "
                             "it needs a connection to open")
    args = parser.parse_args(argv)
    if go is None:
        print("this script needs plotly: pip install plotly (see requirements.txt)")
        return
    output_dir(args.out_dir)
    constants = json.loads(args.constants.read_text(encoding="utf-8")) if args.constants.exists() else {}

    flights = load_flights(args.compare_dir, constants)
    if flights is None:
        return
    fig = build_figure(flights, list(TRACKER_NAMES), constants, args.frame_ms, args.camera)
    out = args.out_dir / "trajectory_animation.html"
    fig.write_html(out, include_plotlyjs="cdn" if args.cdn else True, auto_play=False,
                   full_html=True, default_width="100%", default_height="100vh",
                   config=dict(displaylogo=False, responsive=True,
                               modeBarButtonsToRemove=["resetCameraLastSave3d"]))
    print(f"  saved {out}  (open it in a web browser)")
    print("done")


if __name__ == "__main__":
    main()
