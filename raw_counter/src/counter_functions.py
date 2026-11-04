import cv2
import os
import subprocess
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from IPython import get_ipython
import ipywidgets as widgets
from IPython.display import display
from glob import glob
from datetime import datetime
import time
from jupyter_ui_poll import ui_events



def movie_fps(path):
    """Label frame rate, real average frame rate, number of frames and duration of a movie"""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,duration",
         "-of", "default=nw=1", path],
        capture_output=True, text=True).stdout
    info = dict(line.split("=") for line in out.split())

    def fraction(text):                         # "30/1" -> 30.0, "14200000/986709" -> 14.39
        num, den = text.split("/")
        return float(num) / float(den)

    return (fraction(info["r_frame_rate"]), fraction(info["avg_frame_rate"]),
            int(info["nb_frames"]), float(info["duration"]))

# r_fps, avg_fps, n_frames, duration = movie_fps(movie_loc)
# print(f"r_frame_rate (label): {r_fps:.2f} fps")
# print(f"avg_frame_rate (real): {avg_fps:.2f} fps")
# print(f"{n_frames} frames, {duration:.2f} s")


def extract_frames(movie, out_dir, start=None, duration=None, from_end=None):
    """Save frames of a movie as grayscale PNGs in out_dir (f_0001.png, f_0002.png, ...).
    start/duration: seconds from the beginning; from_end: the last N seconds"""
    os.makedirs(out_dir, exist_ok=True)                 # create the folder if needed
    cmd = ["ffmpeg", "-v", "error", "-y"]               # only show errors; overwrite old files
    if from_end is not None:
        cmd += ["-sseof", f"-{from_end}"]               # start N seconds before the end
    if start is not None:
        cmd += ["-ss", str(start)]                      # start at this time
    cmd += ["-i", movie]
    if duration is not None:
        cmd += ["-t", str(duration)]                    # only this many seconds
    cmd += ["-vf", "format=gray", os.path.join(out_dir, "f_%04d.png")]
    subprocess.run(cmd, check=True)                     # check=True: raise an error if ffmpeg fails
    print(f"{len(os.listdir(out_dir))} frames in {out_dir}")

# first 10 seconds
#extract_frames("movie.mp4", "frames_start", duration=3)
# last 3 seconds
#extract_frames("movie.mp4", "frames_end", from_end=3)
# extract_frames(movie_loc, f"{home_dir}/{date}/{date}_frames_start", duration=3)

def align_background(background, last, rows=2, cols=4, max_shift=30, min_response=0.1):
    """Shift the background to match the last frame, separately in each region
    (rows x cols = the 2 rows of 4 mazes), because the mazes can move a few pixels during the movie"""
    h, w = background.shape
    aligned = background.copy()
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            b = background[y0:y1, x0:x1].astype(np.float32)
            l = last[y0:y1, x0:x1].astype(np.float32)
            (dx, dy), response = cv2.phaseCorrelate(b, l, cv2.createHanningWindow(b.shape[::-1], cv2.CV_32F))
            ok = response >= min_response and abs(dx) <= max_shift and abs(dy) <= max_shift   # skip unreliable shifts
            if ok:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(background, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                aligned[y0:y1, x0:x1] = shifted[y0:y1, x0:x1]
            print(f"region {r},{c}: shift dx={dx:+.1f} dy={dy:+.1f} px (response {response:.2f})"
                  + ("" if ok else "  -> NOT corrected"))
    return aligned





def draw_rois(image, roi_file, update_date_rois=False, names=None, scale=2, n_points=4, rois=None, on_done=None):
    """Click n_points corners per arm (left click = add point, right click = undo last point);
    saves to roi_file after the last arm, then calls on_done() if given. The image is shown at
    1/scale size (faster redraws); saved polygons are in full-size pixels.
    Returns the rois dict (filled while you click) and the connection id"""
    if names is None:
        names = [f"maze{m}_{arm}" for arm in ("up", "down") for m in range(1, 5)]
    get_ipython().run_line_magic("matplotlib", "widget")       # = %matplotlib widget, from inside a function

    rois = {} if rois is None else rois      # a given dict is refilled in place (Redraw)
    rois.clear()
    pts, marks = [], []                      # clicks of the current arm and their dots
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(image[::scale, ::scale], cmap="gray")             # downsampled for display only
    ax.set_title(f"click {n_points} corners: {names[0]}")

    def on_click(event):
        if event.inaxes != ax or len(rois) == len(names):
            return
        if event.button == 3 and pts:                           # right click: undo last point
            pts.pop()
            marks.pop().remove()
        elif event.button == 1:
            pts.append((event.xdata, event.ydata))
            marks.append(ax.plot(event.xdata, event.ydata, "o", color="orange", ms=4)[0])
        if len(pts) == n_points:                                # arm complete
            name = names[len(rois)]
            rois[name] = [[int(x * scale), int(y * scale)] for x, y in pts]   # back to full-size pixels
            ax.add_patch(plt.Polygon(pts, closed=True, fill=False, color="orange", lw=2))
            ax.text(pts[0][0], pts[0][1] - 10 / scale, name, color="orange")
            pts.clear()
            marks.clear()
            if len(rois) < len(names):
                ax.set_title(f"click {n_points} corners: {names[len(rois)]}")
            else:
                json.dump(rois, open(roi_file, "w"), indent=1)
                if update_date_rois:
                    # also save as the ROIs of the whole date: <date_dir>/<date>_rois.json
                    date_dir = roi_file.rsplit("/", 2)[0]
                    date_roi_file = f"{date_dir}/{os.path.basename(date_dir)}_rois.json"
                    json.dump(rois, open(date_roi_file, "w"), indent=1)
                    print(f"also saved to {date_roi_file}")

                ax.set_title(f"done, saved to {roi_file}")
                if on_done is not None:
                    on_done()
        fig.canvas.draw_idle()                                  # one redraw per click only

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return rois, cid


def latest_date_roi_file(home_dir, up_to_date):
    """Most recent <date>/<date>_rois.json with a date <= up_to_date (folder names MM_DD_YYYY)"""
    limit = datetime.strptime(up_to_date, "%m_%d_%Y")
    best_date, best_file = None, None
    for f in glob(f"{home_dir}/*/*_rois.json"):            # date-level files only, not *_rois_3.json
        try:
            d = datetime.strptime(os.path.basename(os.path.dirname(f)), "%m_%d_%Y")
        except ValueError:                                  # folder that isn't a date
            continue
        if d <= limit and (best_date is None or d > best_date):
            best_date, best_file = d, f
    return best_file


def wait_until_done(state):
    """Block the running cell until state["done"] is True, while still handling
    button clicks and mouse clicks on the figures (interrupt the kernel to stop waiting)"""
    with ui_events() as poll:
        while not state["done"]:
            poll(10)                                              # process pending widget events
            time.sleep(0.1)


def load_or_draw_rois(image, local_roi_file, update_date_rois = False, scale=2, wait=True, do_not_show = False):
    """Show saved ROIs on the image with Accept / Redraw buttons, looking in order for:
    local_roi_file, the date's <date>_rois.json, the most recent earlier date's <date>_rois.json.
    If none exists, start drawing them. Returns the rois dict (refilled in place on Redraw).
    wait=True: does not return until you click Accept or finish drawing all arms
    do_not_show=True: if local_roi_file exists, return its ROIs without showing them"""
    date_roi_file_part = (local_roi_file.rsplit("/", 1)[0]).rsplit("/", 1)[0]
    date = date_roi_file_part.rsplit("/",1)[-1]
    date_roi_file = f"{date_roi_file_part}/{date}_rois.json"
    home_dir = os.path.dirname(date_roi_file_part)
    state = {"done": False}

    def finish():
        state["done"] = True

    if os.path.exists(local_roi_file):
        load_file = local_roi_file
        if do_not_show:
            return json.load(open(load_file)), None       # skip the check, use the movie's own ROIs
    
    elif os.path.exists(date_roi_file):
        load_file = date_roi_file
    elif (prev := latest_date_roi_file(home_dir, date)) is not None:
        load_file = prev                                          # ROIs of an earlier date
    else:
        rois, cid = draw_rois(image, local_roi_file, update_date_rois=update_date_rois, scale=scale, on_done=finish)
        if wait:
            wait_until_done(state)
        return rois, cid

    get_ipython().run_line_magic("matplotlib", "widget")
    rois = json.load(open(load_file))
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(image[::scale, ::scale], cmap="gray")
    for name, corners in rois.items():
        pts = np.array(corners) / scale                           # full-size -> display pixels
        ax.add_patch(plt.Polygon(pts, closed=True, fill=False, color="orange", lw=2))
        ax.text(pts[0][0], pts[0][1] - 10 / scale, name, color="orange")
    ax.set_title(f"loaded: {os.path.basename(load_file)}")
    plt.show()

    accept = widgets.Button(description="Accept", button_style="success")
    redraw = widgets.Button(description="Redraw", button_style="warning")
    out = widgets.Output()                                        # the drawing figure appears here

    def on_accept(_):
        accept.disabled = redraw.disabled = True
        if not os.path.exists(local_roi_file):
            json.dump(rois, open(local_roi_file, "w"), indent=1)
        if not os.path.exists(date_roi_file):                     # ROIs from an earlier date become this date's
            json.dump(rois, open(date_roi_file, "w"), indent=1)
        ax.set_title("accepted")
        fig.canvas.draw_idle()
        finish()

    def on_redraw(_):
        accept.disabled = redraw.disabled = True
        plt.close(fig)
        with out:
            # overwrites local_roi_file (and the date file if update_date_rois) when all arms are done
            draw_rois(image, local_roi_file, update_date_rois=update_date_rois, scale=scale, rois=rois,
                      on_done=finish)

    accept.on_click(on_accept)
    redraw.on_click(on_redraw)
    display(widgets.HBox([accept, redraw]), out)
    if wait:
        wait_until_done(state)
    return rois, None


def xtract_frames(movie_loc):
    cap = cv2.VideoCapture(movie_loc)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    print(len(frames), "frames")
    return frames

def background(frames):
    #adjust:
    start_index = 25
    chosen = [frames[i] for i in [24, 32, 42]]
    chosen2 = frames[start_index:start_index + 100:10]
    ####
    background1 = np.median(np.stack(chosen), axis=0).astype(np.uint8)
    background2 = np.median(np.stack(chosen2), axis=0).astype(np.uint8)
    background3 = np.max(np.stack(chosen), axis=0)
    background = np.median(np.stack([background1,background2, background3]), axis=0).astype(np.uint8)
    return background


def loc_flies(background_aligned, last, threshold=15):

    # local minimum: a pixel only counts as fly if it is darker than everything within +-3 px
    # in the background, so remaining small shifts of the walls/edges are ignored
    
    # adjust
    erode_size = 7
    min_area = 60  # smaller blobs are noise, not flies

    bg_min = cv2.erode(background_aligned, np.ones((erode_size, erode_size), np.uint8))
    diff = np.clip(bg_min.astype(np.int16) - last.astype(np.int16), 0, 255).astype(np.uint8)
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
    # fixed threshold (same for all movies); Otsu only printed for reference
    t, _ = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Otsu threshold:", t)
    thr = threshold #15    # your threshold
    mask = cv2.threshold(diff_blur, thr, 255, cv2.THRESH_BINARY)[1]
    #t, mask = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # automatic alternative

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)                 # reduces pixel noise
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]          # [0] is the background
                                 
    print(n - 1, "blobs,", (areas >= min_area).sum(), f"with area >= {min_area} px")
    print(sorted(areas[areas >= min_area]))

    # mask with only the selected blobs
    keep = np.flatnonzero(areas >= min_area) + 1         # +1 because label 0 is the background
    mask_flies = np.isin(labels, keep).astype(np.uint8) * 255

    # overlay: selected blobs in red on the last frame, with a green box and the area in px
    overlay = cv2.cvtColor(last, cv2.COLOR_GRAY2BGR)
    overlay[mask_flies > 0] = (0, 0, 255)
    for i in keep:
        x, y, w, h, a = stats[i]
        cv2.rectangle(overlay, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 255, 0), 2)
        cv2.putText(overlay, str(a), (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return diff, mask_flies, overlay


def count_flies(mask_flies, overlay, rois, loc_training_oddor):
    pi_maze = []
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_flies)
    single_fly_area = np.median(stats[1:, cv2.CC_STAT_AREA])     # most blobs are single flies
    print(f"single fly area: {single_fly_area:.0f} px")

    counts = {}
    for name, corners in rois.items():
        poly = np.array(corners, np.int32)
        counts[name] = 0
        for i in range(1, n):
            cx, cy = centroids[i]
            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:   # blob belongs to the arm that contains its centre
                counts[name] += max(1, round(stats[i, cv2.CC_STAT_AREA] / single_fly_area))
        cv2.polylines(overlay, [poly], True, (255, 128, 0), 3)
        x, y = corners[0]
        cv2.putText(overlay, f"{name}: {counts[name]}", (x + 5, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 128, 0), 3)
    
    for m in range(1, 5):
        up, down = counts[f"maze{m}_up"], counts[f"maze{m}_down"]
        if loc_training_oddor == 'up':
            pi = (up - down) / (up + down) if up + down else float("nan")
        else:
            pi = (down - up) / (up + down) if up + down else float("nan")
        print(f"maze {m}: up {up}, down {down}, PI = {pi:+.2f}")
        pi_maze.append(pi)
    return overlay, counts, pi_maze

# cv2.imwrite(f"{out_dir}/{date}/overlay2_{movie_number}.png", overlay)
# cv2.imwrite(f"{out_dir}/{date}_overlay_{movie_number}.png", overlay)

