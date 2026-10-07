from __future__ import annotations

import cv2
import numpy as np

class OpenCVContourSVGConverter:

    @staticmethod
    def convert(contours,
                canvas_size = None,  # (width, height) in px; if None we'll fit to the data
                stroke = "#00C853",
                stroke_width = 2,
                fill = "none",
                scale = 1,
                close_paths = True,  # True -> <polygon>, False -> <polyline>
                simplify_epsilon = 0.0,  # >0 to simplify with approxPolyDP (in px)
                translate = (0.0, 0.0),  # (tx, ty) in px
                precision = 2,  # decimal places for SVG coords
                segment_name ="untitled"
                ):

        """
        contours: an np.ndarray of shape (N,1,2) or iterable of such arrays, as returned by cv2.findContours
        """
        # Normalize to a list
        if isinstance(contours, np.ndarray):
            contours = [contours]
        contours = [c for c in contours if c is not None and len(c) > 0]
        if not contours:
            raise ValueError("No contours provided.")

        def prepare(c):
            pts = c.reshape(-1, 2).astype(float)
            if simplify_epsilon and simplify_epsilon > 0:
                c_approx = cv2.approxPolyDP(c, simplify_epsilon, True)
                pts = c_approx.reshape(-1, 2).astype(float)
            # scale and translate
            pts = pts * float(scale)
            pts[:, 0] += translate[0]
            pts[:, 1] += translate[1]
            return pts

        prepped = [prepare(c) for c in contours]

        all_pts = np.vstack(prepped)
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)
        # shift so everything is positive in viewBox
        shift = -min_xy
        prepped = [pts + shift for pts in prepped]
        viewbox_width, viewbox_height = (max_xy - min_xy)
        viewbox_width = max(1.0, float(viewbox_width))
        viewbox_height = max(1.0, float(viewbox_height))
        viewbox = (0.0, 0.0, viewbox_width, viewbox_height)

        # Set a manual size based on measured dimensions when supplied
        if canvas_size is not None:
            width, height = canvas_size
            width = float(width)
            height = float(height)
        else:
            # If not measured or manually set, use the viewbox height as a good reference
            width, height = viewbox_width, viewbox_height

        # Build SVG elements
        def fmt_pts(pts):
            q = np.round(pts.astype(float), precision)
            return " ".join(f"{x:.{precision}f},{y:.{precision}f}" for x, y in q)

        elements = []
        tag = "polygon" if close_paths else "polyline"
        for pts in prepped:
            if close_paths and (len(pts) >= 3):
                el = f'<{tag} id="{segment_name}" points="{fmt_pts(pts)}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            else:
                # polyline or too-few points: keep open shape
                el = f'<polyline id="{segment_name}" points="{fmt_pts(pts)}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            elements.append(el)

        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{width:.0f}px" height="{height:.0f}px"
         viewBox="{viewbox[0]:.2f} {viewbox[1]:.2f} {viewbox[2]:.2f} {viewbox[3]:.2f}">
      {"  ".join(elements)}
    </svg>
    '''

        return svg, width, height