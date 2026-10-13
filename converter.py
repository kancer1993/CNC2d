# -*- coding: utf-8 -*-
"""
converter.py — Núcleo de conversión Imagen -> DXF -> G-code para corte CNC.

Pipeline:
  1. Carga de imagen (PNG/JPG/BMP/WEBP/TIFF...) con Pillow/OpenCV.
  2. Preprocesado: escala de grises, filtro gaussiano, umbralizado
     (fijo u Otsu), inversión opcional, limpieza morfológica.
  3. Detección de contornos (OpenCV findContours) con jerarquía
     (contornos externos + agujeros).
  4. Simplificación Douglas-Peucker (epsilon ajustable).
  5. Escalado a milímetros según tamaño físico deseado.
  6. Exportación DXF (ezdxf): LWPOLYLINE cerradas en capa CORTE / GRABADO.
  7. Generación de G-code (GRBL compatible):
        - Modo "profile" : corte perimetral real, con pasadas múltiples (z-step).
        - Modo "engrave" : marcas por líneas horizontales (scanline) dentro de la forma.
"""

import io
import math
import numpy as np
import cv2
from PIL import Image
import ezdxf


# ----------------------------------------------------------------------------
# Carga y preprocesado
# ----------------------------------------------------------------------------

def load_image(file_bytes: bytes) -> np.ndarray:
    """Devuelve la imagen como array BGR de OpenCV."""
    pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess(bgr: np.ndarray, threshold_mode="otsu", threshold_value=127,
               invert=False, blur=3, min_area_px=0, close_mm=0.0, dpi=25.4):
    """Umbraliza y devuelve una máscara binaria (blanco = zona a cortar/grabar)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        k = int(blur) | 1  # kernel impar
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    flag_t = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if threshold_mode == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, flag_t + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, int(threshold_value), 255, flag_t)

    # Cierre morfológico opcional (une bordes rotos). close_mm -> px
    if close_mm and close_mm > 0 and dpi > 0:
        px = max(1, int(round(close_mm * dpi / 25.4)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ----------------------------------------------------------------------------
# Extracción de contornos
# ----------------------------------------------------------------------------

def extract_contours(mask: np.ndarray, simplify_eps_px=1.0, min_area_px=20):
    """
    Devuelve lista de contornos. Cada contorno es:
      {"points": Nx2 float (px), "holes": [Nx2, ...], "area": float}
    Usa RETR_CCOMP; si el objeto detectado rellena casi toda la imagen
    (p.ej. fondo blanco con figuras recortadas) se descartan sus "agujeros"
    que en realidad son los objetos reales, y esos objetos se emiten como
    contornos independientes.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    img_area = float(mask.shape[0] * mask.shape[1])

    externals, holes = [], {}
    for i, (cnt, h) in enumerate(zip(contours, hierarchy)):
        area = abs(cv2.contourArea(cnt))
        if area < min_area_px:
            continue
        parent = h[3]
        if parent == -1:
            externals.append(i)
        else:
            holes.setdefault(parent, []).append(i)

    def build(i, hole_ids):
        pts = _simplify(contours[i].reshape(-1, 2).astype(np.float64),
                        simplify_eps_px)
        if len(pts) < 3:
            return None
        hole_list = []
        for hi in hole_ids:
            hp = _simplify(contours[hi].reshape(-1, 2).astype(np.float64),
                           simplify_eps_px)
            if len(hp) >= 3:
                hole_list.append(hp)
        return {"points": pts, "holes": hole_list,
                "area": abs(cv2.contourArea(contours[i]))}

    result = []
    for i in externals:
        hs = holes.get(i, [])
        # ¿el externo rellena casi toda la imagen? -> es un "marco" de fondo.
        # Sus hijos son las piezas reales (que a su vez pueden tener agujeros).
        if abs(cv2.contourArea(contours[i])) > 0.9 * img_area:
            for hi in hs:
                r = build(hi, holes.get(hi, []))
                if r:
                    result.append(r)
            continue
        r = build(i, hs)
        if r:
            result.append(r)

    # Si el marco de fondo se recorta por min_area_px sus hijos nunca se visitan:
    # fallback con RETR_EXTERNAL + jerarquía anidada por punto-en-polígono.
    if not result and cv2.countNonZero(mask) > 0.5 * img_area:
        ext, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = [(c, abs(cv2.contourArea(c))) for c in ext]
        polys = [(c, a) for c, a in polys if a >= min_area_px]
        for idx, (c, a) in enumerate(polys):
            cx, cy = np.mean(c.reshape(-1, 2), axis=0)
            child_of = None
            for jdx, (c2, a2) in enumerate(polys):
                if jdx == idx or a2 <= a:
                    continue
                if point_in_polygons(cx, cy, [c2.reshape(-1, 2).astype(np.float64)]):
                    if child_of is None or a2 < polys[child_of][1]:
                        child_of = jdx
            if child_of is not None:
                continue  # es un agujero de otro polígono
            hole_ids = [k for k, (ck, ak) in enumerate(polys)
                        if k != idx and ak < a
                        and point_in_polygons(*np.mean(ck.reshape(-1, 2), axis=0),
                                              [c.reshape(-1, 2).astype(np.float64)])]
            pts = _simplify(c.reshape(-1, 2).astype(np.float64), simplify_eps_px)
            if len(pts) < 3:
                continue
            hlist = [_simplify(polys[k][0].reshape(-1, 2).astype(np.float64),
                               simplify_eps_px) for k in hole_ids]
            hlist = [h for h in hlist if len(h) >= 3]
            result.append({"points": pts, "holes": hlist, "area": a})
    return result


def _simplify(points: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0 or len(points) < 4:
        return points
    cnt = points.reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(cnt, eps, closed=True)
    return simplified.reshape(-1, 2).astype(np.float64)


# ----------------------------------------------------------------------------
# Geometría auxiliar (malla -> polígono para grabado)
# ----------------------------------------------------------------------------

def mask_polygon_outlines(mask: np.ndarray, simplify_eps_px=1.0, min_area_px=20):
    """Contornos sin jerarquía, para recorte de líneas de grabado."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if abs(cv2.contourArea(c)) < min_area_px:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        pts = _simplify(pts, simplify_eps_px)
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def point_in_polygons(x, y, polygons):
    inside = False
    for poly in polygons:
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
    return inside


# ----------------------------------------------------------------------------
# Escalado a milímetros
# ----------------------------------------------------------------------------

def compute_scale(img_w, img_h, target_width_mm=None, target_height_mm=None):
    """Escala px->mm preservando proporción. Default: 0.254 mm/px (100 dpi)."""
    if target_width_mm and target_height_mm:
        scale = min(target_width_mm / img_w, target_height_mm / img_h)
    elif target_width_mm:
        scale = target_width_mm / img_w
    elif target_height_mm:
        scale = target_height_mm / img_h
    else:
        scale = 0.254
    return scale


def transform_contours_to_mm(contours_px, scale, flip_y=True):
    """Aplica escala px->mm e invierte Y (imagen: Y abajo; CNC/DXF: Y arriba).
    Devuelve nuevos dicts {points, holes} en mm."""
    out = []
    for c in contours_px:
        pts = [(float(x) * scale, -float(y) * scale if flip_y else float(y) * scale)
               for x, y in c["points"]]
        holes = [[(float(x) * scale, -float(y) * scale if flip_y else float(y) * scale)
                  for x, y in h] for h in c.get("holes", [])]
        out.append({"points": pts, "holes": holes})
    return out


# ----------------------------------------------------------------------------
# Exportación DXF
# ----------------------------------------------------------------------------

def export_dxf(polys_mm, layer="CORTE", dxf_version="R2010"):
    """polys_mm: lista de dicts {points, holes} ya en mm."""
    doc = ezdxf.new(dxf_version, setup=True)
    msp = doc.modelspace()
    for lyr in ("CORTE", "GRABADO"):
        if lyr not in doc.layers:
            doc.layers.add(lyr, color=1 if lyr == "CORTE" else 4)

    def add_loop(pts, lyr):
        if len(pts) < 3:
            return
        msp.add_lwpolyline([(float(x), float(y)) for x, y in pts],
                           close=True, dxfattribs={"layer": lyr})

    for p in polys_mm:
        add_loop(p["points"], layer)
        for h in p.get("holes", []):
            add_loop(h, layer)

    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


# ----------------------------------------------------------------------------
# Generación de G-code
# ----------------------------------------------------------------------------

class GCodeBuilder:
    def __init__(self, units_mm=True, feed_cut=1000, feed_rapid=3000,
                 feed_engrave=1500, spindle_on="M3 S10000", spindle_off="M5",
                 safe_z=5.0):
        self.lines = []
        self.units_mm = units_mm
        self.feed_cut = feed_cut
        self.feed_rapid = feed_rapid
        self.feed_engrave = feed_engrave
        self.spindle_on = spindle_on
        self.spindle_off = spindle_off
        self.safe_z = safe_z

    def add(self, s):
        self.lines.append(s)

    def header(self, comment=""):
        self.add(f"; Generado por Image2CNC — {comment}")
        self.add("; Unidades: MM (G21), plano XY (G17), posición absoluta (G90)")
        self.add("G21")
        self.add("G17")
        self.add("G90")
        self.add("G92 X0 Y0 Z0" )  # cero de trabajo relativo (GRBL usa G54 normalmente)
        self.add("G0 Z%.3f" % self.safe_z)
        self.add(self.spindle_on)
        self.add("G4 P1.0")  # esperar a que el husillo arranque

    def rapid(self, x, y):
        self.add("G0 X%.3f Y%.3f" % (x, y))

    def move_z(self, z, feed=None):
        f = feed or self.feed_rapid
        self.add("G1 Z%.3f F%d" % (z, f))

    def cut_polyline(self, pts, closed, depth_total, z_step, feed):
        """Corte perimetral con pasadas sucesivas en Z."""
        if len(pts) < 2:
            return
        x0, y0 = pts[0]
        self.rapid(x0, y0)
        n_pass = max(1, int(math.ceil(depth_total / z_step)))
        for p in range(1, n_pass + 1):
            z = -min(p * z_step, depth_total)
            self.move_z(z, feed=int(feed / 2))
            seq = list(pts) + ([pts[0]] if closed else [])
            for (x, y) in seq[1:]:
                self.add("G1 X%.3f Y%.3f F%d" % (x, y, feed))
            # levanta entre pasadas para limpiar viruta y regresa al inicio
            if p < n_pass:
                self.add("G0 Z0.5")
                self.rapid(x0, y0)
        self.add("G0 Z%.3f" % self.safe_z)

    def engrave_line(self, x0, y0, x1, y1, z, feed):
        self.rapid(x0, y0)
        self.move_z(z, feed=int(feed / 2))
        self.add("G1 X%.3f Y%.3f F%d" % (x1, y1, feed))
        self.add("G0 Z%.3f" % max(0.5, z + 1.0))

    def footer(self):
        self.add(self.spindle_off)
        self.add("G0 X0 Y0")
        self.add("M2")

    def text(self):
        return "\n".join(self.lines) + "\n"


def generate_gcode_profile(polys_mm, material_thk=3.0, z_step=1.0,
                           tool_comp=0.0, **kw):
    """Corte perimetral (profile). tool_comp: compensación de radio de herramienta:
    contornos externos se desplazan hacia fuera y los agujeros hacia dentro."""
    gb = GCodeBuilder(**kw)
    gb.header("modo=profile")
    for p in polys_mm:
        if tool_comp:
            outer = offset_polygon(p["points"], tool_comp, is_hole=False)
            holes = [offset_polygon(h, tool_comp, is_hole=True)
                     for h in p.get("holes", [])]
        else:
            outer, holes = p["points"], p.get("holes", [])
        for loop in [outer] + holes:
            gb.cut_polyline(loop, closed=True, depth_total=material_thk,
                            z_step=z_step, feed=gb.feed_cut)
    gb.footer()
    return gb.text()


def generate_gcode_engrave(polys_mm, spacing=0.3, engrave_depth=-0.3, **kw):
    """Grabado por líneas horizontales dentro de los contornos (mm)."""
    gb = GCodeBuilder(**kw)
    gb.header("modo=engrave")
    all_polys = []
    for p in polys_mm:
        all_polys.append(p["points"])
        all_polys.extend(p.get("holes", []))
    if not all_polys:
        gb.footer(); return gb.text()

    xs = [pt[0] for poly in all_polys for pt in poly]
    ys = [pt[1] for poly in all_polys for pt in poly]
    ymin, ymax = min(ys), max(ys)
    xmin, xmax = min(xs), max(xs)

    y = ymin
    direction = 1
    while y <= ymax:
        spans = _scanline_spans(y, all_polys)
        spans.sort(key=lambda s: s[0] if direction > 0 else s[1])
        for (x0, x1) in spans:
            xa, xb = (x0, x1) if direction > 0 else (x1, x0)
            xa = max(xa, xmin); xb = max(min(xb, xmax), xmin)
            if abs(xb - xa) < 0.05:
                continue
            gb.engrave_line(xa, y, xb, y, engrave_depth, gb.feed_engrave)
        y += spacing
        direction *= -1 if spans else 1
    gb.add("G0 Z%.3f" % gb.safe_z)
    gb.footer()
    return gb.text()


def _scanline_spans(y, polygons):
    """Intersecciones de la línea horizontal y con los polígonos (xor)."""
    xs = []
    for poly in polygons:
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 <= y < y2) or (y2 <= y < y1):
                t = (y - y1) / (y2 - y1 + 1e-12)
                xs.append(x1 + t * (x2 - x1))
    xs.sort()
    spans = []
    for i in range(0, len(xs) - 1, 2):
        spans.append((xs[i], xs[i + 1]))
    return spans


def offset_polygon(poly, dist, is_hole=False):
    """Desplazamiento simple de un polígono cerrado +/- dist usando normales
    de vértice (bisectriz aproximada). Suficiente para compensación de fresa."""
    if dist == 0:
        return poly
    sign = -dist if is_hole else dist
    n = len(poly)
    out = []
    for i in range(n):
        p_prev = poly[i - 1]
        p = poly[i]
        p_next = poly[(i + 1) % n]
        v1 = np.subtract(p, p_prev); v1 /= (np.linalg.norm(v1) + 1e-12)
        v2 = np.subtract(p_next, p); v2 /= (np.linalg.norm(v2) + 1e-12)
        bis = v1 + v2
        L = np.linalg.norm(bis)
        if L < 1e-9:
            nx, ny = -v2[1], v2[0]
        else:
            bis /= L
            nx, ny = -bis[1], bis[0]
        # factor para mantener distancia perpendicular correcta
        cos_half = max(0.3, float(np.dot([nx, ny], [-v1[1], v1[0]])))
        out.append((p[0] + sign * nx / cos_half, p[1] + sign * ny / cos_half))
    return out
