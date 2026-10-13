# -*- coding: utf-8 -*-
"""
app.py — Aplicación web (Flask) para convertir imágenes en DXF y G-code CNC.

Uso:
    python app.py            # abre http://localhost:5000
"""

import io
import json
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify

import converter

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB


def _f(name, default, cast=float):
    v = request.form.get(name)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except (ValueError, TypeError):
        return default


def _b(name, default=False):
    v = request.form.get(name)
    if v is None:
        return default
    return v in ("1", "true", "on", "yes")


def build_from_request():
    """Procesa el formulario y devuelve (mask, contours_px, contours_mm, params)."""
    f = request.files.get("image")
    if not f or f.filename == "":
        raise ValueError("No se ha subido ninguna imagen.")
    bgr = converter.load_image(f.read())
    h, w = bgr.shape[:2]

    params = {
        "threshold_mode": request.form.get("threshold_mode", "otsu"),
        "threshold_value": _f("threshold_value", 127),
        "invert": _b("invert"),
        "blur": _f("blur", 3),
        "close_mm": _f("close_mm", 0.0),
        "min_area_mm2": _f("min_area_mm2", 2.0),
        "simplify": _f("simplify", 1.0),
        "target_width_mm": _f("target_width_mm", 0.0) or None,
        "target_height_mm": _f("target_height_mm", 0.0) or None,
    }

    mask = converter.preprocess(
        bgr, threshold_mode=params["threshold_mode"],
        threshold_value=params["threshold_value"], invert=params["invert"],
        blur=int(params["blur"]), close_mm=params["close_mm"])

    scale = converter.compute_scale(w, h, params["target_width_mm"],
                                    params["target_height_mm"])
    min_area_px = max(1.0, params["min_area_mm2"] / (scale * scale))
    eps_px = params["simplify"] / scale  # tolerancia dada en mm -> px

    contours_px = converter.extract_contours(mask, simplify_eps_px=eps_px,
                                             min_area_px=min_area_px)
    contours_mm = converter.transform_contours_to_mm(contours_px, scale)
    return mask, contours_px, contours_mm, params, (w, h), scale


def contours_to_svg_overlay(contours_mm, size_mm, img_size):
    """Dibuja los contornos como SVG para previsualización."""
    parts = []
    for c in contours_mm:
        for loop in [c["points"]] + c["holes"]:
            if len(loop) < 2:
                continue
            d = "M " + " L ".join("%.2f %.2f" % (x, -y) for x, y in loop) + " Z"
            parts.append('<path d="%s" fill="none" stroke="#ff3b30" stroke-width="1"/>' % d)
    return "".join(parts)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/preview", methods=["POST"])
def preview():
    try:
        mask, cpx, cmm, params, (w, h), scale = build_from_request()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # máscara como PNG base64
    ok, buf = cv2.imencode(".png", mask)
    mask_b64 = base64.b64encode(buf.tobytes()).decode()

    total_len_mm = 0.0
    n_loops = 0
    for c in cmm:
        for loop in [c["points"]] + c["holes"]:
            n_loops += 1
            arr = np.array(list(loop) + [loop[0]])
            total_len_mm += float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

    info = {
        "contours": len(cmm),
        "loops": n_loops,
        "width_mm": round(w * scale, 2),
        "height_mm": round(h * scale, 2),
        "cut_length_mm": round(total_len_mm, 1),
        "svg": contours_to_svg_overlay(cmm, (w * scale, h * scale), (w, h)),
        "svg_vb": "0 0 %.2f %.2f" % (w * scale, h * scale),
    }
    return jsonify({"mask_png": mask_b64, "info": info})


@app.route("/convert", methods=["POST"])
def convert():
    try:
        mask, cpx, cmm, params, (w, h), scale = build_from_request()
    except Exception as e:
        return Response(str(e), status=400, mimetype="text/plain")

    mode = request.form.get("mode", "profile")
    layer = "GRABADO" if mode == "engrave" else "CORTE"
    dxf_bytes = converter.export_dxf(cmm, layer=layer)

    gc_kw = dict(feed_cut=_f("feed_cut", 1000), feed_rapid=_f("feed_rapid", 3000),
                 feed_engrave=_f("feed_engrave", 1500), safe_z=_f("safe_z", 5.0))
    if mode == "engrave":
        gcode = converter.generate_gcode_engrave(
            cmm, spacing=_f("engrave_spacing", 0.3),
            engrave_depth=-abs(_f("engrave_depth", 0.3)), **gc_kw)
    else:
        gcode = converter.generate_gcode_profile(
            cmm, material_thk=_f("material_thk", 3.0),
            z_step=_f("z_step", 1.0), tool_comp=_f("tool_comp", 0.0), **gc_kw)

    fmt = request.args.get("fmt", "both")
    if fmt == "dxf":
        return Response(dxf_bytes, mimetype="application/dxf",
                        headers={"Content-Disposition":
                                 "attachment; filename=corte.dxf"})
    if fmt == "gcode":
        return Response(gcode, mimetype="text/plain",
                        headers={"Content-Disposition":
                                 "attachment; filename=corte.nc"})

    # ZIP con ambos
    import zipfile
    zf_buf = io.BytesIO()
    with zipfile.ZipFile(zf_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("corte.dxf", dxf_bytes)
        zf.writestr("corte.nc", gcode)
        zf.writestr("mascara.png", mask.tobytes() and
                    cv2.imencode(".png", mask)[1].tobytes())
    zf_buf.seek(0)
    return Response(zf_buf.read(), mimetype="application/zip",
                    headers={"Content-Disposition":
                             "attachment; filename=imagen_a_cnc.zip"})


if __name__ == "__main__":
    print("* Image2CNC — abre http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
