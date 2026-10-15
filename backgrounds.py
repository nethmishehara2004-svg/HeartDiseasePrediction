"""
Medical Blue backgrounds for the Heart Disease Prediction app.

Everything is built from CSS gradients plus three tiny inline SVGs
(waves, ECG trace, network). No image files are loaded, so the page
stays light. The result is one CSS rule for `.stApp`, generated per theme.
"""

import base64
import os
from functools import lru_cache
from urllib.parse import quote

PALETTES = {
    "Dark": {
        "ink": "#6ea4ff",                       # colour of the line art
        "wave_fill": 0.06, "wave_line": 0.10,   # opacities (kept very low)
        "ecg": 0.13, "net": 0.16,
        "veil": "rgba(8,16,34,0.18)",           # translucent overlay for contrast
        "glows": ("radial-gradient(circle at 12% 0%, rgba(79,143,247,0.16) 0%, transparent 42%)",
                  "radial-gradient(circle at 92% 100%, rgba(79,143,247,0.10) 0%, transparent 40%)"),
        "base": "linear-gradient(165deg, #0c1a36 0%, #0a1428 55%, #070e1e 100%)",
        "base_colour": "#0a1428",
        # navy tint over the photo: keeps the heart/ECG visible but calm
        "photo_veil": "linear-gradient(180deg, rgba(8,16,34,0.80) 0%, rgba(8,16,34,0.86) 100%)",
    },
    "Light": {
        "ink": "#1f5fae",
        "wave_fill": 0.10, "wave_line": 0.20,
        "ecg": 0.24, "net": 0.26,
        "veil": "rgba(255,255,255,0.06)",
        "glows": ("radial-gradient(circle at 10% 0%, rgba(255,255,255,0.55) 0%, transparent 45%)",
                  "radial-gradient(circle at 95% 100%, rgba(47,111,214,0.22) 0%, transparent 45%)"),
        "base": "linear-gradient(165deg, #dce8f7 0%, #c8dbf1 50%, #b3cae8 100%)",
        "base_colour": "#c8dbf1",
        # soft blue tint over the photo so it stays visible but calm
        "photo_veil": "linear-gradient(180deg, rgba(200,219,241,0.82) 0%, rgba(179,202,232,0.88) 100%)",
    },
}


def _uri(svg):
    """Turn an SVG string into a CSS url(...) data URI."""
    svg = " ".join(svg.split())
    return 'url("data:image/svg+xml,' + quote(svg, safe="/:=,.- ") + '")'


def _waves(p):
    ink, fill, line = p["ink"], p["wave_fill"], p["wave_line"]
    return _uri(f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 900" preserveAspectRatio="xMidYMax slice">
      <path d="M0 650 C240 590 430 710 720 660 C1010 610 1200 530 1440 590 L1440 900 L0 900 Z"
            fill="{ink}" fill-opacity="{fill}"/>
      <path d="M0 740 C260 690 470 800 760 750 C1050 700 1230 640 1440 690 L1440 900 L0 900 Z"
            fill="{ink}" fill-opacity="{fill * 0.8:.3f}"/>
      <path d="M0 650 C240 590 430 710 720 660 C1010 610 1200 530 1440 590"
            fill="none" stroke="{ink}" stroke-opacity="{line}" stroke-width="1.5"/>
      <path d="M0 600 C220 550 460 650 740 610 C1020 570 1220 490 1440 540"
            fill="none" stroke="{ink}" stroke-opacity="{line * 0.7:.3f}" stroke-width="1"/>
    </svg>""")


def _ecg(p):
    # one heartbeat per 640px tile; starts and ends on the baseline so it repeats seamlessly
    return _uri(f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 120" width="640" height="120">
      <path d="M0 60 H110 Q125 40 140 60 H190 L200 68 L212 14 L224 102 L234 60 H300 Q325 28 355 60 H640"
            fill="none" stroke="{p['ink']}" stroke-opacity="{p['ecg']}" stroke-width="1.6"
            stroke-linecap="round" stroke-linejoin="round"/>
    </svg>""")


def _network(p):
    nodes = [(40, 60), (140, 30), (230, 90), (330, 40), (430, 100), (480, 200),
             (380, 235), (290, 170), (180, 195), (90, 150), (250, 295), (150, 305)]
    links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2), (7, 8),
             (8, 9), (9, 0), (9, 1), (8, 11), (11, 10), (10, 6), (10, 7), (3, 7)]
    ink, op = p["ink"], p["net"]
    lines = "".join(
        f'<line x1="{nodes[a][0]}" y1="{nodes[a][1]}" x2="{nodes[b][0]}" y2="{nodes[b][1]}"/>'
        for a, b in links)
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="3.5"/>' for x, y in nodes)
    return _uri(f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 340" width="520" height="340">
      <g stroke="{ink}" stroke-opacity="{op}" stroke-width="1">{lines}</g>
      <g fill="{ink}" fill-opacity="{op * 1.4:.3f}">{dots}</g>
    </svg>""")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def load_photo():
    """Return (mime, base64) of background.jpg/.jpeg/.png next to app.py, or None."""
    for name, mime in (("background.jpg", "image/jpeg"),
                       ("background.jpeg", "image/jpeg"),
                       ("background.png", "image/png")):
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return mime, base64.b64encode(f.read()).decode()
    return None


def _photo_css(p, photo):
    """Photo background under a theme-coloured translucent veil."""
    mime, data = photo
    return f"""
    .stApp {{
        background-color: {p['base_colour']};
        background-image: {p['photo_veil']}, url(data:{mime};base64,{data});
        background-repeat: no-repeat, no-repeat;
        background-size: auto, cover;
        background-position: 0 0, center center;
        background-attachment: fixed;
    }}
    @media (max-width: 640px) {{
        .stApp {{ background-attachment: scroll; background-position: 0 0, 30% center; }}
    }}
    """


@lru_cache(maxsize=None)
def background_css(theme_name, use_photo=True):
    """CSS for the page background. Uses the photo if present, else the SVG art."""
    p = PALETTES[theme_name]
    photo = load_photo() if use_photo else None
    if photo:
        return _photo_css(p, photo)
    layers = [  # top layer first
        f"linear-gradient({p['veil']}, {p['veil']})",   # contrast overlay
        _network(p),                                    # medical network, top-right
        _ecg(p),                                        # ECG trace, repeats across
        _waves(p),                                      # abstract waves, bottom
        p["glows"][0],                                  # faint blue highlights
        p["glows"][1],
        p["base"],                                      # base gradient
    ]
    image = ", ".join(layers)

    def size(net, ecg_w, ecg_h):
        return f"auto, {net}px auto, {ecg_w}px {ecg_h}px, cover, auto, auto, auto"

    def pos(net_x, net_y, ecg_y):
        return f"0 0, right {net_x}px top {net_y}px, 0 {ecg_y}%, center bottom, 0 0, 0 0, 0 0"

    repeat = "no-repeat, no-repeat, repeat-x, no-repeat, no-repeat, no-repeat, no-repeat"

    return f"""
    .stApp {{
        background-color: {p['base_colour']};
        background-image: {image};
        background-repeat: {repeat};
        background-size: {size(560, 640, 120)};
        background-position: {pos(16, 72, 90)};
        background-attachment: fixed;
    }}
    @media (max-width: 1024px) {{
        .stApp {{
            background-size: {size(400, 520, 98)};
            background-position: {pos(8, 64, 92)};
        }}
    }}
    @media (max-width: 640px) {{
        .stApp {{
            background-size: {size(280, 400, 76)};
            background-position: {pos(0, 56, 94)};
            background-attachment: scroll;
        }}
    }}
    """
