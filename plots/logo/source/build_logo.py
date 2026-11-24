"""Build every nbody_streams logo asset from source/logo_source.json.

    python plots/logo/source/build_logo.py

Writes, next to this folder (plots/logo/):
    nbody_streams_logo_{light,dark}.svg   wordmark, transparent
    nbody_streams_logo_{light,dark}.png   1600 px wide, transparent
    nbody_streams_mark_{light,dark}.svg   512 x 512 galaxy mark
    nbody_streams_social.png              1280 x 640 social preview (white)

Needs numpy and Pillow; the PNGs are rasterised with the system librsvg + cairo
(librsvg-2.so.2, libcairo.so.2) through ctypes. See nbody_streams_logo_spec.md.
"""
import ctypes
import json
import os
import re

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
SRC = json.load(open(os.path.join(HERE, 'logo_source.json')))

# ------------------------------------------------------------------ palette (theme-proof)
# Every colour stays readable on white *and* on GitHub dark (#0d1117), so the logo survives
# a light/dark mismatch. WCAG contrast (white / #0d1117): text light 6.2/3.0, text dark 2.8/6.8,
# muted light 3.1/6.0, muted dark 3.9/4.9.
PAL = {
    'light': dict(text='#6e5d59', muted='#9b8f88', mid='#7d6f6a', tip='#9b8f88',
                  iso_out='#b8a79e', iso_in='#6e5d59', prog='#f3e9e4',
                  halo='#5f4f4b', core='#5f4f4b', knot='#7d6f6a', knot_on='#e9ddd6'),
    'dark': dict(text='#a8978f', muted='#8b7f79', mid='#9d8c84', tip='#8b7f79',
                 iso_out='#6a5850', iso_in='#a8978f', prog='#e9ddd6',
                 halo='#a8978f', core='#cdbfb8', knot='#b5a49c', knot_on='#efe6e1'),
}
TAGLINE = '#6e5d59'

# ------------------------------------------------------------------ galaxy parameters
GAL = dict(bar_pa=170, turn=160, rb=0.07, incl=0.64, w0=0.185, wpow=0.7, bar_w=0.7,
           bulge=(0.115, 0.095), disk=(0.50, 0.42), disk_pa=-25, halo_k=2.1,
           disk_op={'light': 0.12, 'dark': 0.20}, halo_op={'light': 0.75, 'dark': 0.85},
           seed=11)


# ------------------------------------------------------------------ geometry helpers
def resample(p, step, closed=True):
    p = np.asarray(p, float)
    if closed:
        p = np.vstack([p, p[:1]])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, d[-1], max(int(d[-1] / step), 8), endpoint=not closed)
    return np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])]


def rdp(p, eps):
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    ab = b - a
    L = np.hypot(*ab)
    d = (np.hypot(*(p - a).T) if L == 0 else
         np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / L)
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.vstack([rdp(p[:i + 1], eps)[:-1], rdp(p[i:], eps)])
    return np.vstack([a, b])


def rdp_closed(p, eps):
    i, j = sorted((int(np.argmax(p[:, 0])), int(np.argmin(p[:, 0]))))
    a = rdp(p[i:j + 1], eps)
    b = rdp(np.vstack([p[j:], p[:i + 1]]), eps)
    return np.vstack([a[:-1], b[:-1]])


def fmt(v):
    return ('%.1f' % v).rstrip('0').rstrip('.')


def smooth_d(p, corner_deg=40):
    """closed polygon -> cubic Bezier path through its points; corners stay sharp"""
    p = np.asarray(p, float)
    n = len(p)
    prv, nxt = np.roll(p, 1, 0), np.roll(p, -1, 0)
    v1, v2 = p - prv, nxt - p
    ang = np.degrees(np.abs(np.arctan2(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0], (v1 * v2).sum(1))))
    T = nxt - prv
    T /= np.maximum(np.hypot(*T.T), 1e-9)[:, None]
    corner = ang > corner_deg
    out = ['M%s %s' % (fmt(p[0, 0]), fmt(p[0, 1]))]
    for i in range(n):
        j = (i + 1) % n
        L = np.hypot(*(p[j] - p[i])) / 3
        c1 = p[i] if corner[i] else p[i] + T[i] * L
        c2 = p[j] if corner[j] else p[j] - T[j] * L
        out.append('C%s %s %s %s %s %s' % tuple(fmt(v) for v in (*c1, *c2, *p[j])))
    return ''.join(out) + 'Z'


def shift_path(d, dx):
    """shift an absolute M/L/Q/C path horizontally"""
    out, cmd, k = [], None, 0
    for t in re.findall(r'[A-Za-z]|-?[\d.]+', d):
        if t.isalpha():
            cmd, k = t, 0
            out.append(t)
            continue
        v = float(t) + (dx if cmd in 'MLQC' and k % 2 == 0 else 0)
        k += 1
        out.append(fmt(v))
    return re.sub(r'(?<=[A-Za-z]) ', '', ' '.join(out))


def path_points(d):
    return np.array([float(v) for v in re.findall(r'-?[\d.]+', d)]).reshape(-1, 2)


# ------------------------------------------------------------------ galaxy s
def galaxy(cx, cy, H, g=GAL):
    """barred two-arm logarithmic spiral drawn as one tapered ribbon (svg coords, y down)"""
    t = np.linspace(0, 1, 400)
    th0 = np.radians(g['bar_pa'])
    th = th0 - np.radians(g['turn']) * t
    r = g['rb'] * np.exp(np.log(0.5 / g['rb']) * t ** 0.85)
    arm = np.c_[r * np.cos(th), r * np.sin(th)]
    s_bar = np.linspace(-1, 1, 60)[:, None] * g['rb'] * np.array([np.cos(th0), np.sin(th0)])
    cl = resample(np.vstack([-arm[::-1], s_bar[1:-1], arm]), 0.004, closed=False)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(cl, axis=0).T))]
    s /= s[-1]
    u = np.abs(2 * s - 1)                                   # 0 at the centre, 1 at the tips
    ub = g['rb'] / (g['rb'] + np.hypot(*np.diff(arm, axis=0).T).sum())
    au = np.clip((u - ub) / (1 - ub), 0, 1)
    w_arm = (1 - au ** 1.6) ** g['wpow'] * (1 - (1 - g['bar_w']) * np.exp(-(au / 0.18) ** 2))
    w = np.maximum(g['w0'] * np.where(u < ub, g['bar_w'], w_arm), 0.004)
    tang = np.gradient(cl, axis=0)
    tang /= np.hypot(*tang.T)[:, None]
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    outline = np.vstack([cl + nrm * w[:, None] / 2, (cl - nrm * w[:, None] / 2)[::-1]])
    tf = lambda p: np.c_[cx + p[:, 0] * g['incl'] * H, cy - p[:, 1] * H]
    be = tf(np.array([[np.cos(th0), np.sin(th0)], [-np.cos(th0), -np.sin(th0)]]) * g['rb'])
    bar_ang = np.degrees(np.arctan2(*(be[0] - be[1])[::-1]))
    return dict(outline=tf(outline), cl=tf(cl), tips=tf(cl[[0, -1]]), bar_ang=bar_ang,
                ctr=np.array([cx, cy]), H=H)


def unit_height(g=GAL):
    return np.ptp(galaxy(0, 0, 1, g)['outline'][:, 1])


def clusters(gal, g=GAL, n_on=5, n_off=5, n_core=7):
    """artist's-impression detail: bright knots on the arms, tiny stars off the arms and
    around the core -> list of (xy, r, colour key, opacity)"""
    rng = np.random.default_rng(g['seed'])
    cl, H = gal['cl'], gal['H']
    n = len(cl)
    out = []

    def at(half, f):
        i = int((0.5 - f / 2) * (n - 1)) if half == 0 else int((0.5 + f / 2) * (n - 1))
        t = cl[min(i + 1, n - 1)] - cl[max(i - 1, 0)]
        t /= np.hypot(*t)
        return cl[i], np.array([-t[1], t[0]]) * (1 if half == 0 else -1)

    for half in (0, 1):
        for f in np.linspace(0.30, 0.78, n_on) + rng.uniform(-0.03, 0.03, n_on):
            q, nn = at(half, f)
            w = g['w0'] * H * (1 - f ** 1.6) ** g['wpow']
            out.append((q + nn * rng.uniform(-0.25, 0.25) * w,
                        rng.uniform(0.009, 0.016) * H, 'knot_on', rng.uniform(0.75, 1.0)))
        for f in rng.uniform(0.35, 1.0, n_off):
            q, nn = at(half, f)
            w = g['w0'] * H * (1 - min(f, 0.99) ** 1.6) ** g['wpow']
            side = 1 if rng.uniform() < 0.7 else -1
            out.append((q + nn * side * (w / 2 + rng.uniform(0.025, 0.075) * H),
                        rng.uniform(0.005, 0.010) * H, 'knot', rng.uniform(0.6, 1.0)))
    for a in np.linspace(0, 2 * np.pi, n_core, endpoint=False) + rng.uniform(-0.4, 0.4, n_core):
        rr = rng.uniform(0.15, 0.22) * H
        out.append((gal['ctr'] + np.array([np.cos(a) * rr * 0.8, np.sin(a) * rr]),
                    rng.uniform(0.005, 0.009) * H, 'knot', rng.uniform(0.6, 1.0)))
    return out


def galaxy_svg(gal, P, mode, defs, add, g=GAL, disk_scale=1.0):
    """disk glow, arms, core halo, core, clusters; returns the disk's right edge (svg x)"""
    cx, cy = gal['ctr']
    H = gal['H']
    tips = gal['tips']
    defs.append(f'<linearGradient id="gg" gradientUnits="userSpaceOnUse" x1="{tips[0,0]:.0f}" '
                f'y1="{tips[0,1]:.0f}" x2="{tips[1,0]:.0f}" y2="{tips[1,1]:.0f}">'
                f'<stop offset="0" stop-color="{P["tip"]}"/><stop offset="0.3" stop-color="{P["text"]}"/>'
                f'<stop offset="0.7" stop-color="{P["text"]}"/><stop offset="1" stop-color="{P["tip"]}"/>'
                '</linearGradient>')
    da, db = (v * H * disk_scale for v in g['disk'])
    op = g['disk_op'][mode]
    defs.append(f'<radialGradient id="gd"><stop offset="0" stop-color="{P["halo"]}" stop-opacity="{op}"/>'
                f'<stop offset="0.55" stop-color="{P["halo"]}" stop-opacity="{op * 0.4:.3f}"/>'
                f'<stop offset="1" stop-color="{P["halo"]}" stop-opacity="0"/></radialGradient>')
    add(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{da:.1f}" ry="{db:.1f}" '
        f'transform="rotate({g["disk_pa"]} {cx:.1f} {cy:.1f})" fill="url(#gd)"/>')
    add(f'<path d="{smooth_d(rdp_closed(gal["outline"], 0.3), corner_deg=60)}" fill="url(#gg)"/>')
    ba, bb = (v * H for v in g['bulge'])
    rot = f'rotate({gal["bar_ang"]:.1f} {cx:.1f} {cy:.1f})'
    hop = g['halo_op'][mode]
    defs.append(f'<radialGradient id="gh"><stop offset="0" stop-color="{P["text"]}" stop-opacity="{hop}"/>'
                f'<stop offset="0.5" stop-color="{P["text"]}" stop-opacity="{hop * 0.45:.3f}"/>'
                f'<stop offset="1" stop-color="{P["text"]}" stop-opacity="0"/></radialGradient>')
    add(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{ba * g["halo_k"]:.1f}" ry="{bb * g["halo_k"]:.1f}" '
        f'transform="{rot}" fill="url(#gh)"/>')
    add(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{ba:.1f}" ry="{bb:.1f}" transform="{rot}" fill="{P["core"]}"/>')
    for q, r, key, o in clusters(gal, g):
        add(f'<circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="{r:.1f}" fill="{P[key]}" fill-opacity="{o:.2f}"/>')
    a = np.radians(g['disk_pa'])
    return cx + 0.8 * np.hypot(da * np.cos(a), db * np.sin(a))


# ------------------------------------------------------------------ wordmark
def wordmark(mode):
    P = PAL[mode]
    gl, sh = SRC['glyphs'], SRC['shapes']
    str_main, str_tail = np.array(sh['str_main']), np.array(sh['str_tail'])
    px, py, pr = sh['prog']
    # the traced t/r are wider than v2.4.0's, so e/a/m move right to sit 14 units under the r flag
    dx = (str_main[:, 0].max() - 14) - path_points(gl['e'])[:, 0].min()
    e, a, m = (shift_path(gl[k], dx) for k in 'eam')
    H = 575 / unit_height()                                   # galaxy ink height = e/a height
    g0 = galaxy(0, 0, H)
    cx = path_points(m)[:, 0].max() + 50 - g0['outline'][:, 0].min()
    cy = 776.4 - g0['outline'][:, 1].max()                    # ink bottom = e/a overshoot
    gal = galaxy(cx, cy, H)

    defs, body = [], []
    add = body.append
    defs.append(f'<linearGradient id="gs" gradientUnits="userSpaceOnUse" x1="{str_tail[:, 0].min():.0f}" '
                f'y1="0" x2="3990" y2="0"><stop offset="0" stop-color="{P["muted"]}"/>'
                f'<stop offset="0.55" stop-color="{P["mid"]}"/><stop offset="1" stop-color="{P["text"]}"/>'
                '</linearGradient>')
    for k in 'nb':
        add(f'<path d="{gl[k]}" fill="{P["muted"]}"/>')
    add(f'<path d="{gl["o"]}" fill="{P["muted"]}" fill-rule="evenodd"/>')
    ro, ri, io, ii = (np.array(q) for q in sh['o_rings'])
    add(f'<path d="{smooth_d(ro)} {smooth_d(ri)}" fill="{P["iso_out"]}" fill-rule="evenodd"/>')
    add(f'<path d="{smooth_d(io)} {smooth_d(ii)}" fill="{P["iso_in"]}" fill-rule="evenodd"/>')
    for k in 'dy':
        add(f'<path d="{gl[k]}" fill="{P["muted"]}"/>')
    add(f'<path d="{smooth_d(str_tail)}" fill="url(#gs)"/>')
    add(f'<path d="{smooth_d(str_main)}" fill="url(#gs)"/>')
    add(f'<circle cx="{px}" cy="{py}" r="{pr}" fill="{P["text"]}"/>')
    add(f'<circle cx="{px}" cy="{py}" r="{pr * 0.76:.1f}" fill="{P["prog"]}"/>')
    for d in (e, a, m):
        add(f'<path d="{d}" fill="{P["text"]}"/>')
    x_glow = galaxy_svg(gal, P, mode, defs, add)

    ink = (270.1, 38.0, gal['outline'][:, 0].max(), max(1002.6, str_tail[:, 1].max()))
    x0, y0, x1, y1 = ink[0], ink[1], max(ink[2], x_glow), ink[3]
    px_, py_ = 0.04 * (x1 - x0), 0.04 * (y1 - y0)
    vb = (round(x0 - px_), round(y0 - py_), round(x1 - x0 + 2 * px_), round(y1 - y0 + 2 * py_))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb[0]} {vb[1]} {vb[2]} {vb[3]}" '
           f'width="{vb[2] / 10:.1f}" height="{vb[3] / 10:.1f}" role="img" aria-label="nbody_streams">'
           f'<title>nbody_streams</title><defs>{"".join(defs)}</defs>{"".join(body)}</svg>\n')
    return svg, vb, ink


# ------------------------------------------------------------------ mark (512 x 512)
def mark(mode, size=512, ink_frac=0.62):
    P = PAL[mode]
    H = ink_frac * size / unit_height()
    gal = galaxy(size / 2, size / 2, H)
    defs, body = [], []
    galaxy_svg(gal, P, mode, defs, body.append, disk_scale=0.92)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" '
            f'height="{size}" role="img" aria-label="nbody_streams"><title>nbody_streams</title>'
            f'<defs>{"".join(defs)}</defs>{"".join(body)}</svg>\n')


# ------------------------------------------------------------------ rasterising
def render_png(svg_text, vb, width, out_png, bg=None):
    """svg text -> png via librsvg + cairo (ctypes); transparent unless bg=(r,g,b) in 0..1"""
    rs, ca = ctypes.CDLL('librsvg-2.so.2'), ctypes.CDLL('libcairo.so.2')
    rs.rsvg_handle_new_from_data.restype = ctypes.c_void_p
    rs.rsvg_handle_new_from_data.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p]
    rs.rsvg_handle_render_cairo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ca.cairo_image_surface_create.restype = ctypes.c_void_p
    ca.cairo_create.restype = ctypes.c_void_p
    ca.cairo_create.argtypes = [ctypes.c_void_p]
    ca.cairo_scale.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    ca.cairo_translate.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    ca.cairo_set_source_rgb.argtypes = [ctypes.c_void_p] + [ctypes.c_double] * 3
    ca.cairo_paint.argtypes = [ctypes.c_void_p]
    ca.cairo_surface_write_to_png.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    for f in ('cairo_destroy', 'cairo_surface_destroy'):
        getattr(ca, f).argtypes = [ctypes.c_void_p]
    # give the root svg its viewBox size in user units so 1 unit = 1 px before scaling
    t = re.sub(r'<svg([^>]*?) width="[^"]*" height="[^"]*"',
               lambda m_: f'<svg{m_.group(1)} width="{vb[2]}" height="{vb[3]}"', svg_text, count=1)
    data = t.encode()
    h = rs.rsvg_handle_new_from_data(data, len(data), None)
    if not h:
        raise RuntimeError('librsvg could not parse the svg')
    s = width / vb[2]
    W, Ht = int(round(width)), int(round(vb[3] * s))
    surf = ca.cairo_image_surface_create(0, W, Ht)
    cr = ca.cairo_create(surf)
    if bg is not None:
        ca.cairo_set_source_rgb(cr, *bg)
        ca.cairo_paint(cr)
    ca.cairo_scale(cr, s, s)
    rs.rsvg_handle_render_cairo(h, cr)
    ca.cairo_surface_write_to_png(surf, out_png.encode())
    ca.cairo_destroy(cr)
    ca.cairo_surface_destroy(surf)


def social(svg_light, vb, ink, out_png, W=1280, Hh=640, ink_w=980, gap=46):
    """white 1280 x 640 card: wordmark ink 980 px wide, tagline 46 px below, block centred"""
    s = ink_w / (ink[2] - ink[0])
    tmp = out_png + '.tmp.png'
    render_png(svg_light, vb, vb[2] * s, tmp)
    logo = Image.open(tmp).convert('RGBA')
    os.remove(tmp)
    tag_a = Image.open(os.path.join(HERE, 'tagline_alpha.png')).convert('L')
    tag = Image.new('RGBA', tag_a.size, TAGLINE)
    tag.putalpha(tag_a)
    ink_h = (ink[3] - ink[1]) * s
    block = ink_h + gap + tag.height
    top = (Hh - block) / 2
    card = Image.new('RGBA', (W, Hh), 'white')
    lx = (W - ink_w) / 2 - (ink[0] - vb[0]) * s
    ly = top - (ink[1] - vb[1]) * s
    card.alpha_composite(logo, (int(round(lx)), int(round(ly))))
    card.alpha_composite(tag, (int(round((W - tag.width) / 2)), int(round(top + ink_h + gap))))
    card.convert('RGB').save(out_png, optimize=True)


if __name__ == '__main__':
    for mode in ('light', 'dark'):
        svg, vb, ink = wordmark(mode)
        open(os.path.join(OUT, f'nbody_streams_logo_{mode}.svg'), 'w').write(svg)
        render_png(svg, vb, 1600, os.path.join(OUT, f'nbody_streams_logo_{mode}.png'))
        open(os.path.join(OUT, f'nbody_streams_mark_{mode}.svg'), 'w').write(mark(mode))
        if mode == 'light':
            social(svg, vb, ink, os.path.join(OUT, 'nbody_streams_social.png'))
        print(mode, 'viewBox', vb)
