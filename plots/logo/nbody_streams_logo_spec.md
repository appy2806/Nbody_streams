# nbody_streams logo spec

Final variant: Manrope, cursive t and r slanted 11°, uneven stream with a progenitor, spiral-galaxy final s.

## Files

| File | Use |
|---|---|
| `nbody_streams_logo_light.svg` | README / docs on light backgrounds. Transparent, 7284 x 1037 viewBox, 9.6 KB |
| `nbody_streams_logo_dark.svg` | Same geometry, recoloured for `#0d1117` |
| `nbody_streams_mark_light.svg`, `nbody_streams_mark_dark.svg` | Square mark (stream-s only), 512 x 512 viewBox, 1.3 KB |
| `nbody_streams_logo_light.png`, `nbody_streams_logo_dark.png` | 1600 x 228, transparent |
| `nbody_streams_social.png` | 1280 x 640 social preview, white background |

All SVGs contain only `path`, `circle` (mark only), `linearGradient` and `stop`. They have no `<text>`, fonts, rasters, filters, masks, clip paths, strokes or transforms. The logo viewBox is padded by 4% of the ink width on the left and right and 4% of the ink height on the top and bottom.

README usage:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/nbody_streams_logo_dark.svg">
  <img alt="nbody_streams" src="docs/_static/nbody_streams_logo_light.svg" width="480">
</picture>
```

## Typeface

- Manrope (SIL Open Font License 1.1), static fonts from Fontsource 5.3.0.
- "nbody": Manrope Light (300). The o glyph is replaced by the isophote "o" (see below).
- "treams": Manrope SemiBold (600). The t and r are drawn as strokes (geometry below); e, a and m are the stock glyphs.
- Social tagline: Manrope Regular (400), 32 px, shaped with HarfBuzz (kerning on).
- Set at 1000 units/em, so the x-height is X = 540 units. All geometry below is given in units of X. The origin is the baseline at the pen origin of "n". The y-axis points up.
- Tracking: −0.058 em between Light glyphs and −0.047 em between SemiBold glyphs.
- Pen x positions (X): n 0, b 0.973, o 1.947, d 2.970, y 3.944. Stream slot ("s") 5.625. t 6.530, r 7.225, e 8.054, a 9.072, m 10.039, galaxy slot 11.586. Total advance 12.755.
- Gap between "nbody" and the stream slot: 0.831 X (0.68 of the SemiBold underscore advance).

## Colours

| Role | Light | Dark |
|---|---|---|
| Background | `#ffffff` (README); social preview `#ffffff` | `#0d1117` |
| Text ("treams", t/r strokes) | `#2e2424` | `#f6ede8` |
| Muted text ("nbody") | `#9b8f88` | `#ab9f98` |
| o inner isophotes | `#d9c9bf` | `#5a4840` |
| Trailing arm (gradient start) | `#9b8f88` | `#ab9f98` |
| Stream mid-point (gradient middle) | `#645a56` | `#d0c6c0` |
| Leading arm (gradient end) | `#2e2424` | `#f6ede8` |
| Progenitor | `#2e2424` | `#f6ede8` |
| Galaxy arm tips | `#8b7f79` | `#b6aba4` |
| Galaxy bulge and inner arms | `#2e2424` | `#f6ede8` |
| Social tagline | `#645a56` | n/a |

The stream colours are matched to the words. Dark mode needed no extra adjustment for contrast. WCAG contrast against the background:

- Light (vs `#ffffff`): text 15.1, muted text 3.14, stream mid 6.7, galaxy tips 3.9, o isophotes 1.6.
- Dark (vs `#0d1117`): text 16.4, muted text 7.3, stream mid 11.3, galaxy tips 8.4, o isophotes 2.2.

Caveat: the o isophotes fall below 3:1 in both modes on purpose. They are a decorative inner contour, and the o outline carries the letter.

## Gradients

The stream uses a vertical `linearGradient` in user space. It runs from y = 0.052 X (offset 0, trailing colour) to y = 0.852 X (offset 1, leading colour), with the mid colour at offset 0.5. This is the progenitor height ± 0.40 X. In the exported light/dark SVG coordinates it is `x1=200 y1=753 x2=200 y2=321`.

The galaxy uses a `linearGradient` along the axis joining the two arm tips, from (11.635, 0.305) X to (12.617, 0.695) X. Stops:

- 0: tip colour
- 0.270: bulge colour
- 0.728: bulge colour
- 1: tip colour

The flat middle band covers the bulge, so the arms fade toward both tips. In SVG coordinates this is `x1=6483 y1=594 x2=7013 y2=383`.

## Stream (first s): arm geometry

Centreline, in X units (y up):

1. Trailing arm. A cubic Bézier from the tip at P0 = (4.592, −0.240), with controls P1 = (5.296, −0.170) and P2 = (5.809, 0.235), to P3 = (5.952, 0.550). The tip starts 0.14 X right of the y descender and leaves at a slope of 0.10. The arm arrives along the unit vector (1, 2.2)/|(1, 2.2)|.
2. Rise into the leading arm. A cubic Bézier from (5.952, 0.550), with controls (6.056, 0.780) and (6.321, 0.937), to (6.700, 0.937).
3. Leading arm. A straight line from (6.700, 0.937) to the end at (8.075, 0.937). The line forms the t crossbar and the r's head, and ends 0.18 X past the right edge of the stock Manrope SemiBold r.

The centreline is resampled to 1200 points by arc length.

Progenitor: a circle centred on the centreline at (5.897, 0.452) X, with radius 0.129 X.

Width profile (full width, along arc length d measured from the progenitor; Lt and Ll are the trailing and leading arc lengths):

- Base width w = 0.123 X. This is 0.8 of the t crossbar band in Manrope SemiBold (0.154 X).
- Trailing side (d < 0): W/w = 1 − 0.55 (|d|/Lt)^0.8.
- Leading side (d > 0): W/w = 1 − 0.25 (d/Ll)^0.7.
- Swell at the progenitor: multiply by 1 + 0.35 exp[−(d / 0.12(Lt + Ll))²].
- Trailing tip taper: over the last 10% of the trailing arm, multiply by clip((Lt + d)/(0.1 Lt), 0.15, 1).

Resulting widths: 0.166 X at the progenitor, 0.092 X at the leading end, and about 0.008 X at the trailing tip.

Flat top: after the progenitor, the centreline is raised by s·(w − W)/2, where s = smoothstep((y − 0.837 X)/0.10 X). This keeps the leading arm's top edge flat at y = 0.998 X, the top of the Manrope SemiBold t crossbar.

## t and r (slanted 11°)

Both letters are built upright and then sheared with x' = x + tan(11°)·y about the baseline.

- **t.** Butt-capped stroke of width 0.195 X (the Manrope SemiBold t stem). The stem runs at x = 6.838 X from y = 1.278 X down to 0.317 X. A foot arc of radius 0.22 X then sweeps from 180° to 295°, so the hook flicks up 25° past horizontal. The stream line is the crossbar.
- **r.** The stem's left edge is at x = 7.355 X, width 0.195 X, from the baseline up to 0.638 X. A quarter-circle shoulder has outer radius 0.36 X and centre (7.715, 0.638) X, so its top edge meets the stream's top edge at 0.998 X. Along the shoulder, the stroke width eases (smoothstep) from 0.195 X down to the stream width where they meet.

## Galaxy (final s)

- The centreline is a barred two-arm logarithmic spiral built by `galaxy_s`. Parameters: bar angle 156°, bar length 0.34, arm end angle 15°, inclination 0.72 (x-compression), arm taper 0.35. It is scaled to x-height + 12 units and shifted 0.10 X left.
- Arm width 0.138 X at the centre, tapering to the tips.
- Bulge: an ellipse with semi-axes 0.237 X × 0.199 X, centred at (12.125, 0.501) X, rotated −148.3° in SVG coordinates, filled with the bulge colour.

## o (nbody)

The o is a triaxial isophote outline filled in the muted colour. Inside it are two thin nested contours (0.035 X) in the isophote colour. The inner contour is a bar.

## Exported coordinates

In both logo SVGs, x_svg = x + 199.8 and y_svg = 758.4 − y, with x and y in font units (X = 540). Multiply the X values above by 540 first. The viewBox is 0 0 7284 1037.

## Mark

The mark reuses the wordmark's stream centreline, re-stroked heavier so it holds at 32 px.

- The centreline is cropped at x ≤ 7.05 X (0.35 X of flat leading arm kept) and centred on the progenitor.
- y is stretched by 1.6. The curve is scaled to a 512 square (1 X = 170.9 px horizontally) and smoothed with a Gaussian (σ = 60 of 1200 arc-length samples) to round the upper bend.
- Width: w0 = 0.08 S = 41 px, with the same profile as the wordmark but trailing minimum 0.55, leading minimum 0.85 and swell 0.30. The trailing tip eases in over its first 22% as 0.12 + 0.88 sin(πu/2)^0.8.
- Progenitor: r = 47.1 px at (268.2, 234.4) in SVG coordinates.
- Ink box: (55.7, 85.6)–(456.3, 426.4), centred in the square.
- Gradient: `x1=0 y1=401 x2=0 y2=111`. Light stops are `#9b8f88` / `#645a56` / `#2e2424`. Dark stops are `#ab9f98` / `#d0c6c0` / `#f6ede8`.
- At 32 px the trailing arm is about 1.4 px wide before its tip taper, the leading arm 2.2 px and the progenitor 5.9 px across.

## Social preview

- 1280 x 640, background `#ffffff`.
- The wordmark ink is 980 px wide, centred horizontally. Below it is the tagline "GPU N-body simulations of stellar streams in Python" (Manrope 400, 32 px, `#645a56`), 46 px under the logo ink. The two together are centred vertically.
- Ink extents: x 150–1130, y 212–428. This is inside the central 1100 x 540 safe area (x 90–1190, y 50–590).
