"""Pure deterministic HTML and SVG renderers."""

from __future__ import annotations

import html
import math
from pathlib import PurePosixPath

from .models import (
    AssetBlock,
    DocumentSpec,
    Link,
    NoticeBlock,
    PageKind,
    PageSpec,
    PlotBlock,
    PlotGroupBlock,
    PlotKind,
    PlotPoint,
    PlotSpec,
    ReportBlock,
    TableBlock,
)

_COLORS = (
    "#2457c5",
    "#0f766e",
    "#7c3aed",
    "#d65a1f",
    "#087c95",
    "#c63131",
)


def render_document(document: DocumentSpec) -> dict[PurePosixPath, str]:
    """Render all pages without filesystem access or external commands."""

    return {page.path: render_page(document, page) for page in document.pages}


def render_page(document: DocumentSpec, page: PageSpec) -> str:
    """Render one validated page using only escaped typed content."""

    if page.target.kind is PageKind.PROJECT:
        navigation = '<span aria-current="page">Project overview</span>'
    else:
        navigation = '<a href="../../index.html">Project overview</a><span>/</span>'
        navigation += f'<span aria-current="page">{_escape(page.title)}</span>'
    sections = "".join(
        _render_block(block, slot.slot.value)
        for slot in page.slots
        for block in slot.blocks
    )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<meta http-equiv="Content-Security-Policy" '
        'content="default-src &#39;none&#39;; style-src &#39;unsafe-inline&#39;; '
        'img-src &#39;self&#39; data:; base-uri &#39;none&#39;; form-action &#39;none&#39;">\n'
        f"<title>{_escape(page.title)} · {_escape(document.title)}</title>\n"
        f"<style>{base_css()}</style>\n"
        "</head>\n<body>\n"
        f'<header><nav aria-label="Breadcrumb">{navigation}</nav>'
        f"<h1>{_escape(page.title)}</h1></header>\n"
        f"<main>{sections}</main>\n"
        f'<footer>Schema: <code>{_escape(document.schema_version)}</code></footer>\n'
        "</body>\n</html>\n"
    )


def render_plot_svg(plot: PlotSpec, *, width: int = 920, height: int = 400) -> str:
    """Render a typed line, scatter, or bar plot as an inline SVG."""

    if width < 360 or height < 260:
        raise ValueError("SVG dimensions are too small")
    points = tuple(point for series in plot.series for point in series.points)
    xs = tuple(point.x for point in points)
    ys = tuple(point.y for point in points)
    x_min, x_max = _padded_range(xs, include_zero=plot.kind is PlotKind.BAR)
    if plot.x_min is not None:
        x_min = plot.x_min
    y_min, y_max = _padded_range(ys, include_zero=plot.zero_line or plot.kind is PlotKind.BAR)
    if plot.y_min is not None:
        y_min = plot.y_min
    left, right, top, bottom = 74.0, width - 28.0, 32.0, height - 72.0
    sx = _scale(x_min, x_max, left, right)
    sy = _scale(y_min, y_max, bottom, top)
    parts = [
        f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_attr(plot.y_label)} by {_attr(plot.x_label)}" '
        'xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]
    parts.extend(
        _axes(
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
    )
    if plot.zero_line and y_min <= 0 <= y_max:
        zero_y = sy(0.0)
        parts.append(
            f'<line class="zero-line" x1="{left:.2f}" y1="{zero_y:.2f}" '
            f'x2="{right:.2f}" y2="{zero_y:.2f}"/>'
        )

    all_x = sorted(set(xs))
    x_step = min(
        (
            second - first
            for first, second in zip(all_x, all_x[1:], strict=False)
        ),
        default=max(x_max - x_min, 1.0),
    )
    bar_width = max(2.0, abs(sx(x_min + x_step) - sx(x_min)) * 0.7 / len(plot.series))
    for series_index, series in enumerate(plot.series):
        color = _COLORS[series_index % len(_COLORS)]
        ordered = tuple(sorted(series.points, key=lambda point: (point.x, point.y, point.label)))
        if plot.kind is PlotKind.LINE:
            # Absent occupancy bins are not sampled states. Do not connect a
            # curve across those gaps or invent an infinite/zero energy value.
            segments: list[list[PlotPoint]] = []
            for point in ordered:
                if not segments or point.break_before:
                    segments.append([])
                segments[-1].append(point)
            for segment in segments:
                if len(segment) < 2:
                    continue
                coordinates = " ".join(
                    f"{sx(point.x):.2f},{sy(point.y):.2f}" for point in segment
                )
                parts.append(
                    f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                    f'stroke-width="2.4"><title>{_escape(series.label)}</title></polyline>'
                )
        for point in ordered:
            title = f"{series.label}: {point.x:g}, {point.y:g}"
            if point.label:
                title = f"{point.label} · {title}"
            if plot.kind is PlotKind.BAR:
                zero_y = sy(0.0)
                point_y = sy(point.y)
                offset = (series_index - (len(plot.series) - 1) / 2) * bar_width
                parts.append(
                    f'<rect x="{sx(point.x) + offset - bar_width / 2:.2f}" '
                    f'y="{min(zero_y, point_y):.2f}" width="{bar_width:.2f}" '
                    f'height="{max(1.0, abs(zero_y - point_y)):.2f}" fill="{color}" rx="2">'
                    f"<title>{_escape(title)}</title></rect>"
                )
            elif plot.kind is PlotKind.SCATTER or point.emphasized or point.break_before:
                radius = 2.6 if plot.kind is PlotKind.SCATTER else 2.8
                opacity = (
                    ' fill-opacity="0.48"'
                    if plot.kind is PlotKind.SCATTER
                    else ""
                )
                parts.append(
                    f'<circle cx="{sx(point.x):.2f}" cy="{sy(point.y):.2f}" '
                    f'r="{radius:.1f}" fill="{color}"{opacity}>'
                    f"<title>{_escape(title)}</title></circle>"
                )
        legend_x = left + series_index % 4 * ((right - left) / 4)
        legend_y = height - 18 - series_index // 4 * 17
        parts.append(
            f'<rect x="{legend_x:.2f}" y="{legend_y - 9:.2f}" width="18" height="3" '
            f'fill="{color}"/><text class="legend" x="{legend_x + 24:.2f}" '
            f'y="{legend_y:.2f}">{_escape(series.label)}</text>'
        )
    parts.extend(
        (
            f'<text class="axis-label" x="{(left + right) / 2:.2f}" y="{height - 38:.2f}" '
            f'text-anchor="middle">{_escape(plot.x_label)}</text>',
            f'<text class="axis-label" x="18" y="{(top + bottom) / 2:.2f}" '
            f'text-anchor="middle" transform="rotate(-90 18 {(top + bottom) / 2:.2f})">'
            f"{_escape(plot.y_label)}</text>",
            "</svg>",
        )
    )
    return "".join(parts)


def base_css() -> str:
    return """
:root {
  color-scheme: light;
  --ink: #172033;
  --muted: #667085;
  --line: #d9dee8;
  --blue: #2457c5;
  --panel: #fff;
  --stripe: #f8fafc;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: #eef1f6;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
body::before { content: ""; display: block; height: 6px; background: var(--blue); }
header, main, footer { width: min(1280px, calc(100vw - 32px)); margin-inline: auto; }
header { padding: 22px 0 2px; }
nav { display: flex; gap: 8px; color: var(--muted); font-size: 13px; }
h1 { margin: 14px 0 18px; font-size: 27px; }
h2 {
  margin: 0 0 14px;
  padding-left: 10px;
  border-left: 3px solid var(--blue);
  font-size: 18px;
}
section {
  margin: 16px 0;
  padding: 20px 24px;
  border: 1px solid var(--line);
  border-radius: 9px;
  background: var(--panel);
  box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
section.notice.info { border-left: 5px solid #4b74ce; }
section.notice.warning { border-left: 5px solid #d68b18; background: #fffdf5; }
section.notice.blocked { border-left: 5px solid #c43131; background: #fff8f8; }
.muted, .caption { color: var(--muted); }
.caption { margin: -4px 0 12px; font-size: 13px; }
.table-scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 7px; }
table { width: 100%; border-collapse: collapse; background: #fff; }
th, td { padding: 8px 10px; border-bottom: 1px solid #e6e9ef; text-align: left; }
th { background: #f1f4f8; color: #344054; font-size: 12px; }
tbody tr:nth-child(even) td { background: var(--stripe); }
td.numeric, th.numeric { text-align: right; font-variant-numeric: tabular-nums; }
.unit { display: block; color: var(--muted); font-size: 10px; }
.asset-list { display: grid; gap: 9px; margin: 0; padding: 0; list-style: none; }
.asset-list li { padding: 10px 12px; border: 1px solid var(--line); border-radius: 6px; }
.asset-description { display: block; margin-top: 3px; color: var(--muted); font-size: 12px; }
a { color: var(--blue); font-weight: 650; text-decoration: none; }
a:hover { text-decoration: underline; }
.plot-wrap { max-width: 100%; overflow-x: auto; }
.plot { display: block; width: min(100%, 920px); height: auto; margin: 0 auto; }
.plot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.plot-panel {
  min-width: 0;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: #fff;
}
.plot-panel h3 { margin: 0 0 8px; font-size: 15px; }
.plot-panel .plot { width: 100%; }
.axis { stroke: #667085; stroke-width: 1; }
.grid { stroke: #e4e8ef; stroke-width: 1; }
.zero-line { stroke: #9b1c1c; stroke-width: 1.2; stroke-dasharray: 6 5; }
.tick, .legend { fill: #475467; font-size: 11px; }
.axis-label { fill: #344054; font-size: 12px; font-weight: 650; }
footer { padding: 14px 0 32px; color: var(--muted); font-size: 12px; }
code { padding: 2px 5px; border-radius: 4px; background: #e7ebf2; }
@media (max-width: 620px) {
  header, main, footer { width: min(100% - 18px, 1280px); }
  section { padding: 16px 13px; }
  th, td { padding: 7px 8px; font-size: 12px; }
  .plot-grid { grid-template-columns: 1fr; }
}
""".strip()


def _render_block(block: ReportBlock, slot: str) -> str:
    if isinstance(block, NoticeBlock):
        body = "<br>".join(_escape(line) for line in block.body.splitlines())
        return (
            f'<section id="{_attr(block.block_id)}" data-slot="{_attr(slot)}" '
            f'class="notice {block.level.value}"><h2>{_escape(block.title)}</h2>'
            f"<p>{body}</p></section>\n"
        )
    if isinstance(block, TableBlock):
        return _render_table(block, slot)
    if isinstance(block, PlotBlock):
        caption = _caption(block.caption)
        return (
            f'<section id="{_attr(block.block_id)}" data-slot="{_attr(slot)}">'
            f"<h2>{_escape(block.title)}</h2>{caption}"
            f'<div class="plot-wrap">{render_plot_svg(block.plot)}</div></section>\n'
        )
    if isinstance(block, PlotGroupBlock):
        caption = _caption(block.caption)
        panels = "".join(
            '<article class="plot-panel">'
            f"<h3>{_escape(panel.title)}</h3>{_caption(panel.caption)}"
            f'<div class="plot-wrap">{render_plot_svg(panel.plot, width=640, height=330)}</div>'
            "</article>"
            for panel in block.panels
        )
        return (
            f'<section id="{_attr(block.block_id)}" data-slot="{_attr(slot)}">'
            f"<h2>{_escape(block.title)}</h2>{caption}"
            f'<div class="plot-grid">{panels}</div></section>\n'
        )
    if isinstance(block, AssetBlock):
        caption = _caption(block.caption)
        links = "".join(
            "<li>"
            f'{_render_link(link)}'
            + (
                f'<span class="asset-description">{_escape(link.description)}</span>'
                if link.description
                else ""
            )
            + "</li>"
            for link in block.links
        )
        return (
            f'<section id="{_attr(block.block_id)}" data-slot="{_attr(slot)}">'
            f"<h2>{_escape(block.title)}</h2>{caption}"
            f'<ul class="asset-list">{links}</ul></section>\n'
        )
    raise TypeError(f"Unsupported report block: {type(block).__name__}")


def _render_table(block: TableBlock, slot: str) -> str:
    caption = _caption(block.caption)
    headings = "".join(
        f'<th scope="col" class="{"numeric" if column.numeric else ""}">'
        f"{_escape(column.label)}"
        + (f'<span class="unit">{_escape(column.unit)}</span>' if column.unit else "")
        + "</th>"
        for column in block.columns
    )
    if block.rows:
        rows = "".join(
            "<tr>"
            + "".join(
                f'<td class="{"numeric" if column.numeric else ""}">{_render_cell(cell)}</td>'
                for column, cell in zip(block.columns, row, strict=True)
            )
            + "</tr>"
            for row in block.rows
        )
    else:
        rows = (
            f'<tr><td colspan="{len(block.columns)}" class="muted">'
            "No records were supplied.</td></tr>"
        )
    return (
        f'<section id="{_attr(block.block_id)}" data-slot="{_attr(slot)}">'
        f"<h2>{_escape(block.title)}</h2>{caption}"
        f'<div class="table-scroll"><table><thead><tr>{headings}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div></section>\n"
    )


def _render_cell(value: object) -> str:
    if isinstance(value, Link):
        return _render_link(value)
    if value is None or value == "":
        return '<span class="muted">N/A</span>'
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return _escape(str(value))


def _render_link(link: Link) -> str:
    external = bool(link.href.split(":", 1)[0].lower() in {"http", "https", "mailto"})
    rel = ' rel="noopener noreferrer"' if external else ""
    return f'<a href="{_attr(link.href)}"{rel}>{_escape(link.label)}</a>'


def _caption(value: str) -> str:
    return f'<p class="caption">{_escape(value)}</p>' if value else ""


def _axes(
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[str]:
    parts = [
        f'<line class="axis" x1="{left:.2f}" y1="{bottom:.2f}" '
        f'x2="{right:.2f}" y2="{bottom:.2f}"/>',
        f'<line class="axis" x1="{left:.2f}" y1="{top:.2f}" '
        f'x2="{left:.2f}" y2="{bottom:.2f}"/>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + fraction * (right - left)
        x_value = x_min + fraction * (x_max - x_min)
        y = bottom - fraction * (bottom - top)
        y_value = y_min + fraction * (y_max - y_min)
        parts.extend(
            (
                f'<line class="grid" x1="{x:.2f}" y1="{top:.2f}" '
                f'x2="{x:.2f}" y2="{bottom:.2f}"/>',
                f'<text class="tick" x="{x:.2f}" y="{bottom + 17:.2f}" '
                f'text-anchor="middle">{x_value:g}</text>',
                f'<line class="grid" x1="{left:.2f}" y1="{y:.2f}" '
                f'x2="{right:.2f}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{left - 8:.2f}" y="{y + 4:.2f}" '
                f'text-anchor="end">{y_value:g}</text>',
            )
        )
    return parts


def _padded_range(values: tuple[float, ...], *, include_zero: bool) -> tuple[float, float]:
    if include_zero:
        values = (*values, 0.0)
    low, high = min(values), max(values)
    if math.isclose(low, high):
        padding = max(abs(low) * 0.08, 1.0)
        return low - padding, high + padding
    padding = (high - low) * 0.06
    return low - padding, high + padding


def _scale(low: float, high: float, output_low: float, output_high: float):
    span = high - low

    def scale(value: float) -> float:
        return output_low + (value - low) / span * (output_high - output_low)

    return scale


def _escape(value: object) -> str:
    return html.escape(str(value), quote=False)


def _attr(value: object) -> str:
    return html.escape(str(value), quote=True)
