#!/usr/bin/env python3
"""
Generate macro positions CSV for --load-macros.

Places macro clusters at die corners (or quadrant centers with --center).
Reads cluster assignments from macro_colors.csv and macro sizes from LEF.

Usage:
    python3 tools/place_macros.py \
        --lef tests/ispd19_test8/ispd19_test8.input.lef \
        --def tests/ispd19_test8/ispd19_test8.input.def \
        --colors macro_colors.csv \
        --output macros.csv [--center]
"""

import argparse
import csv
import re
from collections import defaultdict


def parse_die(def_path):
    with open(def_path) as f:
        for line in f:
            m = re.search(r'DIEAREA\s*\(\s*\d+\s+\d+\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', line)
            if m:
                units_line = None
    with open(def_path) as f:
        content = f.read()
    units = int(re.search(r'UNITS DISTANCE MICRONS (\d+)', content).group(1))
    m = re.search(r'DIEAREA\s*\(\s*\d+\s+\d+\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
    return int(m.group(1)) / units, int(m.group(2)) / units


def parse_macro_sizes(lef_path):
    """Returns {macro_name: (width, height)}"""
    sizes = {}
    with open(lef_path) as f:
        current = None
        for line in f:
            m = re.match(r'\s*MACRO\s+(\S+)', line)
            if m:
                current = m.group(1)
            if current:
                m = re.search(r'SIZE\s+([\d.]+)\s+BY\s+([\d.]+)', line)
                if m:
                    sizes[current] = (float(m.group(1)), float(m.group(2)))
    return sizes


def parse_fixed_macros(def_path):
    """Returns {inst_name: macro_type} for FIXED instances."""
    macros = {}
    with open(def_path) as f:
        for line in f:
            m = re.match(r'\s*-\s+(\S+)\s+(\S+)\s+\+\s+FIXED', line)
            if m:
                macros[m.group(1)] = m.group(2)
    return macros


def parse_colors(colors_path):
    """Returns {instance_id: (r,g,b)}"""
    colors = {}
    with open(colors_path) as f:
        for row in csv.DictReader(f):
            colors[int(row['instance_id'])] = (int(row['r']), int(row['g']), int(row['b']))
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lef', required=True)
    parser.add_argument('--def', dest='def_path', required=True)
    parser.add_argument('--colors', required=True, help='macro_colors.csv')
    parser.add_argument('--output', default='macros.csv')
    parser.add_argument('--center', action='store_true', help='Place at quadrant centers instead of corners')
    args = parser.parse_args()

    dw, dh = parse_die(args.def_path)
    macro_sizes = parse_macro_sizes(args.lef)
    colors = parse_colors(args.colors)

    # Group instance_ids by color cluster
    cluster_map = defaultdict(list)
    for iid, color in colors.items():
        cluster_map[color].append(iid)
    clusters = sorted(cluster_map.values(), key=lambda c: c[0])
    n_clusters = len(clusters)

    gap = 5.0

    if args.center:
        import math
        cols = math.ceil(math.sqrt(n_clusters))
        rows = math.ceil(n_clusters / cols)
        anchors = []
        for ci in range(n_clusters):
            col, row = ci % cols, ci // cols
            anchors.append((dw * (col + 0.5) / cols, dh * (row + 0.5) / rows, 0, 0))
    else:
        # Corners: bottom-left, bottom-right, top-left, top-right
        anchors = [
            (0,   0,   1,  1),
            (dw,  0,  -1,  1),
            (0,   dh,  1, -1),
            (dw,  dh, -1, -1),
        ]

    # Map instance_id to macro type:
    # Rust DB puts fixed instances at end, in order they appear in DEF COMPONENTS
    # Count movable instances (non-FIXED) to find offset
    fixed_in_order = []
    with open(args.def_path) as f:
        for line in f:
            m = re.match(r'\s*-\s+(\S+)\s+(\S+)\s+\+\s+FIXED', line)
            if m:
                fixed_in_order.append((m.group(1), m.group(2)))

    # Total components
    with open(args.def_path) as f:
        content = f.read()
    total = int(re.search(r'COMPONENTS\s+(\d+)', content).group(1))
    num_movable = total - len(fixed_in_order)

    iid_to_type = {}
    for k, (name, mtype) in enumerate(fixed_in_order):
        iid_to_type[num_movable + k] = mtype

    result = []
    for ci, cluster in enumerate(clusters):
        if args.center:
            cx, cy, _, _ = anchors[ci]
            # Group by type within cluster
            by_type = defaultdict(list)
            for iid in cluster:
                mtype = iid_to_type.get(iid, list(macro_sizes.keys())[0])
                by_type[mtype].append(iid)

            # Stack rows of each type, centered on cx,cy
            total_h = sum(macro_sizes[t][1] for t in by_type) + gap * (len(by_type) - 1)
            y0 = cy - total_h / 2
            for mtype, iids in sorted(by_type.items()):
                mw, mh = macro_sizes[mtype]
                total_w = len(iids) * mw + (len(iids) - 1) * gap
                x0 = cx - total_w / 2
                for k, iid in enumerate(iids):
                    result.append((iid, max(0, x0 + k * (mw + gap)), max(0, y0)))
                y0 += mh + gap
        else:
            ax, ay, xd, yd = anchors[ci % len(anchors)]
            by_type = defaultdict(list)
            for iid in cluster:
                mtype = iid_to_type.get(iid, list(macro_sizes.keys())[0])
                by_type[mtype].append(iid)

            row_offset = 0
            for mtype, iids in sorted(by_type.items()):
                mw, mh = macro_sizes[mtype]
                for k, iid in enumerate(iids):
                    x = ax + xd * (k * (mw + gap)) if xd > 0 else ax + xd * ((k + 1) * mw + k * gap)
                    y = ay + yd * row_offset if yd > 0 else ay + yd * (row_offset + mh)
                    result.append((iid, max(0, min(x, dw - mw)), max(0, min(y, dh - mh))))
                row_offset += mh + gap

    with open(args.output, 'w') as f:
        f.write("instance_id,x,y\n")
        for iid, x, y in sorted(result):
            f.write(f"{iid},{x:.4f},{y:.4f}\n")
    print(f"Wrote {len(result)} macro positions to {args.output}")


if __name__ == '__main__':
    main()
