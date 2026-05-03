#!/usr/bin/env python3

import os
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================
design_name = "bigblue4"
aux_file    = "bigblue4.aux"

DBU       = 1
LEF_SCALE = 1.0
PL_SCALE  = 1.0

# =========================================================
# AUX PARSER
# =========================================================
def parse_aux(path):
    files = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("RowBasedPlacement"):
                parts = line.split(":")
                if len(parts) > 1:
                    files = parts[1].strip().split()
    return files


# =========================================================
# SCL BOUNDS
# =========================================================
def load_scl_bounds(file):
    if not os.path.exists(file):
        return None

    min_x = min_y =  float('inf')
    max_x = max_y = -float('inf')
    cur_y = cur_h = cur_x = cur_w = None

    with open(file) as f:
        for line in f:
            line = line.strip()
            tok  = line.split()
            if not tok:
                continue
            if tok[0] == "Coordinate"   and len(tok) >= 3: cur_y = float(tok[2])
            if tok[0] == "Height"       and len(tok) >= 3: cur_h = float(tok[2])
            if tok[0] == "SubrowOrigin" and len(tok) >= 3: cur_x = float(tok[2])
            if tok[0] == "Sitespacing"  and len(tok) >= 3: cur_w = float(tok[2])
            if tok[0] == "NumSites"     and len(tok) >= 3 \
                    and cur_x is not None and cur_w is not None:
                row_max_x = cur_x + float(tok[2]) * cur_w
                min_x = min(min_x, cur_x)
                max_x = max(max_x, row_max_x)
            if cur_y is not None and cur_h is not None:
                min_y = min(min_y, cur_y)
                max_y = max(max_y, cur_y + cur_h)

    if min_x == float('inf'):
        return None
    return min_x, min_y, max_x, max_y


# =========================================================
# NODE LOADER
# =========================================================
def load_nodes(file):
    nodes = {}
    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "NumNodes" in line or "NumTerminals" in line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                try:
                    w = float(parts[1]) / PL_SCALE
                    h = float(parts[2]) / PL_SCALE
                    if w > 0 and h > 0:
                        nodes[name] = (w, h)
                except:
                    pass
    return nodes


# =========================================================
# PLACEMENT LOADER
# =========================================================
def load_pl(file):
    pl = {}
    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                try:
                    x = float(parts[1]) / PL_SCALE
                    y = float(parts[2]) / PL_SCALE
                    pl[name] = (x, y)
                except:
                    pass
    return pl


# =========================================================
# NET LOADER — captures per-pin offsets
# returns list of nets, each net = list of (inst, dx, dy)
# =========================================================
def load_nets(file, valid_nodes):
    nets    = []
    current = []

    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "NetDegree" in line:
                if current:
                    nets.append(current)
                current = []
                continue
            # format: inst_name  DIR : dx  dy
            #      or inst_name  DIR          (no offset)
            parts = line.replace(":", " ").split()
            if not parts:
                continue
            inst = parts[0]
            if inst not in valid_nodes:
                continue
            try:
                dx = float(parts[2]) / PL_SCALE if len(parts) >= 4 else 0.0
                dy = float(parts[3]) / PL_SCALE if len(parts) >= 4 else 0.0
            except:
                dx, dy = 0.0, 0.0
            current.append((inst, dx, dy))

    if current:
        nets.append(current)

    # deduplicate pins per net (keep first occurrence per inst+offset)
    clean = []
    for net in nets:
        seen = set()
        c = []
        for (inst, dx, dy) in net:
            key = (inst, dx, dy)
            if key not in seen:
                c.append((inst, dx, dy))
                seen.add(key)
        if len(c) >= 2:
            clean.append(c)

    return clean


# =========================================================
# SCL LOADER
# =========================================================
def load_scl(file):
    if not os.path.exists(file):
        return []
    rows = []
    with open(file) as f:
        for line in f:
            if line.strip().startswith("CoreRow"):
                rows.append(line.strip())
    return rows


# =========================================================
# MAIN LOAD
# =========================================================
aux_files = parse_aux(aux_file)
print("[AUX]"); print(aux_files)

nodes_file = "bigblue4.nodes"
nets_file  = "bigblue4.nets"
pl_file    = "bigblue4.pl"
scl_file   = "bigblue4.scl"

nodes = load_nodes(nodes_file)
pl    = load_pl(pl_file)
nets  = load_nets(nets_file, nodes)
rows  = load_scl(scl_file)

print("\n[SUMMARY]")
print("nodes    :", len(nodes))
print("placement:", len(pl))
print("nets     :", len(nets))
print("rows     :", len(rows))


# =========================================================
# COORDINATE / SIZE DIAGNOSTICS
# =========================================================
all_x = [pl.get(n, (0, 0))[0] for n in nodes]
all_y = [pl.get(n, (0, 0))[1] for n in nodes]
all_w = [w for (w, h) in nodes.values()]
all_h = [h for (w, h) in nodes.values()]

min_x, max_x = min(all_x), max(all_x)
min_y, max_y = min(all_y), max(all_y)
min_w, max_w = min(all_w), max(all_w)
min_h, max_h = min(all_h), max(all_h)

print("\n[COORD RANGE]  (after PL_SCALE)")
print(f"  PL  X : {min_x:.1f} .. {max_x:.1f}")
print(f"  PL  Y : {min_y:.1f} .. {max_y:.1f}")
print(f"  NODE W: {min_w:.1f} .. {max_w:.1f}")
print(f"  NODE H: {min_h:.1f} .. {max_h:.1f}")


# =========================================================
# MACRO vs STD CELL CLASSIFICATION
# =========================================================
areas = sorted(w * h for (w, h) in nodes.values())
median_area = areas[len(areas) // 2]
macro_threshold = 10.0 * median_area

is_macro = {}
for inst, (w, h) in nodes.items():
    is_macro[inst] = (w * h) >= macro_threshold

n_macros = sum(1 for v in is_macro.values() if v)
n_std    = len(nodes) - n_macros
print(f"\n[CLASSIFICATION]  median_area={median_area:.2f}  threshold={macro_threshold:.2f}")
print(f"  macros   : {n_macros}")
print(f"  std cells: {n_std}")


# =========================================================
# DIE AREA — macros anchor the die, all instances must fit
# =========================================================
die_x, die_y = 0, 0
for inst in nodes:
    if not is_macro[inst]:
        continue
    w, h = nodes[inst]
    x, y = pl.get(inst, (0, 0))
    die_x = max(die_x, int((x + w) * DBU / LEF_SCALE) + 1)
    die_y = max(die_y, int((y + h) * DBU / LEF_SCALE) + 1)

print(f"\n[DIE AREA from macros] ({die_x} x {die_y}) DBU")

# expand to fit all instances
for inst, (w, h) in nodes.items():
    x, y = pl.get(inst, (0, 0))
    die_x = max(die_x, int((x + w) * DBU / LEF_SCALE) + 1)
    die_y = max(die_y, int((y + h) * DBU / LEF_SCALE) + 1)

print(f"[DIE AREA after expansion] ({die_x} x {die_y}) DBU")

# utilization check
total_cell_area_dbu = sum(
    (w * DBU / LEF_SCALE) * (h * DBU / LEF_SCALE)
    for (w, h) in nodes.values()
)
die_area_dbu = die_x * die_y
print(f"[UTILIZATION CHECK] cell={total_cell_area_dbu:.3e}  "
      f"die={die_area_dbu:.3e}  "
      f"util={100*total_cell_area_dbu/die_area_dbu:.1f}%")


# =========================================================
# MACRO MAP  (shape → LEF macro name)
# =========================================================
macro_map  = {}   # (w,h) → "MACRO_N"
inst_macro = {}   # inst  → "MACRO_N"

mid = 0
for inst, (w, h) in nodes.items():
    key = (w, h)
    if key not in macro_map:
        macro_map[key] = f"MACRO_{mid}"
        mid += 1
    inst_macro[inst] = macro_map[key]


# =========================================================
# COLLECT UNIQUE PINS PER MACRO SHAPE  (from net offsets)
# =========================================================
macro_pins = defaultdict(set)   # (w,h) → {(dx,dy), ...}

for net in nets:
    for (inst, dx, dy) in net:
        w, h = nodes[inst]
        macro_pins[(w, h)].add((dx, dy))

# ensure every macro shape has at least one pin
for key in macro_map:
    if not macro_pins[key]:
        macro_pins[key].add((0.0, 0.0))

# assign stable index to each (dx,dy) per macro shape
macro_pin_index = {}   # (w,h) → { (dx,dy): pin_idx }
for key, offsets in macro_pins.items():
    macro_pin_index[key] = {off: i for i, off in enumerate(sorted(offsets))}

print("\n[PIN STATS]")
print("max pins on single macro shape:",
      max(len(v) for v in macro_pin_index.values()))
print("total unique (shape, pin) combos:",
      sum(len(v) for v in macro_pin_index.values()))


# =========================================================
# WRITE LEF
# =========================================================
lef_file = "bigblue4.lef"

with open(lef_file, "w") as f:
    f.write("VERSION 5.8 ;\n")
    f.write('BUSBITCHARS "[]" ;\n')
    f.write('DIVIDERCHAR "/" ;\n\n')

    f.write("UNITS\n")
    f.write(f"  DATABASE MICRONS {DBU} ;\n")
    f.write("END UNITS\n\n")

    f.write("PROPERTYDEFINITIONS\n")
    f.write("  LAYER LEF58_CORNERSPACING STRING ;\n")
    f.write("END PROPERTYDEFINITIONS\n\n")

    f.write("CLEARANCEMEASURE EUCLIDEAN ;\n")
    f.write("MANUFACTURINGGRID 0.0005 ;\n")
    f.write("USEMINSPACING OBS ON ;\n\n")

    f.write("LAYER M1\n")
    f.write("  TYPE ROUTING ;\n")
    f.write("  DIRECTION HORIZONTAL ;\n")
    f.write("  PITCH 0.2 ;\n")
    f.write("  WIDTH 0.1 ;\n")
    f.write("END M1\n\n")

    for (w, h), macro in macro_map.items():
        W = max(w / LEF_SCALE, 0.001)
        H = max(h / LEF_SCALE, 0.001)

        f.write(f"MACRO {macro}\n")
        f.write("  CLASS CORE ;\n")
        f.write("  ORIGIN 0 0 ;\n")
        f.write(f"  SIZE {W:.6f} BY {H:.6f} ;\n\n")

        pin_map = macro_pin_index[(w, h)]
        for (dx, dy), pi in sorted(pin_map.items(), key=lambda x: x[1]):
            # offset is from cell center → convert to origin-relative
            px = W / 2 + dx / LEF_SCALE
            py = H / 2 + dy / LEF_SCALE
            # clamp inside macro boundary
            pw = min(0.1, W)
            ph = min(0.1, H)
            px = max(0.0, min(W - pw, px))
            py = max(0.0, min(H - ph, py))

            f.write(f"  PIN P{pi}\n")
            f.write("    DIRECTION INOUT ;\n")
            f.write("    PORT\n")
            f.write("      LAYER M1 ;\n")
            f.write(f"      RECT {px:.6f} {py:.6f} {px+pw:.6f} {py+ph:.6f} ;\n")
            f.write("    END\n")
            f.write(f"  END P{pi}\n\n")

        f.write(f"END {macro}\n\n")

    f.write("END LIBRARY\n")

print(f"\n[LEF] written → {lef_file}")


# =========================================================
# WRITE DEF
# =========================================================
def write_rows(f, scl_file, dbu, lef_scale, pl_scale):
    if not os.path.exists(scl_file):
        return
    cur = {}
    row_id = 0
    with open(scl_file) as sf:
        for line in sf:
            tok = line.strip().split()
            if not tok:
                continue
            if tok[0] == "Coordinate"   and len(tok) >= 3:
                cur['y'] = float(tok[2]) / pl_scale
            if tok[0] == "Height"       and len(tok) >= 3:
                cur['h'] = float(tok[2]) / pl_scale
            if tok[0] == "SubrowOrigin" and len(tok) >= 3:
                cur['x'] = float(tok[2]) / pl_scale
            if tok[0] == "Sitespacing"  and len(tok) >= 3:
                cur['step'] = float(tok[2]) / pl_scale
            if tok[0] == "NumSites"     and len(tok) >= 3:
                cur['n'] = int(tok[2])
                xi   = max(0, int(round(cur['x']    * dbu / lef_scale)))
                yi   = max(0, int(round(cur['y']    * dbu / lef_scale)))
                step = max(1, int(round(cur['step'] * dbu / lef_scale)))
                f.write(
                    f"ROW ROW_{row_id} core {xi} {yi} N"
                    f" DO {cur['n']} BY 1 STEP {step} 0 ;\n"
                )
                row_id += 1
                cur = {}
    print(f"[ROWS] written {row_id} rows")


def_file = "bigblue4.def"
valid_nets = [net for net in nets if len(net) >= 2]

with open(def_file, "w") as f:
    f.write("VERSION 5.8 ;\n")
    f.write('DIVIDERCHAR "/" ;\n')
    f.write('BUSBITCHARS "[]" ;\n\n')
    f.write(f"UNITS DISTANCE MICRONS {DBU} ;\n\n")
    f.write(f"DESIGN {design_name} ;\n\n")
    f.write(f"DIEAREA ( 0 0 ) ( {die_x} {die_y} ) ;\n\n")

    write_rows(f, scl_file, DBU, LEF_SCALE, PL_SCALE)
    f.write("\n")

    # COMPONENTS
    f.write(f"COMPONENTS {len(nodes)} ;\n")
    for inst in nodes:
        x, y  = pl.get(inst, (0, 0))
        xi    = max(0, int(round(x * DBU / LEF_SCALE)))
        yi    = max(0, int(round(y * DBU / LEF_SCALE)))
        lef_macro = inst_macro[inst]
        status    = "FIXED" if is_macro[inst] else "PLACED"
        f.write(f"  - {inst} {lef_macro} + {status} ( {xi} {yi} ) N ;\n")
    f.write("END COMPONENTS\n\n")

    # NETS — use real pin offsets
    f.write(f"NETS {len(valid_nets)} ;\n")
    for nid, net in enumerate(valid_nets):
        f.write(f"  - net{nid}")
        for (inst, dx, dy) in net:
            w, h    = nodes[inst]
            pin_idx = macro_pin_index[(w, h)][(dx, dy)]
            f.write(f" ( {inst} P{pin_idx} )")
        f.write(" ;\n")
    f.write("END NETS\n\n")
    f.write("END DESIGN\n")

print(f"[DEF] written → {def_file}")


# =========================================================
# FINAL VALIDATION
# =========================================================
total_net_pins  = sum(len(n) for n in valid_nets)
total_inst_pins = sum(len(macro_pin_index[nodes[inst]]) for inst in nodes)

print("\n[FINAL CHECK]")
print("instances      :", len(nodes))
print("nets           :", len(valid_nets))
print("total net pins :", total_net_pins)
print("total inst pins:", total_inst_pins)

if total_net_pins <= total_inst_pins:
    print(f"PASS  net_pins {total_net_pins} <= inst_pins {total_inst_pins}")
else:
    print(f"FAIL  net_pins {total_net_pins}  > inst_pins {total_inst_pins}")

print("\nDONE")
