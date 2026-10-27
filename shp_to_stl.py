#!/usr/bin/env python3
"""
Created by KJ Ayres-Guerra 10.27.2026
For Printing Resaca Files for Cameron County Gift

shp_to_stl.py - Turn a shapefile into an STL you can 3D print.

There are two ways to print your map:

  RAISED (the default)  - features stand UP off a flat base, like little walls.
                          Good for land, parcels, habitat patches, anything you
                          want people to touch.

  CARVED (--carve)      - features are cut DOWN into a solid block, like
                          channels. Good for water: resacas, creeks, lagoons.
                          Prints as one sturdy piece.

Install (one time):
    pip install geopandas trimesh shapely mapbox_earcut networkx

Examples:
    # RAISED: everything 3 mm tall, model 150 mm wide, 2 mm base plate
    python shp_to_stl.py parcels.shp parcels.stl

    # RAISED: height from an attribute (e.g. elevation or a priority score)
    python shp_to_stl.py marsh.shp marsh.stl --height-field SCORE --max-height 15

    # RAISED: shoreline lines printed as 1.5 mm wide ridges, no base
    python shp_to_stl.py shoreline.shp shoreline.stl --line-width 1.5 --no-base

    # CARVED: thin water bodies cut 2 mm into a block as channels
    python shp_to_stl.py resacas.shp resacas.stl --width 200 --grow 0.4 --carve --height 2


===========================================================================
HOW IT WORKS, IN PLAIN ENGLISH
===========================================================================

STEP 1 - Read the map
    Open the shapefile.
    Throw out any rows that have no shape in them.

STEP 2 - Get everything into real distances
    IF the map is in latitude/longitude (degrees):
        Convert it to UTM (meters), because degrees stretch and squish
        the shapes and the print would come out distorted.

STEP 3 - Shrink the map down to printer size
    Find the box that fits around every feature.
    Figure out how much to shrink it so the longest side = --width mm.
    Slide everything so the bottom-left corner sits at (0, 0).

STEP 4 - Decide how tall each feature is
    IF you gave a --height-field:
        Lowest value in that column  -> --min-height mm
        Highest value in that column -> --max-height mm
        Everything else falls in between.
    OTHERWISE:
        Every feature is --height mm tall.

STEP 5 - Clean up each feature, one at a time
    FOR each feature on the map:
        IF it's a point:   skip it (a dot has no area to print).
        IF it's a line:    give it some width (--line-width) so it
                           becomes a skinny strip you can print.
        IF --grow is set:  fatten it by that many mm on every side,
                           so features too thin for the nozzle can print.
        Drop any leftover crumbs smaller than the printer can make.

        Now the two modes split:

        ---- RAISED --------------------------------------------------
        Pull the shape straight up into a solid wall (like a cookie
        cutter pushed through clay) and set it on top of the base.

        ---- CARVED --------------------------------------------------
        Don't build it yet. Just set it aside in a pile - these
        shapes will become the holes later.

STEP 6 - Build the finished model

    ---- RAISED ------------------------------------------------------
    Make a flat rectangle a little bigger than the map (--margin)
    and --base mm thick.
    The raised features from Step 5 are already sitting on top of it.
    (Use --no-base to skip the rectangle and print features alone.)

        Side view:     ##    ###      ##        <- features stick up
                    ==========================  <- base plate

    ---- CARVED ------------------------------------------------------
    Make a flat rectangle --base mm thick. This is the FLOOR, and it's
    what shows at the bottom of each channel.
    Make a second rectangle the same size.
    Cut every shape from the pile out of it, like a cookie cutter
    punching holes.
    Pull that hole-punched sheet up --height mm and stack it on the floor.

        Side view:  ####  ######  ####  ####    <- block with channels cut out
                    ==========================  <- floor (bottom of channels)

    Tip: pause the print at the top of the floor and switch filament
    colors, and the channels will print in a different color from the land.

STEP 7 - Save
    Glue all the pieces into one model and write the .stl file.
    Print the final size so you can check it fits your printer bed.
"""

import argparse
import sys

import geopandas as gpd
import numpy as np
import trimesh
import mapbox_earcut as earcut
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate, scale
from shapely.geometry.polygon import orient


def extrude(poly, height):
    """Pull a flat 2D shape straight up into a solid 3D block.

    In plain English: draw the shape on the floor, draw the same shape
    `height` mm up in the air, then connect the two with walls all the way
    around (including around any holes). The result is sealed with no gaps,
    which is what slicers need.

    (Does its own triangulation so it works across trimesh/mapbox_earcut versions.)
    """
    # Make sure the outline runs counter-clockwise and holes run clockwise.
    poly = orient(poly, sign=1.0)
    rings = [np.asarray(poly.exterior.coords)[:-1, :2]]
    rings += [np.asarray(r.coords)[:-1, :2] for r in poly.interiors]
    verts2d = np.vstack(rings).astype(np.float64)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)

    # Chop the flat shape into triangles (3D files are made entirely of triangles).
    tri = np.asarray(earcut.triangulate_float64(verts2d, ends), dtype=np.int64).reshape(-1, 3)

    # One copy of every corner on the floor, one copy up at `height`.
    n = len(verts2d)
    bottom = np.column_stack([verts2d, np.zeros(n)])
    top = np.column_stack([verts2d, np.full(n, height)])
    vertices = np.vstack([bottom, top])

    # Bottom face points down, top face points up.
    faces = [tri[:, ::-1], tri + n]

    # Walls: connect each edge on the floor to the matching edge up top.
    start = 0
    for r in rings:
        m = len(r)
        i = np.arange(start, start + m)
        j = np.roll(i, -1)
        faces.append(np.column_stack([i, j, j + n]))
        faces.append(np.column_stack([i, j + n, i + n]))
        start += m

    mesh = trimesh.Trimesh(vertices, np.vstack(faces), process=True)
    mesh.fix_normals()
    return mesh


def to_polygons(geom):
    """Break any shape (or group of shapes) into a simple list of single polygons."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):  # GeometryCollection
        out = []
        for g in geom.geoms:
            out.extend(to_polygons(g))
        return out
    return []


def main():
    # ---- Settings you can change from the command line ----
    p = argparse.ArgumentParser(description="Convert a shapefile to an STL for 3D printing.")
    p.add_argument("input", help="input .shp")
    p.add_argument("output", help="output .stl")
    p.add_argument("--width", type=float, default=150,
                   help="longest side of the printed model in mm (default 150)")
    p.add_argument("--height", type=float, default=3,
                   help="RAISED: how tall features stand, mm. CARVED: how deep channels go, mm (default 3)")
    p.add_argument("--height-field", help="RAISED only: attribute column to drive height")
    p.add_argument("--min-height", type=float, default=1,
                   help="height for the lowest attribute value, mm (default 1)")
    p.add_argument("--max-height", type=float, default=10,
                   help="height for the highest attribute value, mm (default 10)")
    p.add_argument("--line-width", type=float, default=1.0,
                   help="printed width of line features in mm (default 1.0)")
    p.add_argument("--base", type=float, default=2,
                   help="RAISED: base plate thickness. CARVED: floor thickness under channels. mm (default 2)")
    p.add_argument("--margin", type=float, default=5,
                   help="extra border around the map, mm (default 5)")
    p.add_argument("--no-base", action="store_true", help="RAISED only: skip the base plate")
    p.add_argument("--grow", type=float, default=0,
                   help="thicken features by this many mm on each side so thin ones can print")
    p.add_argument("--carve", action="store_true",
                   help="CARVED mode: cut features INTO a block as channels instead of raising them")
    args = p.parse_args()

    print("Mode:", "CARVED (channels cut into a block)" if args.carve
          else "RAISED (features standing on a base)")

    # ---- STEP 1: Read the map ----
    gdf = gpd.read_file(args.input)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        sys.exit("No geometry found in the shapefile.")

    # ---- STEP 2: Get everything into real distances (meters, not degrees) ----
    if gdf.crs is None:
        print("Warning: no CRS found, using coordinates as-is.")
    elif gdf.crs.is_geographic:
        utm = gdf.estimate_utm_crs()
        print(f"Reprojecting from {gdf.crs.to_string()} to {utm.to_string()}")
        gdf = gdf.to_crs(utm)

    # ---- STEP 3: Shrink the map down to printer size ----
    minx, miny, maxx, maxy = gdf.total_bounds
    span = max(maxx - minx, maxy - miny)
    if span == 0:
        sys.exit("Features have zero extent - nothing to scale.")
    k = args.width / span  # how many mm on the print per map unit
    print(f"Scale: 1 mm on the print = {1 / k:,.1f} map units")

    def to_mm(geom):
        # Slide to (0,0), then shrink to printer size.
        g = translate(geom, -minx, -miny)
        return scale(g, xfact=k, yfact=k, origin=(0, 0))

    # ---- STEP 4: Decide how tall each feature is ----
    if args.height_field:
        if args.height_field not in gdf.columns:
            sys.exit(f"Field '{args.height_field}' not found. Columns: {list(gdf.columns)}")
        vals = gdf[args.height_field].astype(float).to_numpy()
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if hi == lo:
            heights = np.full(len(vals), args.max_height)
        else:
            heights = args.min_height + (vals - lo) / (hi - lo) * (args.max_height - args.min_height)
        heights = np.nan_to_num(heights, nan=args.min_height)
    else:
        heights = np.full(len(gdf), args.height)

    # RAISED features sit on top of the base plate; CARVED builds its own floor later.
    z0 = 0 if (args.no_base or args.carve) else args.base
    meshes = []   # finished 3D pieces
    shapes = []   # CARVED only: the pile of shapes that will become holes
    skipped = 0

    # ---- STEP 5: Clean up each feature, one at a time ----
    for geom, h in zip(gdf.geometry, heights):
        g = to_mm(geom)
        gtype = g.geom_type

        if "Point" in gtype:
            skipped += 1          # a dot has no area to print
            continue
        if "Line" in gtype:
            g = g.buffer(args.line_width / 2, cap_style=1, join_style=1)  # give lines width

        # Fatten thin features (--grow), or just tidy up messy outlines.
        g = g.buffer(args.grow, join_style=1) if args.grow > 0 else g.buffer(0)

        for poly in to_polygons(g):
            if poly.area < 0.01:  # crumbs too small for any printer
                continue

            if args.carve:
                # CARVED: set aside for now - becomes a hole in Step 6.
                shapes.append(poly)
            else:
                # RAISED: pull it straight up and set it on the base.
                m = extrude(poly, float(h))
                m.apply_translation([0, 0, z0])
                meshes.append(m)

    if skipped:
        print(f"Skipped {skipped} point feature(s).")

    # ---- STEP 6: Build the finished model ----
    w, d = (maxx - minx) * k, (maxy - miny) * k
    plate = box(-args.margin, -args.margin, w + args.margin, d + args.margin)

    if args.carve:
        # CARVED: floor on the bottom...
        from shapely.ops import unary_union
        floor = max(args.base, 0.6)
        meshes.append(extrude(plate, floor))

        # ...then a sheet with every feature punched out of it, stacked on top.
        top = plate.difference(unary_union(shapes)).buffer(0)
        for poly in to_polygons(top):
            m = extrude(poly, args.height + 0.02)
            m.apply_translation([0, 0, floor - 0.02])  # tiny overlap so the slicer fuses them
            meshes.append(m)

    else:
        # RAISED: features are already built; just add the base plate under them.
        if not meshes:
            sys.exit("Nothing to extrude.")
        if not args.no_base:
            meshes.append(extrude(plate, args.base))

    # ---- STEP 7: Save ----
    model = trimesh.util.concatenate(meshes)
    model.export(args.output)

    ex = model.extents
    print(f"Wrote {args.output}: {ex[0]:.1f} x {ex[1]:.1f} x {ex[2]:.1f} mm, "
          f"{len(model.faces):,} triangles")


if __name__ == "__main__":
    main()
