import os
import openslide as osl
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image #for resizing images
from matplotlib import pyplot as plt
import shapely
import json
from rasterio.features import rasterize
from openslide.deepzoom import DeepZoomGenerator
import itertools
import geopandas
import time
from typing import List

@dataclass
class Tile:
    image_data: np.ndarray
    mask: np.ndarray
    x: int
    y: int

class ImageProcessor:
  def __init__(self, json_path: str, slide_path: str):
      self.json_path = json_path
      self.slide_path = slide_path
      self.slide = osl.open_slide(slide_path)

  def get_geojson_geometry_new(self) -> list:
    filename = self.json_path
    with open(filename) as f:
      json_list = json.load(f)

    output = []
    for j in json_list['features']: ##modified
      if j['properties']['classification']['name'] == 'UDH-sure':
        output.append(j['geometry']['coordinates'])

    return output

  def get_geojson_geometry_old(self) -> list:
    filename = self.json_path
    if filename == '../prelimary_data/BRACS_1494.geojson':
        with open(filename) as f:
            json_list = json.load(f)
        output = []
        for j in json_list: ##modified
            if j['properties']['classification']['name'] == 'UDH-sure':
                output.append(j['geometry']['coordinates'])
    else:
        print('This function is not supported for this file')
    
    return output

  def generate_masks(self, tile_size:int, rows:int, columns:int)->dict:
    filename = self.json_path
    if filename == '../prelimary_data/BRACS_1494.geojson':
        mask_polygons = list(
            itertools.chain.from_iterable(self.get_geojson_geometry_old())
        )
    else:
        mask_polygons = list(
            itertools.chain.from_iterable(self.get_geojson_geometry_new())
        )
    mask_polygons = [shapely.Polygon(polygon) for polygon in mask_polygons]

    region_polygons = []
    for i in range(0, columns):
      for j in range(0, rows):
        x1, y1, x2, y2 = i*tile_size, (rows-j)*tile_size, (i+1)*tile_size, (rows-j-1)*tile_size
        region_polygons.append(shapely.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]))

    annotations_df = geopandas.GeoDataFrame({"geometry" : mask_polygons})
    tiles_df = geopandas.GeoDataFrame({"geometry" : region_polygons})
    tiles_df['id'] = tiles_df.index
    annotations_df['id'] = annotations_df.index

    intersections_df = geopandas.overlay(tiles_df, annotations_df, how='intersection')

    intersecting_ids = intersections_df['id_1'].unique()

    masks = {}

    for id in intersecting_ids:
      id_intersections_df = intersections_df[intersections_df['id_1'] == id]
      for polygon in id_intersections_df['geometry']:

        bounds = tiles_df[tiles_df['id'] == id]['geometry'][id].bounds
        minx, miny, _, maxy = bounds

        mask = np.zeros((tile_size, tile_size), dtype=np.float32)

        shapes = [(polygon, 1)]
        rasterize(shapes, out=mask, transform=(1, 0, minx, 0, -1, maxy))

        masks[(minx, maxy)] = mask.reshape((1, tile_size, tile_size))

    return masks

  def generate_tile(self, tile_size = 512, masked_tiles_only=False) -> List[Tile]:

    tiles = DeepZoomGenerator(self.slide, tile_size= tile_size, overlap=0, limit_bounds=False)
    last_level = tiles.level_count - 1 #the most zoomed in level
    columns, rows = tiles.level_tiles[last_level]
    info_tile = tiles.get_tile_coordinates(last_level, (0, 0))
    x, y = info_tile[2] #get the size of the tile
    masks = self.generate_masks(tile_size, rows, columns)

    output = []

    for row in tqdm(range(rows)):
      for column in range(columns):

        x1, y1 = column*x, row*y

        if (x1, y1) not in masks.keys() and masked_tiles_only:
          continue

        info_tile = tiles.get_tile_coordinates(last_level, (column, row))
        # Get section of slide for region
        region = tiles.get_tile(last_level, (column, row))
        region = region.convert('RGB')
        region_np = np.array(region)

        region_np_std = region_np.std()
        if region_np_std < 12.5:
          continue

        region_np = region_np.transpose((2, 0, 1))

        mask = masks.get((x1, y1), np.zeros((1, tile_size, tile_size), dtype=np.float32))

        output.append(Tile(region_np, mask, x1, y1))
    return output