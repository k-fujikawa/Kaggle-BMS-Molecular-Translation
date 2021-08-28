import copy
from io import BytesIO

import cv2
import cairosvg
import numpy as np
import lxml.etree as et
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import MolDrawOptions
from svg.path import parse_path
from PIL import Image


class MolecularImageGenerator:
    def __init__(
        self,
        size=256,
        render_size=1024,
    ):
        self.size = size
        self.render_size = render_size

    def __call__(self, inchi: str):
        try:
            mol = Chem.inchi.MolFromInchi(inchi, treatWarningAsError=True)
        except Exception:
            return None, None, None, None
        # ノイズ無し訓練用画像作成
        options = MolDrawOptions()
        options.minFontSize = 50
        options.maxFontSize = 50
        svg = self.inchi2svg(mol, options)
        svg_text = et.tostring(svg)
        image = self.svg2image(svg)

        # Resize
        image = cv2.resize(image, (self.size, self.size))

        # Trim whitespaces
        min_x, min_y, max_x, max_y = self.calc_object_area(image)
        image = image[min_y:max_y, min_x:max_x]

        return image, svg_text

    def inchi2svg(self, mol, options=None):
        mol = copy.deepcopy(mol)
        dm = Chem.Draw.PrepareMolForDrawing(mol)
        d2d = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(self.render_size, self.render_size)
        if options is not None:
            d2d.SetDrawOptions(options)
        d2d.DrawMolecule(dm)
        d2d.FinishDrawing()
        svg_str = d2d.GetDrawingText()
        svg = et.fromstring(svg_str.encode('iso-8859-1'))
        return svg

    def svg2image(self, svg):
        svg_str = et.tostring(svg)
        png = cairosvg.svg2png(bytestring=svg_str)
        image = np.array(Image.open(BytesIO(png)), dtype=np.float32)
        image[:, :, :] = image.mean(axis=-1, keepdims=True)
        image = np.where(image > 254, 255, 0)
        image = np.uint8(image)
        return image

    def calc_object_area(self, image):
        height, width, _ = image.shape
        row_indices, col_indices, _ = list(np.where(image < 255))
        min_x, max_x = col_indices.min(), col_indices.max()
        min_y, max_y = row_indices.min(), row_indices.max()
        margin = min(min_x, min_y, width - max_x, height - max_y)
        if min_x > min_y:
            min_x, max_x = min_x - margin, max_x + margin
            min_y, max_y = 0, height
        else:
            min_y, max_y = min_y - margin, max_y + margin
            min_x, max_x = 0, width

        return min_x, min_y, max_x, max_y
