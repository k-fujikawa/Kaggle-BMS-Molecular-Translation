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


class MolecularBBoxGenerator:
    def __init__(
        self,
        size=256,
        render_size=1024,
        drop_bonds_ratio=0.03,
    ):
        self.size = size
        self.render_size = render_size
        self.drop_bonds_ratio = drop_bonds_ratio

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

        # ノイズあり訓練用画像作成
        noised_svg = self.drop_bonds(svg, self.drop_bonds_ratio)
        noised_image = self.svg2image(noised_svg)

        # アノテーション情報抽出用画像作成
        options = MolDrawOptions()
        options.minFontSize = 50
        options.maxFontSize = 50
        for i, atom in enumerate(mol.GetAtoms()):
            options.atomLabels[i] = "x"
        atom_annotated_svg = self.inchi2svg(mol, options=options)
        bbox_annotation_df = pd.concat([
            self.annotate_bboxes(svg),
            self.annotate_bboxes(atom_annotated_svg),
        ], ignore_index=True).drop_duplicates(
            subset=["type", "idx"],
            keep="first",
        ).sort_values(["type", "idx"])
        additional_annotation_df = pd.concat([
            self.annotate_atoms(mol),
            self.annotate_bonds(mol)
        ], ignore_index=True)
        annotation_df = bbox_annotation_df.merge(
            additional_annotation_df,
            how="left"
        )
        annotation_df = pd.concat([
            annotation_df.query("type == 'atom' and label.isna()")
            .fillna(dict(label="H", n_Hs=1)),
            annotation_df.query("type == 'bond' and label.isna()")
            .fillna(dict(label="SINGLE", n_Hs=-1)),
            annotation_df,
        ], ignore_index=False).drop_duplicates(
            subset=["type", "idx"],
            keep="first",
        ).sort_values(["type", "idx"])
        annotation_df = annotation_df.astype(dict(n_Hs=int))

        # Resize
        image = cv2.resize(image, (self.size, self.size))
        noised_image = cv2.resize(noised_image, (self.size, self.size))
        annotation_df[["x_min", "x_max", "y_min", "y_max"]] \
            *= self.size / self.render_size

        # Remove white space
        min_x, min_y, max_x, max_y = self.calc_object_area(image)
        image = image[min_y:max_y, min_x:max_x]
        noised_image = noised_image[min_y:max_y, min_x:max_x]
        annotation_df[["x_min", "x_max"]] = \
            annotation_df[["x_min", "x_max"]] - min_x
        annotation_df[["y_min", "y_max"]] = \
            annotation_df[["y_min", "y_max"]] - min_y

        return image, noised_image, svg_text, annotation_df

    def resize(self, image, annotation_df):
        image = cv2.resize(image, (self.size, self.size))
        reduction_rate = self.size / self.render_size
        annotation_df[["x_min", "x_max", "y_min", "y_max"]] *= reduction_rate
        return image, annotation_df

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

    def drop_bonds(self, svg, drop_bonds_ratio):
        svg = copy.deepcopy(svg)
        bond_elems = svg.xpath(
            r'//svg:path[starts-with(@class,"bond-")]',
            namespaces={'svg': 'http://www.w3.org/2000/svg'}
        )
        mask = np.random.rand(len(bond_elems)) < drop_bonds_ratio
        drop_indices = np.where(mask)[0]
        for idx in drop_indices:
            bond_elem = bond_elems[idx]
            if bond_elem.getparent() is not None:
                bond_elem.getparent().remove(bond_elem)
        return svg

    def annotate_bboxes(self, svg):
        annotations = []
        for child in svg.iterchildren():
            if ("class" not in child.attrib) or (not child.attrib["class"].startswith(("atom-", "bond-"))):
                continue
            path = parse_path(child.attrib["d"])
            obj, index = child.attrib["class"].split("-")
            for element in path:
                for e in  [element.start, element.end]:
                    annotations.append(dict(
                        type=obj,
                        idx=int(index),
                        x=e.real,
                        y=e.imag,
                    ))
        annotations_df = pd.DataFrame(annotations)
        return annotations_df.groupby(["type", "idx"]).agg(
            x_min=("x", "min"),
            y_min=("y", "min"),
            x_max=("x", "max"),
            y_max=("y", "max"),
        ).reset_index()

    def annotate_atoms(self, mol):
        annotations = []
        for idx, atom in enumerate(mol.GetAtoms()):
            annotations.append(dict(
                type="atom",
                idx=idx,
                label=atom.GetSymbol(),
                n_Hs=atom.GetTotalNumHs(),
            ))
        return pd.DataFrame(annotations)

    def annotate_bonds(self, mol):
        annotations = []
        for idx, bond in enumerate(mol.GetBonds()):
            annotations.append(dict(
                type="bond",
                idx=idx,
                label=str(bond.GetBondType()),
                n_Hs=-1,
            ))
        return pd.DataFrame(annotations)

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
