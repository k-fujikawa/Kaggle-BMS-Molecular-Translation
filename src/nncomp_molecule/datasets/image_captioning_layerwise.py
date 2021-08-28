import nncomp.registry as R
from .image_captioning import ImageCaptioningDataset


def parse_inchi(inchi):
    inchi_layers = inchi.split("/")
    inchi_h = ""
    inchi_sublayers = []
    if len(inchi_layers) > 3:
        if inchi_layers[3].startswith("h"):
            inchi_h = inchi_layers[3]
        else:
            inchi_sublayers.append(inchi_layers[3])
    if len(inchi_layers) > 4:
        inchi_sublayers.extend(inchi_layers[4:])
    return {
        "InChI_atom": inchi_layers[1],
        "InChI_c": inchi_layers[2],
        "InChI_h": inchi_h,
        "InChI_sub": "/".join(inchi_sublayers),
    }


@R.DatasetRegistry.add
class InChILayerWiseImageCaptioningDataset(ImageCaptioningDataset):

    def transform_inchi(self, sample):
        parsed_inchi = parse_inchi(sample["InChI"])
        parsed_inchis = {
            f"{layer}_{k}": v
            for layer in ["atom", "c", "h", "sub"]
            for k, v in self.inchi_transforms(
                parsed_inchi[f"InChI_{layer}"]
            ).items()
        }
        return parsed_inchis


LABEL_MAPPING = dict(
    C=1,
    O=2,
    S=3,
    N=4,
    Br=5,
    F=6,
    Cl=7,
    P=8,
    Si=9,
    B=10,
    I=11,
    H=12,
    UNSPECIFIED=13,
    SINGLE=14,
    DOUBLE=15,
    TRIPLE=16,
    AROMATIC=17,
)
