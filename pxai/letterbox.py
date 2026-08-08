"""Aspect-preserving square pad. NEW FILE -- pxai/data.py is not modified."""
from PIL import Image
import torchvision.transforms.functional as TF


class LetterboxSquare:
    """Pad the shorter side to make the image square, THEN it can be resized safely.

    `transforms.Resize((S, S))` stretches, which distorts object shape by an amount set
    by the source aspect ratio -- a 4.84x spread across this dataset's 13 native sizes.
    Padding first makes the subsequent resize isotropic, so a circle stays a circle.

    fill defaults to the ImageNet mean in 8-bit, matching the value the normalisation
    maps to zero, so the border is neutral after normalisation rather than a black bar.
    """

    def __init__(self, fill=(124, 116, 104)):
        self.fill = tuple(int(f) for f in fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        left, top = (s - w) // 2, (s - h) // 2
        return TF.pad(img, [left, top, s - w - left, s - h - top], fill=self.fill)

    def __repr__(self):
        return f"{type(self).__name__}(fill={self.fill})"
