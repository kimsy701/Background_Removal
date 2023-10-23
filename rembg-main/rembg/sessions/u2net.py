import os
from typing import List

import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class U2netSession(BaseSession):
    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                #img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (512, 512)
            ),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.LANCZOS)

        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        #fname = f"{cls.name(*args, **kwargs)}.onnx"
        fname = "/content/drive/MyDrive/rembg-main/rembg/sessions/u2net_bce_itr_10000_train_1.299397_tar_0.162223.onnx"
        '''
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
            "./u2net_bce_itr_10000_train_1.299397_tar_0.162223.onnx",
            None
            if cls.checksum_disabled(*args, **kwargs)
            else "md5:60024c5c889badc19c04ad937298a77b",
            fname=fname,
            path=cls.u2net_home(*args, **kwargs),
            progressbar=True,
        )'''

        #return os.path.join(cls.u2net_home(*args, **kwargs), fname)
        return os.path.join(fname)

    @classmethod
    def name(cls, *args, **kwargs):
        return "u2net"
