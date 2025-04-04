from pathlib import Path

import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import numpy as np
# pytorch=1.7.1
# pip install opencv-python


class RawImageExtractorCV2:
    def __init__(self, centercrop=False, size=224):
        self.centercrop = centercrop
        self.size = size
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose(
            [
                Resize([n_px, n_px], interpolation=Image.BICUBIC),
                # CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
                # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.224)),
            ],
        )

    def image_to_tensor(self, image_file, preprocess):
        image = Image.open(image_file).convert("RGB")
        image_data = preprocess(image)

        return {"image": image_data}

    def load_image(self, data):
        # Eğer veri bir PIL Image nesnesiyse
        if isinstance(data, Image.Image):
            # PIL Image işlemleri yapılabilir
            return self.get_image_data(data)
        # Eğer veri bir dosya yoluysa
        
        elif isinstance(data, (str, Path)):
            # Dosyayı yükleyin
            return self.get_image_data_path(data)
        elif isinstance(data, torch.Tensor):  # Eğer Tensor ise
            return self.get_image_data_tensor(data)
        else:
            print("*"*25)
            print(type(data))
            print(data.shape)
            print("*"*25)
            raise ValueError("Geçersiz veri tipi")

    def get_image_data_path(self, image_path):
        image_input = self.image_to_tensor(image_path, self.transform)
        return image_input

    def get_image_data(self, image_loaded):
        image_loaded = image_loaded.convert("RGB")
        image_loaded = self.transform(image_loaded)

        return {"image": image_loaded}
    
    def get_image_data_tensor(self, tensor):
        numpy_img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        image = Image.fromarray(numpy_img)

        # PIL görüntüsüne çevir
        return self.get_image_data(image)

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(
            -1,
            1,
            tensor_size[-3],
            tensor_size[-2],
            tensor_size[-1],
        )
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data


# An ordinary video frame extractor based CV2
RawImageExtractor = RawImageExtractorCV2
