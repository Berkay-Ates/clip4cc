import numpy as np

from clip4cc.rawimage_util import RawImageExtractor
from clip4cc.tokenization_clip import SimpleTokenizer as ClipTokenizer

import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class TextTokenizerClip4CC():
    def __init__(self) -> None:
        self.max_words = 77
        self.image_resolution = 224
        self.tokenizer = ClipTokenizer()

        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

    def extract_raw_sentences(self,sentence: str) -> list[str]:
        tokens = sentence.split()
        return " ".join(tokens)

    def get_text(self, caption):
        k = 1
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        caption = self.extract_raw_sentences(caption)

        words = self.tokenizer.tokenize(caption)

        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words

        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)
        pairs_segment[0] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment
    
class ImagePreprocess():

    def __init__(self) -> None:
        pass

    def get_rawimage(self, image):
        # Pair x L x T x 3 x H x W
        image = np.zeros(
            (
                1,
                3,
                self.rawImageExtractor.size,
                self.rawImageExtractor.size,
            ),
            dtype=np.float32,
        )

        raw_image_data = self.rawImageExtractor.get_image_data(image_path)
        raw_image_data = raw_image_data["image"].reshape(1, 3, 224, 224)

        image[0] = raw_image_data

        return image
    