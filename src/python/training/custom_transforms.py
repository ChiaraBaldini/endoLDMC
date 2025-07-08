"""Custom transforms to load non-imaging data."""
from typing import Optional

from monai.config import KeysCollection
from monai.data.image_reader import ImageReader
from monai.transforms.transform import MapTransform, Transform
from transformers import CLIPTokenizer


# Single-item transform: applies the CLIP tokenizer to a text string
class ApplyTokenizer(Transform):
    """Transformation to apply the CLIP tokenizer."""

    def __init__(self) -> None:
        # Load the CLIP tokenizer from the pretrained model
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer"
        )

    def __call__(self, text_input: str):
        # Tokenize the input text, truncating to the model's max length
        tokenized_sentence = self.tokenizer(
            text_input,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # Return only the input IDs tensor
        return tokenized_sentence.input_ids


# Dictionary-based transform: applies the tokenizer to specified keys in a data dict
class ApplyTokenizerd(MapTransform):
    """
    Dictionary transform to apply the CLIP tokenizer to one or more keys in a sample.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        # Initialize the parent MapTransform with the keys to process
        super().__init__(keys, allow_missing_keys)
        # Create an instance of the single-item tokenizer transform
        self._padding = ApplyTokenizer(*args, **kwargs)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        # Make a copy of the input dictionary
        d = dict(data)
        # Iterate over the specified keys and apply the tokenizer
        for key in self.key_iterator(d):
            data = self._padding(d[key])
            d[key] = data

        # Return the updated dictionary
        return d
