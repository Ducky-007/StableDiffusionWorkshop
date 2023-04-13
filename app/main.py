import os

import keras_cv.models
from PIL import Image


class Leo:
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    directory = os.path.join("app", "images")
    model = keras_cv.models.StableDiffusionV2(
        img_height=512,
        img_width=768,
        jit_compile=True,
    )

    def __call__(self, name: str, prompt: str, neg_prompt: str, count: int, epochs: int):
        renders = self.model.text_to_image(
            prompt=prompt,
            negative_prompt=neg_prompt,
            batch_size=count,
            num_steps=epochs,
        )
        for idx, render in enumerate(renders, 1):
            filepath = os.path.join(self.directory, f"{name}-{idx}.png")
            Image.fromarray(render).save(filepath)


if __name__ == '__main__':
    text_to_image = Leo()
    text_to_image(
        name="megalodon",
        prompt="Photograph of megalodon, a 60 foot long shark with a dark top, "
               "light under belly with rows of razor sharp teeth.",
        neg_prompt="",
        count=3,
        epochs=10,
    )
