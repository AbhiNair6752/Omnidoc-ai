from PIL import Image
from pathlib import Path

class ImagePreprocessor:

    def __init__(self, max_dimension=1600):
        self.max_dimension = max_dimension

    def preprocess(self, image_path: str) -> str:

        image_path = Path(image_path)

        image = Image.open(image_path)

        original_width, original_height = image.size

        print(
            f"Original image size is:"
            f"{original_width} x {original_height}"
        )

        if image.mode != "RGB":
            image = image.convert("RGB")


        largest_dimension = max(
            original_width, original_height
        )

        if largest_dimension > self.max_dimension:

            scale = (
                self.max_dimension /
                largest_dimension
            )

            new_width = int(
                original_width * scale
            )

            new_height = int(
                original_height * scale
            )

            image = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )

            print(
                f"resized image:"
                f"{new_height} x {new_width}"
            )

        else:

            print(
                "Image size is already suitable"
            )

        output_path = (
            image_path.parent /
            f"optimized_{image_path.name}"
        )

        image.save(
            output_path,
            quality=85,
            optimize=True
        )

        print(
            f"Optimized image saved {output_path}"
        )

        return str(output_path)