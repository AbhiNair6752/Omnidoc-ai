from paddleocr import PaddleOCR

from app.preprocessing.image_preprocessor import ImagePreprocessor

class OCREngine:

    def __init__(self):

        self.preprocessor = ImagePreprocessor(
            max_dimension=1600
        )


        self.ocr = PaddleOCR(
            lang="en",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )


    def extract_text(self, image_path:str) -> str:

        optimized_image = self.preprocessor.preprocess(
            image_path
        )

        result = self.ocr.predict(optimized_image)

        extracted_text = []

        for page_result in result:
            if "rec_texts" in page_result:
                extracted_text.extend(page_result["rec_texts"])

        return "\n".join(extracted_text)

        """print("\n====RAW OCR RESULT=====\n")

        for page_result in result:
            print(type(page_result))
            print(page_result)

        return result"""