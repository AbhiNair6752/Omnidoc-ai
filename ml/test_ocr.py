from app.ocr.engine import OCREngine
import time

def main():

    image_path = "sample_document.jpg"

    print("1. Starting OCR test....", flush=True)

    start = time.time()

    print("2. Creating OCR engine....", flush=True)

    ocr_engine = OCREngine()

    print(f"3. OCR engine created in {time.time() - start:2f} seconds ", flush=True)

    print("4. Starting OCR Prediction...", flush=True)
    prediction_start = time.time()

    text = ocr_engine.extract_text(image_path)

    print(
        f"5. OCR prediction completed in {time.time() - prediction_start:.2f} seconds",
        flush=True
    )

    print("\n=====Extracted TEXT =======\n")
    print(text)


if __name__=="__main__":
    main()