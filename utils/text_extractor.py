import pytesseract
from PIL import Image
import io
import numpy as np
import cv2
class TextExtractor:
    def extract_text_from_image(uploaded_file):
        if uploaded_file is None:
            return ""

        # Read image bytes
        bytes_data = io.BytesIO(uploaded_file.read())
        image = Image.open(bytes_data)

        # Convert to RGB (fix RGBA screenshots)
        image = image.convert("RGB")

        # Convert to OpenCV format
        img = np.array(image)

        # ---- OCR PREPROCESSING ----
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)

        # 3. Remove noise
        gray = cv2.medianBlur(gray, 3)

        # 4. Threshold (important for screenshots)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to PIL
        processed_img = Image.fromarray(thresh)

        # ---- OCR ----
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_img, config=custom_config)

        return text.strip()