import pytesseract
from PIL import Image

class TextExtractor:
    def extract_from_image(image_bytes):

        # Open the image using Pillow (PIL)
        try:
            with Image.open(image_bytes) as image:
                
                try:
                    extracted_text = pytesseract.image_to_string(image)
                    return extracted_text
                except pytesseract.NotFoundError:
                    print(f"Error: Tesseract engine not found. Ensure it is installed and the path is correct")
                    return None
                except Exception:
                    print("Error: An error occured during text extraction")

        except FileNotFoundError:
            print("Error: The image file was not found.")
        except:
            print("Error: An unexpected error occured")