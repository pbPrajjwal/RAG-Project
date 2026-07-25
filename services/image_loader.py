from langchain_core.documents import Document
from PIL import Image
import pytesseract



def load_images(images):

    documents = []


    for image in images:


        img = Image.open(image)


        extracted_text = pytesseract.image_to_string(
            img
        )


        documents.append(

            Document(

                page_content=extracted_text,


                metadata={

                    "source": image.name,

                    "type": "image"

                }

            )

        )


    return documents