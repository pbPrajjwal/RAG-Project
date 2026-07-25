from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdfs(pdf_files):

    documents = []


    for pdf in pdf_files:

        reader = PdfReader(pdf)


        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()


            if text:


                documents.append(
                    Document(

                        page_content=text,


                        metadata={

                            "source": pdf.name,

                            "type": "pdf",

                            "page": page_number
                        }

                    )
                )


    return documents