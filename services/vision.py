from transformers import pipeline


captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)



def generate_caption(image):

    result = captioner(image)

    return result[0]["generated_text"]