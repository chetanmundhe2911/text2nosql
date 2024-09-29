import logging

logging.basicConfig(level=logging.INFO)

def store_text(text):
    try:
        document = {'text': text}
        collection.insert_one(document)
        logging.info("Text stored successfully!")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
