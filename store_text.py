from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['text_db']  # Create or connect to a database
collection = db['texts']  # Create or connect to a collection

def store_text(text):
    # Create a document
    document = {
        'text': text
    }
    # Insert the document into the collection
    collection.insert_one(document)
    print("Text stored successfully!")

def main():
    print("Welcome to the Text to NoSQL Application!")
    while True:
        user_input = input("Enter text to store (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        store_text(user_input)

if __name__ == '__main__':
    main()
