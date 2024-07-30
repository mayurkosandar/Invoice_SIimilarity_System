import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to extract features from text
def extract_features(text):
    # Extract keywords, invoice number, date, and total amount using regex
    keywords = re.findall(r'\b\w+\b', text.lower())
    invoice_number = re.search(r'invoice\s*#?\s*(\w+)', text, re.IGNORECASE)
    date = re.search(r'date\s*:\s*(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
    total_amount = re.search(r'total\s*:\s*\$?(\d+(?:\.\d{2})?)', text, re.IGNORECASE)
    
    return {
        'keywords': ' '.join(keywords),
        'invoice_number': invoice_number.group(1) if invoice_number else None,
        'date': date.group(1) if date else None,
        'total_amount': float(total_amount.group(1)) if total_amount else None,
    }

# Function to calculate similarity score
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

# Main function to find the most similar invoice
def find_most_similar_invoice(input_pdf_path, invoice_db):
    input_text = extract_text_from_pdf(input_pdf_path)
    input_features = extract_features(input_text)
    
    max_similarity = 0
    most_similar_invoice = None
    
    for invoice in invoice_db:
        similarity = calculate_similarity(input_features['keywords'], invoice['keywords'])
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_invoice = invoice
    
    return most_similar_invoice, max_similarity

# Correct file path
provided_pdf_path = r"C:\DeepLogicAi\documentsimilarity\train\2024.03.15_1145.pdf"
# Extract features from the provided PDF file 
provided_text = extract_text_from_pdf(provided_pdf_path)
provided_features = extract_features(provided_text)

# Sample database of invoices
invoice_db = [
    {
        'keywords': ' '.join(['invoice', '10732', '21347', 'edeka', 'markt', 'marco', 'krawczyk', 'k', 'nördlinger', 'straße', '91126', 'schwabach', 'ficken', 'likör', 'wikinger', 'noorgaard', 'gin', 'berliner', 'luft', 'strong', 'pfefferminzlikör']),
        'invoice_number': '10732',
        'date': '15/03/2024',
        'total_amount': 0,  # Assume total amount is not specified in the text
    },
   
   {
        'keywords': ' '.join(['invoice', '20456', '34567', 'supermarket', 'smith', 'jones', 'ltd', 'high', 'street', '12345', 'newtown', 'apples', 'bananas', 'milk', 'bread', 'cheese']),
        'invoice_number': '20456',
        'date': '22/07/2024',
        'total_amount': 157.85,
    },
    {
        'keywords': ' '.join(['invoice', '30578', '45678', 'bookstore', 'james', 'cameron', 'inc', 'main', 'avenue', '67890', 'metropolis', 'books', 'novels', 'stationery']),
        'invoice_number': '30578',
        'date': '10/08/2024',
        'total_amount': 89.40,
    }
]

# Find the most similar invoice in the database
most_similar_invoice, similarity_score = find_most_similar_invoice(provided_pdf_path, invoice_db)
print("Most Similar Invoice:", most_similar_invoice)
print("Similarity Score:", similarity_score)

