import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load the NLTK data needed for the script
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the tokens back into a string
    text = ' '.join(tokens)
    
    return text

def calculate_plagiarism_score(text1, text2):
    # Preprocess the texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    # Convert the texts to spaCy documents
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    # Calculate the cosine similarity between the documents
    similarity = cosine_similarity([doc1.vector], [doc2.vector])[0][0]
    
    # Calculate the plagiarism score
    plagiarism_score = similarity * 100
    
    return plagiarism_score

# Example usage
file1 = "C:\\Users\RAMEEZCOMPUTER\Downloads\Documents\Python Development.pdf"
file2 = "C:\\Users\RAMEEZCOMPUTER\Downloads\Documents\Treasury_Rules_of_FG_Vol_II.pdf"

plagiarism_score = calculate_plagiarism_score(file1, file2)
print("Plagiarism score:", plagiarism_score)