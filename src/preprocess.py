
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

_STEMMER = PorterStemmer()
_STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Clean and tokenize raw text:
      1. Lowercase
      2. Remove HTML tags & URLs
      3. Keep only alphanumeric + spaces
      4. Tokenize
      5. Remove stopwords & stem
      6. Rejoin tokens
    """
    text = text.lower()
    # remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # remove URLs
    text = re.sub(r'http\S+', ' ', text)
    # keep letters & numbers
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    cleaned = [
        _STEMMER.stem(tok)
        for tok in tokens
        if tok not in _STOPWORDS and tok.isalpha()
    ]
    return ' '.join(cleaned)