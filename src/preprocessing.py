

import re


try:
    from tashaphyne.stemming import ArabicLightStemmer
    import pyarabic.araby as araby
    PYARABIC_AVAILABLE = True
    ARABIC_STEMMER_AVAILABLE = True
    
    stemmer = ArabicLightStemmer()

except ImportError:
    PYARABIC_AVAILABLE = False
    ARABIC_STEMMER_AVAILABLE = False
    print("[WARNING] pyarabic not installed. Arabic normalization will be limited.")
    print("          Run: pip install pyarabic")

try:
    from nltk.stem import PorterStemmer
    from nltk.stem.isri import ISRIStemmer
    NLTK_STEMMERS_AVAILABLE = True
    _porter = PorterStemmer()
    _isri   = ISRIStemmer()
except ImportError:
    NLTK_STEMMERS_AVAILABLE = False
    print("[WARNING] nltk stemmers not found. Run: pip install nltk")


ENGLISH_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "not", "no", "nor", "so", "yet",
    "both", "either", "neither", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "as",
    "its", "it", "this", "that", "these", "those", "i", "me",
    "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "his", "her", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "all", "any", "about", "above", "after",
    "before", "between", "into", "through", "during", "up", "down",
    "out", "off", "over", "under", "again", "then", "once", "here",
    "there", "only", "own", "same", "also", "s", "t",
}



ARABIC_STOP_WORDS = {
    "في", "من", "على", "الى", "عن", "مع", "هذا", "هذه", "ذلك",
    "تلك", "التي", "الذي", "الذين", "اللواتي", "وهو", "وهي",
    "فهو", "فهي", "او", "ان", "كان", "كانت", "يكون",
    "تكون", "قد", "لا", "لم", "لن", "ما", "هل", "بل", "بعد",
    "قبل", "حين", "عند", "غير", "حتى", "اذا", "لو", "كل",
    "بعض", "جميع", "اي", "ثم", "ايضا", "كما", "ولا", "وقد",
    "وكان", "فان", "وان", "ولم", "اما", "اذ", "منذ", "حول",
    "خلال", "عبر", "ضد", "نحو", "رغم", "وفق", "طبقا", "الا",
    "فقط", "هو", "هي", "هم", "هن", "انا", "نحن", "انت", "انتم",
    "وهذا", "وهذه", "وذلك", "فقد", "لكن", "لكي", "كي", "مما",
    "حيث", "بين", "دون","و", "تحت", "فوق", "امام", "خلف",
}

# detect_language


def detect_language(text):
    """
    Detect if text is:
    - English
    - Arabic
    - Mixed (returns English by requirement)
    """

    if not text or not text.strip():
        raise ValueError("empty text provided for language detection.")
    
    if re.search(r'[a-zA-Z]', text):
        # if mixed, we default to English as per requirement
        #
        return "english" 
    

    elif re.search(r'[\u0600-\u06FF]', text):
        return "arabic"
    
    raise ValueError("Unable to detect language. Text may be empty or contain unsupported characters.")


# ENGLISH PIPELINE

def tokenize_english(text):

    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    return re.findall(r'[a-z]+', text)


def remove_english_stopwords(tokens):
    
    if not tokens:
        return []
    result = []
    for t in tokens:
        if t not in ENGLISH_STOP_WORDS:
            result.append(t)
    return result
    


def stem_english(word):
    
    if not word:
        return ""
    if len(word) <= 2:
        return word

    if NLTK_STEMMERS_AVAILABLE:
        try:
            return _porter.stem(word.lower())

        except Exception:
            pass

    # fallback لو حصل مشكلة
    w = word.lower()


    return w if w else word   


def preprocess_english(text):

    tokens  = tokenize_english(text)
    tokens  = remove_english_stopwords(tokens)
    stemmed = []

    for t in tokens:
        stemmed.append(stem_english(t))

    
    result = []

    for t in stemmed:
        if t:
            result.append(t) # filter out empty strings  

    return result




# ARABIC PIPELINE

def normalize_arabic (text) :
    
    if not text or not isinstance(text, str):
        return ""
    #re.sub(pattern, replacement, text)
    '''
    The Unicode range [\u064B-\u065F] represents the Arabic diacritics (tashkeel/harakat) and special symbols in the Unicode character set.
    '''
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # harakat and tanween
    
    text = re.sub(r'\u0640', '', text) # tatweel
    
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'لآ', 'لا', text)

    
    return text


def tokenize_arabic(text):

    if not text or not isinstance(text, str):
        return []
    return re.findall(r'[\u0600-\u06FF]+', text) # Arabic letters only (basic range)


def remove_arabic_stopwords(tokens):
    
    if not tokens:
        return []
    return [t for t in tokens if t not in ARABIC_STOP_WORDS]


def stem_arabic(word):

    if not word:
        return ""
    if len(word) <= 3:
        return word

    # ISRI stemmer
    if NLTK_STEMMERS_AVAILABLE and ARABIC_STEMMER_AVAILABLE:
        try:
            
            result =  _isri.stem(word)
            if len(result) <= 3:
                stemmer.light_stem(word)
                return stemmer.get_stem()
            else:
                return result
            
        except Exception:
            pass
        
    
    return word   


def preprocess_arabic(text):
    """
    Full Arabic pipeline: normalize → tokenize → stop-word removal → stemming.
    Returns a list of clean, stemmed tokens.

    This is the single entry point for Arabic text.
    Both documents AND queries must use this function.
    """
    text    = normalize_arabic(text)
    tokens  = tokenize_arabic(text)
    tokens  = remove_arabic_stopwords(tokens)
    stemmed = [stem_arabic(t) for t in tokens]
    return [t for t in stemmed if t]



# UNIFIED ENTRY POINT

def preprocess(text, language):
    
    if not text:
        return []
    
    lang = language.lower().strip() if language else ""

    if lang == "english":
        return preprocess_english(text)
    elif lang == "arabic":
        return preprocess_arabic(text)
    else:
        print(f"[WARNING] Unknown language '{language}'. No preprocessing applied.")
        return []



if __name__ == "__main__":
    print("=" * 55)
    print("Preprocessing Pipeline — Self Test")
    print("=" * 55)

    en_samples = [
        "The Students are running quickly in the university",
        "Climate change affects the global economy greatly",
        "",          # edge: empty
        None,        # edge: None
        "123 456",   # edge: numbers only
        "is the a",  # edge: all stop words
    ]

    print("\n── English Pipeline ──────────────────────────────────")
    for s in en_samples:
        result = preprocess(s, "english")
        print(f"  IN : {repr(s)}")
        print(f"  OUT: {result}")
        print()

    ar_samples = [
        "الطلاب يدرسون في الجامعات المصرية",
        "تغير المناخ يؤثر على الاقتصاد العالمي محمد كان يحب المهارات و المهارة حب ",
        "",          # edge: empty
        None,        # edge: None
        "في على من", # edge: all stop words
    ]

    print("── Arabic Pipeline ───────────────────────────────────")
    for s in ar_samples:
        result = preprocess(s, "arabic")
        print(f"  IN : {repr(s)}")
        print(f"  OUT: {result}")
        print()
