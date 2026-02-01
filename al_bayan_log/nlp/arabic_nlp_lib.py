import re

# Simple rule-based Arabic NLP library

# Prefixes to strip (ordered by length/frequency)
PREFIXES = [
    "وال", "فال", "كال", "بال", "لل",  # Prepositions + Al
    "ال", "و", "ف", "ب", "ك", "ل", "س" # Single letters
]

# Suffixes to strip
SUFFIXES = [
    "ات", "ون", "ين", "ان", "كم", "هم", "نا", "ها", "ه", "ي", "ة"
]

def segment(text):
    """
    Segments text into words and separates prefixes/suffixes.
    Simple rule-based approach.
    """
    words = text.split()
    segmented_words = []
    
    for word in words:
        original = word
        prefix = ""
        suffix = ""
        stem = word
        
        # Check prefixes
        for p in PREFIXES:
            if stem.startswith(p) and len(stem) > len(p) + 1:
                prefix = p
                stem = stem[len(p):]
                break
        
        # Check suffixes
        for s in SUFFIXES:
            if stem.endswith(s) and len(stem) > len(s) + 1:
                suffix = s
                stem = stem[:-len(s)]
                break
                
        segmented_words.append({
            "original": original,
            "prefix": prefix,
            "stem": stem,
            "suffix": suffix
        })
        
    return segmented_words

def pos_tag(text):
    """
    Simple rule-based POS tagging.
    Returns list of (word, tag).
    Tags: NOUN, VERB, PART, ADJ, UNK
    """
    words = text.split()
    tags = []
    
    for word in words:
        tag = "UNK"
        
        # Basic patterns
        if word.startswith("ال"):
            tag = "NOUN"
        elif word.endswith("ة"):
            tag = "NOUN"
        elif word.endswith("ون") or word.endswith("ين"): # Plural
            tag = "NOUN"
        elif word.startswith("ي") or word.startswith("ت") or word.startswith("ن") or word.startswith("أ"): # Present tense signs (approx)
            # Very rough heuristic
            if len(word) > 3:
                 tag = "VERB"
        elif word in ["في", "من", "على", "الى", "عن", "مع", "ب", "ل", "ك", "و", "أو", "ثم"]:
            tag = "PART"
        elif word in ["هذا", "هذه", "ذلك", "تلك", "هؤلاء"]:
            tag = "NOUN" # Demostrative
        elif word in ["هو", "هي", "هم", "نحن", "أنا", "أنت"]:
            tag = "NOUN" # Pronoun
            
        tags.append({"word": word, "tag": tag})
        
    return tags

def extract_entities(text):
    """
    Basic rule-based NER.
    Entities: PERSON, LOCATION, ORG
    """
    words = text.split()
    entities = []
    
    # Very basic lists for demo purposes
    LOCATIONS = ["الرياض", "جدة", "مكة", "المدينة", "القاهرة", "دبي", "عمان", "بيروت", "دمشق", "بغداد", "الكويت", "الدوحة", "المنامة", "مسقط", "صنعاء", "الخرطوم", "طرابلس", "تونس", "الجزائر", "الرباط"]
    PERSON_INDICATORS = ["السيد", "الدكتور", "المهندس", "الشيخ", "الأمير", "الملك"]
    
    i = 0
    while i < len(words):
        word = words[i]
        
        # Check Location
        # Strip punctuation
        clean_word = re.sub(r'[^\w\s]', '', word)
        if clean_word in LOCATIONS:
            entities.append({"text": clean_word, "type": "LOCATION", "start": i, "end": i+1})
            i += 1
            continue
            
        # Check Person (Indicator + Next Word)
        if clean_word in PERSON_INDICATORS and i + 1 < len(words):
            name = words[i+1]
            entities.append({"text": f"{clean_word} {name}", "type": "PERSON", "start": i, "end": i+2})
            i += 2
            continue
            
        i += 1
        
    return entities
