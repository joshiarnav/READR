import re
from typing import List, Dict, Any

# Optional import of spacy - will be handled with try/except

# Load spaCy model for NER
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError) as e:
    # If model not found or spacy not installed, provide instructions
    print(f"SpaCy error: {e}")
    print("Please download the spaCy model with: python -m spacy download en_core_web_sm")
    nlp = None

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text using spaCy.
    
    Args:
        text: Input text
        
    Returns:
        dict: Dictionary of entity types and their values
    """
    if not nlp:
        return {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "WORK_OF_ART": []}
    
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    
    return entities

def extract_literary_elements(text: str) -> Dict[str, List[str]]:
    """Extract potential literary elements from text.
    
    Args:
        text: Input text
        
    Returns:
        dict: Dictionary of literary element types and their values
    """
    # Simple regex patterns for potential literary elements
    patterns = {
        "symbols": r"(?:symbol|symbolize|represents|metaphor|allegory)\s+(?:of|for)?\s+([\w\s]+)",
        "themes": r"(?:theme|motif|concept|idea)\s+(?:of|about)?\s+([\w\s]+)",
        "characters": r"(?:character|protagonist|antagonist|narrator)\s+([A-Z][\w\s]+)"
    }
    
    results = {}
    
    for element_type, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            results[element_type] = list(set([match.strip() for match in matches if len(match.strip()) > 2]))
    
    return results

def format_citations(sources: List[Dict[str, Any]]) -> str:
    """Format sources as citations.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        str: Formatted citations
    """
    if not sources:
        return ""
    
    citations = "\n\n### Sources\n"
    
    for i, source in enumerate(sources):
        title = source.get("title", "Unknown Source")
        url = source.get("url", "")
        source_type = source.get("source", "reference")
        
        if url:
            citations += f"{i+1}. [{title}]({url}) ({source_type})\n"
        else:
            citations += f"{i+1}. {title} ({source_type})\n"
    
    return citations

def detect_book_metadata(text: str) -> Dict[str, str]:
    """Detect book metadata from text.
    
    Args:
        text: Input text
        
    Returns:
        dict: Dictionary of book metadata
    """
    metadata = {}
    
    # Try to detect title
    title_pattern = r"(?:title|book):?\s+([\w\s]+)\n"
    title_match = re.search(title_pattern, text, re.IGNORECASE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    
    # Try to detect author
    author_pattern = r"(?:by|author):?\s+([\w\s\.]+)\n"
    author_match = re.search(author_pattern, text, re.IGNORECASE)
    if author_match:
        metadata["author"] = author_match.group(1).strip()
    
    # Try to detect chapter
    chapter_pattern = r"(?:chapter|section)\s+([\w\s\.]+)\n"
    chapter_match = re.search(chapter_pattern, text, re.IGNORECASE)
    if chapter_match:
        metadata["chapter"] = chapter_match.group(1).strip()
    
    return metadata
