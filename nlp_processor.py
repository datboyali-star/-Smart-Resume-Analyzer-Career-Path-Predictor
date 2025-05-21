from typing import Dict, Any
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import spacy
import nltk

class NLPProcessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

        self.stop_words = set(stopwords.words('english'))

    def analyze(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parsed resume data using NLP techniques"""
        raw_text = parsed_data.get('raw_text', '')
        
        analyzed_data = {
            'skills': self._extract_skills(raw_text),
            'key_phrases': self._extract_key_phrases(raw_text),
            'entities': self._extract_entities(raw_text),
            'summary': self._generate_summary(raw_text)
        }

        return analyzed_data

    def _extract_skills(self, text: str) -> list:
        """Extract technical skills and competencies"""
        doc = self.nlp(text)
        skills = []

        # Extract noun phrases as potential skills
        for chunk in doc.noun_chunks:
            if not any(word.is_stop for word in chunk):
                skills.append(chunk.text)

        # Filter and clean skills
        skills = list(set(skills))  # Remove duplicates
        return skills

    def _extract_key_phrases(self, text: str) -> list:
        """Extract important key phrases from the text"""
        doc = self.nlp(text)
        key_phrases = []

        # Extract verb phrases and their associated noun phrases
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                    phrase = ' '.join([child.text for child in token.subtree])
                    key_phrases.append(phrase)

        return key_phrases

    def _extract_entities(self, text: str) -> Dict[str, list]:
        """Extract named entities (organizations, dates, etc.)"""
        doc = self.nlp(text)
        entities = {}

        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

        return entities

    def _generate_summary(self, text: str) -> str:
        """Generate a brief summary of the resume"""
        doc = self.nlp(text)
        sentences = list(doc.sents)

        # Simple extractive summarization
        # Select first sentence and any sentences with important entities
        important_sentences = [sentences[0].text]
        for sent in sentences[1:]:
            if any(ent.label_ in ['ORG', 'DATE', 'SKILL'] for ent in sent.ents):
                important_sentences.append(sent.text)

        # Limit summary to 3 sentences
        summary = ' '.join(important_sentences[:3])
        return summary