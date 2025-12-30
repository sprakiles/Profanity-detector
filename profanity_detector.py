"""
Profanity Detector - catches bad words even when ppl try to hide them lol

made this cuz i was tired of seeing toxic stuff online and wanted to do something about it
the model can detect stuff like:
    - normal bad words (duh)
    - sneaky ones like b@st@rd, sh!t, etc
    - when ppl space it out like "f u c k"
    - leetspeak garbage like h3ll

works pretty well honestly!! got like 93% accuracy which is cool i guess

author: mahmoud
date: dec 2024
"""

import os
import re
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


class ProfanityDetector:
    """
    ok so this class does all the heavy lifting
    
    u can use it to:
        - check if text has bad words
        - get a score of how "bad" the text is (0 to 1)
        - censor the text with #### if its bad
    
    example:
        detector = ProfanityDetector()
        is_bad, confidence = detector.detect("ur an idiot")
        # is_bad = True, confidence = 0.87 or something
    """
    
    # ppl use these chars to try and sneak past filters.. not on my watch lmao
    CHAR_MAP = {
        '@': 'a', '4': 'a', '^': 'a',  # a substitutes
        '8': 'b',                       # b
        '(': 'c', '<': 'c',             # c
        '3': 'e',                       # e - classic leetspeak
        '6': 'g', '9': 'g',             # g
        '#': 'h',                       # h
        '1': 'i', '!': 'i', '|': 'i',   # i - lots of options here
        '0': 'o',                       # o - obv
        '5': 's', '$': 's',             # s - $ is so common
        '7': 't', '+': 't',             # t
    }
    
    def __init__(self, model_path: str = None):
        """
        loads the model from disk
        
        args:
            model_path: where the model files are. leave empty to use default
        """
        if model_path is None:
            # default path is in the model folder next to this file
            model_path = Path(__file__).parent / "model"
        else:
            model_path = Path(model_path)
        
        # load our trained model and the text processor thingy
        try:
            self.preprocessor = joblib.load(model_path / "preprocessor.joblib")
            self.model = joblib.load(model_path / "svm_model.joblib")
        except FileNotFoundError:
            raise Exception(f"bruh model files not found at {model_path}... did u forget to download them?")
    
    def _normalize_text(self, text: str) -> str:
        """
        cleans up the text and handles all the sneaky tricks ppl use
        
        like turning b@st@rd into bastard so we can catch it
        """
        if not isinstance(text, str):
            return ""
        
        # make everything lowercase first
        text = text.lower()
        
        # swap out all the sneaky characters
        for char, replacement in self.CHAR_MAP.items():
            text = text.replace(char, replacement)
        
        # handle stuff like f.u.c.k or f-u-c-k
        text = re.sub(r'(\w)[.\-_*](\w)', r'\1\2', text)
        
        # this part handles spaced out words like "f u c k"
        # kinda complex but basically joins single letters together
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            if len(words[i]) == 1:
                # found a single letter, check if theres more
                single_chars = [words[i]]
                j = i + 1
                while j < len(words) and len(words[j]) == 1:
                    single_chars.append(words[j])
                    j += 1
                
                # if we got 3+ single letters in a row, prob a hidden word
                if len(single_chars) >= 3:
                    result.append(''.join(single_chars))
                    i = j
                else:
                    result.append(words[i])
                    i += 1
            else:
                result.append(words[i])
                i += 1
        
        # also handle repeated chars like fuuuuck -> fuck
        text = ' '.join(result)
        text = re.sub(r'(.)\1+', r'\1', text)
        
        return text
    
    def detect(self, text: str) -> Tuple[bool, float]:
        """
        checks if text is profane
        
        returns a tuple: (is_profane, confidence)
        is_profane is True/False
        confidence is how sure we are (0.0 to 1.0)
        
        example:
            is_bad, conf = detector.detect("hello world")
            # is_bad = False, conf = 0.95 (95% sure its clean)
        """
        # clean up the text first
        normalized = self._normalize_text(text)
        
        # turn text into numbers the model understands
        word_features = self.preprocessor['word_vectorizer'].transform([normalized])
        
        if self.preprocessor.get('char_vectorizer'):
            char_features = self.preprocessor['char_vectorizer'].transform([normalized])
            features = hstack([word_features, char_features])
        else:
            features = word_features
        
        # ask the model what it thinks
        prediction = self.model['model'].predict(features)[0]
        proba = self.model['model'].predict_proba(features)[0]
        
        is_profane = bool(prediction)
        # confidence is probability of whichever class we predicted
        confidence = float(proba[1] if is_profane else proba[0])
        
        return is_profane, confidence
    
    def censor(self, text: str, threshold: float = 0.5) -> str:
        """
        replaces bad text with #### 
        
        if profanity confidence > threshold, whole text becomes #####
        the number of # matches the original text length (including spaces)
        
        args:
            text: what u want to check
            threshold: how confident we need to be to censor (default 50%)
        
        example:
            detector.censor("go to hell", threshold=0.5)
            # returns "##########" (10 chars, same as original)
        """
        is_profane, confidence = self.detect(text)
        
        # only censor if confidence is above threshold
        if is_profane and confidence >= threshold:
            return '#' * len(text)  # same length as original!
        
        return text
    
    def get_profanity_score(self, text: str) -> float:
        """
        just gives u a number from 0 to 1
        higher = more likely to be profane
        
        useful if u want more control over what to do with the text
        """
        normalized = self._normalize_text(text)
        
        word_features = self.preprocessor['word_vectorizer'].transform([normalized])
        
        if self.preprocessor.get('char_vectorizer'):
            char_features = self.preprocessor['char_vectorizer'].transform([normalized])
            features = hstack([word_features, char_features])
        else:
            features = word_features
        
        proba = self.model['model'].predict_proba(features)[0]
        return float(proba[1])  # prob of being profane


# ============================================
# quick functions for lazy ppl (like me lol)
# ============================================

_detector = None  # cached detector so we dont load model every time

def _get_detector():
    """internal func to get/create detector"""
    global _detector
    if _detector is None:
        _detector = ProfanityDetector()
    return _detector

def is_profane(text: str) -> bool:
    """
    quick check - is this text bad?
    returns True or False
    """
    is_prof, _ = _get_detector().detect(text)
    return is_prof

def get_profanity_score(text: str) -> float:
    """
    get the badness score (0 to 1)
    """
    return _get_detector().get_profanity_score(text)

def censor(text: str, threshold: float = 0.5) -> str:
    """
    censor bad text with ####
    
    threshold is how confident we need to be (default 50%)
    u can use 0.6 for 60% if u want less false positives
    """
    return _get_detector().censor(text, threshold)


# ============================================
# test it out!
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("  PROFANITY DETECTOR - testing time!")
    print("=" * 50)
    
    detector = ProfanityDetector()
    
    # some test cases
    test_texts = [
        "Hello, how are you today?",
        "This is bullshit",
        "What the h3ll",
        "f u c k off",
        "Have a great day!",
        "ur such an @sshole",
    ]
    
    print("\nrunning tests...\n")
    
    for text in test_texts:
        is_prof, conf = detector.detect(text)
        censored = detector.censor(text, threshold=0.5)
        
        status = "[BAD]" if is_prof else "[OK]"
        
        print(f"Input:    '{text}'")
        print(f"Result:   {status} (confidence: {conf:.1%})")
        print(f"Censored: '{censored}'")
        print("-" * 40)
    
    print("\ndone! hope it works for u :)")
