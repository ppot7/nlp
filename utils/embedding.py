from collections import Counter
from urllib.request import urlopen
from typing import Tuple
import certifi
import ssl
import re

# downloads a book from Project Gutenberg by its ID as a string
# Much of this code was taken verbatim from langchain-community
# document_loader.Gutenberg object but needed access to urllib
# to be able to alter the certificate chain.
# Gutenberg object alone may throw Exception due to expired
# certificate or out-of-date certificate chain. This is especially
# true developing on MacOS.
def from_gutenberg(file_path: str) -> str:
    # taken directly from Gutenberg.__init__ code
    if not file_path.startswith("https://www.gutenberg.org"):
        raise ValueError("file path must start with 'https://www.gutenberg.org'")

    if not file_path.endswith(".txt"):
        raise ValueError("file path must end with '.txt'")

    """Load a book from Project Gutenberg by its path to .txt file."""
    # This is the part causing self-signed certificate error
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    with urlopen(file_path, context=ssl_context) as response:
        text = "".join([str(elements.decode("utf-8")) for elements in response])

    return text

def create_corpus_list(text: str, remove_punctuation: bool = True, convert_caps: bool = True) -> list[str]:
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    if convert_caps:
        text = text.lower()

    return text.split()

def create_vocabulary_index(corpus_list: list[str], crop_pct: float = 0.0, remove_punctuation: bool = True, 
                            convert_caps: bool = True) -> Tuple[dict[str], dict[int]]:
    # Cropping more that the highest 25 percent of words in corpus 
    # would likely make any word embedding scheme very limited
    if crop_pct <0.0 or crop_pct > .25 :
        raise ValueError("Crop pct must be between 0.0 and 0.25 to be valid")

    total_words: int = len(corpus_list)
    treshold: int = total_words - round(crop_pct * total_words)

    counter: int = 0
    keyed_by_word: dict = {}
    keyed_by_idx: dict = {}
    for words in Counter(corpus_list).most_common():
        if total_words <= treshold:
            keyed_by_word[words[0]] = {"idx": counter, "freq": words[1]}
            keyed_by_idx[counter] = {"word": words[0], "freq": words[1]}
            counter = counter + 1
        total_words = total_words - int(words[1])
    
    return keyed_by_word, keyed_by_idx

def main():
    book_path = 'https://www.gutenberg.org/cache/epub/64317/pg64317.txt'

    text = from_gutenberg(book_path)

    vocab_list = create_vocabulary_index(text)
    print(vocab_list)
    
if __name__ == "__main__":
    main()