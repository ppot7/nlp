from urllib.request import urlopen
from collections import Counter
import numpy as np
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

# Converts corpus text to list of words, optionally allowing for removal of punctuation
# and conversion of capital letters to lower case.
def create_corpus_list(text: str, remove_punctuation: bool = True, convert_caps: bool = True) -> list[str]:
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    if convert_caps:
        text = text.lower()

    return text.split()

# Utility for creating vocabulary dictionary for corpus text. The basic functionality
# returns a dictionary entry for every unique word in the corpus. This dictionary is
# defined as follows:
#   keyed_by_word: dict = { 
#       [unique word]: 
#           {"idx": [index value (int) for retrieval], "freq": [# of instances in corpus]}
#   }
# 
#  keyed by_index [dict] = provides same information as keyed_by_word but indexes on "idx" 
#   keyed_by_idx: dict = { 
#       [unique index value (int)]: 
#           {"word": [string representing unique value in corpus], "freq": [# of instances in corpus]}
#   }
#
#   NOTE ON crop_pct
#       In many instances common words such as "the", "a", "and", etc. add little to the context of a 
#       corpus list. Because of there frequency, however, they may affect the factored results of a 
#       word embedding. (For example the main state variable may refer to a datapoint containing "the")
#       To alleviate this the crop_pct removes the words appearing the most frequently. For example, 
#       a crop_pct will remove the most common words making up 10 percent of the original corpus.
# 
#       crop_pct (float) - remove the most common words that make up the highest percent in terms of
#           frequency. Values can range from zero (0.0 - all words included) to .25 (most common words
#           making up the highest 25 percent in terms of frequency.)
# 
#       common_words(set[str]) - set of common words removed from original corpus.  
def create_vocabulary_index(corpus_list: list[str], crop_pct: float = 0.0) -> tuple[dict[str], dict[int], set[str]]:
    # Cropping more that the highest 25 percent of words in corpus 
    # would likely make any word embedding scheme very limited
    if crop_pct <0.0 or crop_pct > .25 :
        raise ValueError("Crop pct must be between 0.0 and 0.25 to be valid")

    total_words: int = len(corpus_list)
    treshold: int = total_words - round(crop_pct * total_words)

    counter: int = 0
    keyed_by_word: dict = {}
    keyed_by_idx: dict = {}
    common_words: set[str] = set()
    for words in Counter(corpus_list).most_common():
        if total_words > treshold:
            common_words.add(words[0])
        else:
            keyed_by_word[words[0]] = {"idx": counter, "freq": words[1]}
            keyed_by_idx[counter] = {"word": words[0], "freq": words[1]}
            counter = counter + 1
        total_words = total_words - int(words[1])
    
    return keyed_by_word, keyed_by_idx, common_words


# Utility for removing certain words from corpus.
def filter_corpus_list(corpus_list: list[str], common_words: set[str]) -> list[str]:
    return [word for word in corpus_list if word not in common_words]

# Writes a corpus list (ordered list) to file.
def write_corpus_list(corpus_list: list[str], filename: str, overwrite_if_exist: bool = True):
    if overwrite_if_exist:
        mode: str = "wt"
    else:
        mode: str = "+wt"

    with open(filename, mode=mode) as file_writer:
        file_writer.write(",".join(corpus_list))

    return

# Reads a corpus list from file.
def read_corpus_list(filename: str) -> list[str]:
    with open(filename, mode="rt") as file_reader:
        text = file_reader.read()

    return text.split(",")

def create_datasets(corpus: list[str], words_before: int = 3, words_after: int = 2, fixed_arrays: bool = False) -> list[tuple]:
    total_words: int = len(corpus)
    dataset: list[tuple] = []

    if not fixed_arrays:
        for index in range(words_before):
            dataset.append((corpus[index], [corpus[x] for x in range(index+words_after+1) if x != index]))

    for index in range(words_before, total_words-words_after):
        dataset.append((corpus[index], [corpus[x] for x in range(index-words_before, index+words_after+1) if x != index]))

    if not fixed_arrays:
        for index in range(total_words-words_after, total_words):
            dataset.append((corpus[index], [corpus[x] for x in range(index-words_before, total_words) if x != index]))

    return dataset
    
def create_occurrence_matrices(dataset: list[tuple], vocabulary: dict, ignore_nonexistence: bool = False) -> tuple[np.array, np.array]:
    x_array: np.array = np.zeros((len(dataset), len(vocabulary)))
    y_array: np.array = np.zeros((len(dataset), len(vocabulary)))

    try:
        for index, record in enumerate(dataset[:10]):
            y_array[index, vocabulary[record[0]]["idx"]] = 1
            for word in record[1]:
                x_array[index, vocabulary[word]["idx"]] += 1
    except KeyError as key_error:
        if not ignore_nonexistence:
            raise key_error
        
    return y_array, x_array

def main():
    book_path = 'https://www.gutenberg.org/cache/epub/64317/pg64317.txt'

    # text = from_gutenberg(book_path)
    filename: str = "gatsby_corpus.csv"
    # revised_corpus: list[str] = create_corpus_list(text[1455:])

    # write_corpus_list(revised_corpus, filename=filename)
    original_corpus: list[str] = read_corpus_list(filename)

    vocab_by_word, vocab_by_idx, common_words = create_vocabulary_index(original_corpus, .2)

    # filter out common words removed from vocabulary
    revised_corpus: list[str] = filter_corpus_list(original_corpus, common_words=common_words)
    dataset: list[tuple] = create_datasets(revised_corpus)

    y_matrix, x_matrix = create_occurrence_matrices(dataset, vocab_by_word)
    
    cov_matrix: np.array = y_matrix.T @ x_matrix
    
    print(y_matrix.shape)
    print(x_matrix.shape)
    print(cov_matrix.shape)

    u_matrix, sigma, v_trans = np.linalg.svd(cov_matrix)
    
    print(f"U Matrix: {u_matrix.shape}")
    print(f"Sigma: {sigma.shape}")
    print(f"V_transpose: {v_trans.shape}")

    norm_vector = np.diag(sigma)
    norm_vector = norm_vector / np.linalg.norm(norm_vector)

    print(norm_vector.shape)
    for count in range(10):
        print(norm_vector[count, count])

    word_embeddings: np.array = u_matrix[:, :10]
    print(word_embeddings.shape)

    # determine cutoff
    # produce embeddings

if __name__ == "__main__":
    main()