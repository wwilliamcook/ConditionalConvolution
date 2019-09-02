import string
import numpy as np


def DynamicTextToIndices(vocab_size):
    """Returns a text-to-indices function that dynamically builds a vocabulary of the given size as needed.
    """
    vocab = []  # Dynamically build vocabulary
    non_vocab = []
    counts = np.array([], dtype=np.uint64)
    non_counts = np.array([], dtype=np.uint64)

    def get_char_type(c):
        if c in string.ascii_letters:
            return 'letter'
        elif c in string.digits:
            return 'digit'
        elif c in string.punctuation:
            return 'punctuation'
        elif c in string.whitespace:
            return 'whitespace'
        else:
            return 'other'
    
    def tokenize(s):
        # Split tokens on changing character type (letter, digit, punctuation, etc.)
        tokens = []
        new_char_type = get_char_type(s[0])
        last_split_index = 0
        for i in range(len(s)):
            last_char_type = new_char_type
            new_char_type = get_char_type(s[i])
            if new_char_type != last_char_type:
                tokens.append(s[last_split_index:i])
                last_split_index = i
        tokens.append(s[last_split_index:])
        
        return tokens
    
    def get_token_index(token, training):
        nonlocal vocab, counts, non_vocab, non_counts
        try:  # Get token index in vocab
            index = vocab.index(token)
            counts[index] += 1

            return index
        except ValueError:
            if len(vocab) < vocab_size:  # Add token to vocab
                vocab.append(token)
                counts = np.concatenate([counts, [1]])

                return len(vocab) - 1
            elif training:  # Try to swap this token with a less-used vocab token
                # Get index of token in non_vocab
                try:
                    non_index = non_vocab.index(token)
                    non_counts[non_index] += 1
                except ValueError:  # Add token to non_vocab
                    non_index = len(non_vocab)
                    non_vocab.append(token)
                    non_counts = np.concatenate([non_counts, [1]])
                # Get index of least-used vocab token
                min_index = np.argmin(counts)
                # Swap tokens into and out of vocab if neccessary
                if non_counts[non_index] > counts[min_index]:
                    # Replace the under-used vocab term with the over-used unknown term
                    vocab[min_index], non_vocab[non_index] = non_vocab[non_index], vocab[min_index]
                    counts[min_index], non_counts[non_index] = non_counts[non_index], counts[min_index]
                else:
                    return vocab_size  # Unknown token
            else:
                return vocab_size  # Unknown token
    
    def text_to_indices(s, training=False):
        return np.uint32([get_token_index(token, training) for token in tokenize(s)])
    
    return text_to_indices