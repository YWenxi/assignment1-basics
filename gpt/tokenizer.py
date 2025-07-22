import json
from typing import Iterable
from functools import lru_cache
import regex as re

from .bpe import PRETOKENIZE_REGEX

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def get_pairs(pretoken: list[bytes]) -> list[tuple[bytes, bytes]]:
    """
    Returns all pairs of bytes in the pretoken.
    """
    return [(pretoken[i], pretoken[i + 1]) for i in range(len(pretoken) - 1)]


class Tokenizer:
    """
    Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes
    text into integer IDs and decodes integer IDs into text.
    """
    
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str]
    ):
        
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens
        
        # add user-specified special tokens to the vocab
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token
            # Sort special tokens by length in descending order to handle overlapping tokens correctly
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.split_pattern = "(" + "|".join(map(re.escape, sorted_special_tokens)) + ")"
        else:
            self.special_tokens = []
            self.split_pattern = None

        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            merge: i for i, merge in enumerate(merges)
        }
        self.vocab_size = len(self.vocab)
        self.vocab_to_id = {v: k for k, v in self.vocab.items()}
        
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        """ Class method that constructs and return a Tokenizer 
        from a serialized vocabulary and list of merges (in the 
        same format that your BPE training code output) and 
        (optionally) a list of special tokens. 
        
        Args:
            vocab_filepath: Path to the serialized vocabulary file, json format.
            merges_filepath: Path to the serialized merges file, txt format.
            special_tokens: List of special tokens to include in the tokenizer.
        
        Returns:
            A Tokenizer instance.
        """
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)
    
    def pretokenize(self, text: str) -> Iterable[list[bytes]]:
        """
        Pretokenizes the text based on GPT-2 regex
        returns dict {pretoken : freq}
        """
        if self.split_pattern:
            text_chunks = re.split(self.split_pattern, text)
        else:
            text_chunks = [text]
        for chunk in text_chunks:
            if chunk in self.special_tokens:
                yield [chunk.encode('utf-8')]
            else:
                pretokens = (
                    match.group(0) 
                    for match in re.finditer(PRETOKENIZE_REGEX, chunk)
                )
                for word in pretokens:
                    yield [bytes([b]) for b in word.encode('utf-8')]

    def _merge_word(self, word: list[bytes]) -> list[int]:
        """
        Merges the word into vocabulary tokens (no merges across pre-token boundaries)
        Apply merges to our pre-tokens in the same order of creation.
        """
        tokens = word[:]
        while True:
            pairs = get_pairs(tokens)
            merge_candidates = [pair for pair in pairs if pair in self.merge_ranks]
            if not merge_candidates:
                break
            best_pair = min(merge_candidates, key=lambda pair: self.merge_ranks[pair])
            new_token = best_pair[0] + best_pair[1]
            new_tokens = []
            index = 0
            while index < len(tokens):
                if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == best_pair:
                    new_tokens.append(new_token)
                    index += 2
                else:
                    new_tokens.append(tokens[index])
                    index += 1
            tokens = new_tokens
        return [self.vocab_to_id[token] for token in tokens]
        
    def encode(self, text: str) -> list[int]:
        """
        Steps:
            1. pretokenize the text and represent each pre-token as a sequence of utf-8 bytes
            2. merges these bytes into vocabulary tokens (no merges across pre-token boundaries)
        
        Returns:
            A list of integer IDs corresponding to the vocabulary tokens in the text.
        """
        word_iter = self.pretokenize(text)
        
        return sum((self._merge_word(word) for word in word_iter), [])
    
    def encode_iterable(self, iterable: Iterable[str]) -> list[int]:
        return sum((self.encode(text) for text in iterable), [])
    
    def decode(self, ids: list[int]) -> str:
        """
        Notice:
            special handling for special tokens
        """    
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")