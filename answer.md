## Answers to problems in [cs336_basics](cs336_spring2025_assignment1_basics.pdf)

### Problem (unicode2): Unicode Encodings

1.  What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
    > - UTF-8 is a variable-length encoding that uses:
    >    - 1 byte for ASCII (U+0000 to U+007F), 2–4 bytes for other Unicode characters
	>    - 2–4 bytes for other Unicode characters
    > - UTF-8 leads to smaller input representations in memory.
2. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
    ```python
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
    
    >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
    'hello'
    ```
    > The function is incorrect because it does not handle multi-byte UTF-8 characters correctly.
    > ```python
    > >>> decode_utf8_bytes_to_str_wrong("café".encode("utf-8"))
    > Traceback (most recent call last):
    > File "<stdin>", line 1, in <module>
    > File "<stdin>", line 1, in decode_utf8_bytes_to_str_wrong
    > File "<stdin>", line 1, in <listcomp>
    > UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data
    > ```
3. Give a two byte sequence that does not decode to any Unicode character(s).
    > The two byte sequence that does not decode to any Unicode character(s) is `0xc3 0x28`.
    