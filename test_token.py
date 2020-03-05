import tokenization

vocab_file="vocab.txt"
tokenizer = tokenization.FullTokenizer(vocab_file)
print(tokenizer.tokenize("xin chào"))
print(tokenizer.tokenize("kính chào"))
print(tokenizer.tokenize("người giàu cũng khóc"))
