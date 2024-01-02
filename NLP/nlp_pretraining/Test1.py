from Vocab import  Vocab
token_tmp = ['ww','rr','tt','oo','ww','ww','ww']
vocab = Vocab(token_tmp)
print(list(vocab.token_to_index.items())[:10])
