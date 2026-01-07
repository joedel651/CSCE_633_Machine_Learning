from hw5_ske import Vocabulary

vocab = Vocabulary(max_size=10)
vocab.add_word('the')
vocab.add_word('movie')
vocab.add_word('was')
vocab.build_vocab()

print("word2idx:", vocab.word2idx)
print()

result = vocab.text_to_indices(['the', 'movie'], max_len=4, model_type='transformer')
print("Result:", result)
print("Expected: [2, 3, 4, 0]")