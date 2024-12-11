from sentence_transformers import SentenceTransformer

# Quick script for checking if Sentence Transformers works the way it should. :)

original = "This is an example document. It has multiple sentences. And also many many words."

rewrite = "Here is another piece of text. This one is slightly different than the previous one, but also has many sentences and lots of words. Let's see how similar they are!"

model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code='True')
original_embedding = model.encode([original])
rewrite_embeddings = model.encode([rewrite])
print(model.similarity(original_embedding, rewrite_embeddings))


model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code='True')
original_embedding = model.encode([original])
rewrite_embeddings = model.encode([rewrite])
print(model.similarity(original_embedding, rewrite_embeddings))
