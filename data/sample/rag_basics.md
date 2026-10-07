# Retrieval-Augmented Generation (RAG) Basics

RAG combines a retrieval step with a language model. Instead of relying only on what the model memorised during training, the system first retrieves relevant passages from an external knowledge base and gives them to the model as context.

Chunking splits documents into smaller passages before embedding. Chunks that are too large dilute the meaning of the embedding, while chunks that are too small lose context. Overlap between neighbouring chunks helps preserve information that straddles a boundary.

An embedding model converts text into a dense vector so that semantically similar texts end up close together. BGE and GTE are popular open-source embedding model families. BGE query embeddings work best when the query is prefixed with a short instruction.

A vector database such as ChromaDB stores embeddings and answers nearest-neighbour queries using cosine similarity or inner product. Approximate indexes like HNSW make search fast on large collections.

Retrieval quality is commonly measured with hit-rate at k, which is the fraction of questions where a correct passage appears in the top k results, and with mean reciprocal rank, which rewards placing the correct passage higher in the ranking.

Hallucination is reduced by instructing the model to answer only from the retrieved context and to cite its sources, and by refusing to answer when retrieval finds nothing relevant.
