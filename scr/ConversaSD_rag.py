"""
Module: index_builder.py
Découpe le texte en chunks, calcule les embeddings et construit un index FAISS.
Usage:
    python index_builder.py <chemin_vers_dossier_extrait>
Exemple:
    python index_builder.py "/kaggle/working/Supports_programmation"
"""
import sys
import pickle
import nltk
import faiss
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
import ConversaSD_clearfil as clearfil 

# Télécharger le tokenizer de NLTK (une seule fois)
nltk.download('punkt', quiet=True)


def chunk_text(text, chunk_size=256, chunk_overlap=64):
    """
    Découpe une liste de textes en chunks à l'aide de spaCy.

    :param texts: liste de chaînes de caractères
    :param chunk_size: taille maximale d'un chunk
    :param chunk_overlap: nombre de tokens de chevauchement entre les chunks
    :return: liste de chunks (chaînes de caractères)
    """
    # Chunking :https://www.pinecone.io/learn/chunking-strategies/
    # fr : https://datacorner.fr/spacy/
    # Charger le modèle spaCy pour le français
    #  #!pip install https://huggingface.co/spacy/fr_core_news_md/resolve/main/fr_core_news_md-any-py3-none-any.whl or idk -m spacy download fr_core_news_md
    text_splitter = SpacyTextSplitter(
        pipeline="fr_core_news_md",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # découper le texte (chunks)
    chunks = text_splitter.split_text(text)
    chunks.extend(chunks) #Si le chunk est trop long prnit taille

    return chunks

"""def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap
    return chunks"""

def build_index(logger,
                corpus,
                embed_model_name: str = 'all-MiniLM-L6-v2',
                index_path: str = 'faiss.index',
                mapping_path: str = 'chunks.pkl',
                chunk_size: int = 256, 
                chunk_overlap: int = 64):
    # 1) charger et nettoyer
    if not corpus:
        msg = "ERROR] Aucun document trouvé."
        logger.error(msg)
        raise FileNotFoundError(msg)
    
    # 2) découpe en chunks
    all_chunks = []
    metadata   = []
    for path, text in corpus:
        for chunk in chunk_text(text,chunk_size,chunk_overlap):
            all_chunks.append(chunk)
            metadata.append(path)
    logger.info(f"Créé {len(all_chunks)} chunks à partir de {len(corpus)} documents.")

    # 3) embeddings
    logger.info("Génération des embeddings avec SentenceTransformer…")
    model = SentenceTransformer(embed_model_name)
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    if embeddings.ndim != 2:
        msg = f"[ERROR] Embeddings inattendus : forme {embeddings.shape}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    dim = embeddings.shape[1]

    # 4) index FAISS
    logger.info(f"Création de l'index FAISS (dimension : {dim})…")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    logger.info(f"Index FAISS sauvegardé dans '{index_path}'.")

    # 5) mapping chunks→fichiers
    with open(mapping_path, 'wb') as f:
        pickle.dump({'chunks': all_chunks, 'meta': metadata}, f)
    logger.info(f"Mapping chunks→fichiers sauvegardé dans '{mapping_path}'.")
    
