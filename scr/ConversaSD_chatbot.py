
import time
import pickle
import faiss
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import login, whoami

# Demande à l'utilisateur de se connecter s'il ne l'est pas
def ensure_login(logger):
    login(new_session=False)  # Demande de login
    # Boucle jusqu'à ce que le login soit effectif
    for _ in range(6):
        try:
            info = whoami()  # Essaie d'obtenir les infos du compte connecté
            logger.info(f"Connecté à Hugging Face en tant que : {info.get('name') or info.get('username')}")
            return
        except Exception as e:
            time.sleep(10)
    msg = """Impossible de se connecter à Hugging Face.
             ou plus de 60s pour remplire le login"""
    logger.error(msg)
    raise RuntimeError(msg)


# ——— 5) Chargement des ressources ———
def load_resources(gen_model,
                   embed_model_name,
                   index_path,
                   mapping_path):
    idx = faiss.read_index(index_path)
    with open(mapping_path,"rb") as f:
        chunks = pickle.load(f)["chunks"]
    emb_model = SentenceTransformer(embed_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        gen_model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        gen_model, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    return idx, chunks, emb_model, tokenizer, model

##
# ——— 6) Recherche du top-1 chunk ———
def retrieve_top(query, idx, chunks, emb_model,topk):
    emb_q = emb_model.encode([query], show_progress_bar=False)
    _, ids = idx.search(emb_q, topk)
    return [chunks[i] for i in ids[0]]

# ——— 7) Construction du prompt & génération ———
def answer(query, idx, chunks, emb_model, tokenizer, model,max_new_tokens,topk):
    snippet = retrieve_top(query, idx, chunks, emb_model,topk)
    snippet = " ".join(snippet)
    preview = snippet.replace("\n", " ")[:200]
    prompt = (
        "Vous êtes un tuteur Python expert. "
        "Répondez **uniquement en français**, en **une seule phrase claire**.\n\n"
        f"Contexte : {preview}\n\n"
        f"Question : {query}\nRéponse :"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,      # beam search déterministe
        num_beams=4,
        early_stopping=True
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("Réponse :")[-1].strip(), snippet
