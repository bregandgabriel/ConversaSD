#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  :  Gabriel BREGAND gabriel.bregand@gmail.com 
#                Zouhour KAMTAOUI Zouhour.Kamtaoui@gmail.com
#                Cordelia GUY guy.cordelia@gmail.com
#                Najoua AJOUIRJA najoua.ajouirja@gmail.com
# Created Date: June 2025
# Modif Date v1 : June 2025
# version ='1'
# ---------------------------------------------------------------------------
"""
Script principal du programme ConversaSD.

Ce script gère l'ensemble du processus d’extraction, d’indexation et de recherche 
dans les documents, avec une interaction finale via un chatbot avec RAG. 

Étapes principales :
- Décompression de fichiers si besoin
- Conversion de documents (.pdf, .csv, etc.) en texte
- Création d’un index vectoriel à partir du corpus
- Lancement d’un chatbot RAG interactif basé sur les chunks les plus pertinents

Parameters
----------
args : 'sys'
    Paramètres saisis par l'utilisateur,
    inclus dans un ensemble de paramètres énumérés
    ci-dessous (main->parser->argument)
    et dans la documentation.

Returns
-------
Aucune valeur retournée directement.
Le script exécute les actions prévues et affiche les réponses via un prompt interactif.

Note
----
Compatible avec les environnements Kaggle et locaux.
Paramètres définis dans le module `ConversaSD_parametre`.

"""
# ---------------------------------------------------------------------------

#import
import logging
import os
import sys
import argparse
import importlib
import nltk
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
from pathlib import Path
import pickle
from huggingface_hub import login
import torch
# link 
from ConversaSD_log import setup_logger
import ConversaSD_parametre as parametre
import ConversaSD_clearfil as clearfil
import ConversaSD_rag as rag
import ConversaSD_chatbot as chatbot

def get_config():
    # Création du parser pour les arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Script de ...")
    
    # Ajout des arguments avec types et options valides pour --log-level
    parser.add_argument("-ll","--log-level", 
                        type=str, 
                        default="debug", 
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Niveau de log à utiliser (debug, info, warning, error, critical)")
    parser.add_argument("-lf","--log-file",
                        type=str,
                        default="log_execution.log",
                        help="Nom du fichier log")
    parser.add_argument("-uz","--unzipfil",
                        type=bool,
                        default=True,
                        help="Extrere du zip contenu dans 'data' les fichier pdf,csv,...")
    parser.add_argument("-et","--extractetxt",
                        type=bool,
                        default=True,
                        help="convertie les fichier pdf,csv,... en txt")
    parser.add_argument("-cs","--chunk_size",
                        type=int,
                        default=parametre.chunk_size_default,
                        help="Nombre de token par chunk")
    parser.add_argument("-co","--chunk_overlap",
                        type=int,
                        default=parametre.chunk_overlap_default,
                        help=" Nombre tokens pouvant etre partages entre 2 chunk consécutifs")
    parser.add_argument("-tk","--topk",
                        type=int,
                        default=parametre.topk_default,
                        help="Nombre de chunk fourni au chatbot")
    parser.add_argument("-mt","--max_tokens",
                        type=int,
                        default=parametre.max_tokens_default,
                        help="Nombre de token par reponse du chatboot")

    # Analyse des arguments
    args, _ = parser.parse_known_args()

    # Retourner la configuration sous forme de dictionnaire
    config = {
        "log_level": args.log_level,
        "log_file": args.log_file,
        "unzipfil": args.unzipfil,
        "extractetxt" : args.extractetxt,
        "chunk_size" : args.chunk_size,
        "chunk_overlap" : args.chunk_overlap,
        "topk": args.topk,
        "max_tokens" : args.max_tokens,
    }

    return config


def run(config, logger):
    logger.info(f"Début du programme parametrage : {config}")
    if config["unzipfil"]:
        try:
            clearfil.unzip(parametre.zip_file, parametre.extract_folder, logger)
            logger.info(f"Extraction du fichier {parametre.zip_file} terminée avec succès.")
        except Exception as e:
            # Gestion des exceptions et affichage de l'erreur
            # SUR KAGGLE unzip auto importer le fichier celon la doc
            if parametre.emplacement_code == "/kaggle/" :
                print("KAGGLE unzip auto")
                pass
            else :
                msg = "Une erreur s'est produite lors de l'extraction du fichier ZIP : {str(e)}"
                logger.error(msg)
                raise FileNotFoundError(msg) 

    if config["extractetxt"]:
        clearfil.converte_fil_to_individual_texts(parametre.output_txt_file, parametre.full_paths, logger)
        
    base_folder = Path(parametre.output_txt_file)

    corpus = []
    
    for file_path in base_folder.glob("*.txt"):
        filename = file_path.stem  # nom fichier sans extension
        parts = filename.split("-", 1)  # split en 2 parties max sur le premier "-"
        
        if len(parts) == 2: #un peut bricole avec la combinaison de code final TODO best si temp
            new_path = f"{parts[0]}/{parts[1]}"
        else:
            new_path = parts[0]
    
        with file_path.open(encoding="utf-8") as f:
            text = f.read()
    
        corpus.append( (new_path, text) )

    rag.build_index(logger=logger,
                    corpus = corpus, 
                    embed_model_name= parametre.embed_model_name,
                    index_path = parametre.index_path,
                    mapping_path = parametre.mapping_path)
    
    #login(new_session=False)
    # Appelle la fonction de login sécurisé
    chatbot.ensure_login(logger=logger)

    idx, chunks, emb_model, tokenizer, model = chatbot.load_resources(parametre.gen_model,
                                                                      parametre.embed_model_name,
                                                                      parametre.index_path,
                                                                      parametre.mapping_path) 


    print("Chatbot RAG prêt. (tapez 'exit' pour quitter)\n")
    logger.info("lancement du chat bot")
    while True:
        q = input("Votre question : ").strip()
        if not q or q.lower() in ("exit", "quit"):
            print("Au revoir !")
            logger.info("exit chatbot")
            break
        logger.info(f"question posé : {q}")
        try:
            resp, used_chunk = chatbot.answer(q, idx, chunks, emb_model, tokenizer, model, config["max_tokens"],config["topk"])
            logger.debug("\n Chunk utilisé :\n")
            logger.debug(used_chunk)
            print(f"\n Réponse : {resp}\n")
        except Exception as e:
            msg = "Error chat bot"
            logger.error(msg)
            raise RuntimeError(msg)

if __name__ == "__main__":
    importlib.reload(parametre) #POUR PERMETRE A KAGGLE DE PAS GARDER EN MEMOIRE UN IMPORT MODIF !
    importlib.reload(chatbot) #POUR PERMETRE A KAGGLE DE PAS GARDER EN MEMOIRE UN IMPORT MODIF !
    importlib.reload(clearfil)
    config = get_config()
    logger = setup_logger(log_level=config["log_level"], log_file=config["log_file"])
    run(config, logger)
