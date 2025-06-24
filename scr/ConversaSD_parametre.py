#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  :  Gabriel bregand gabriel.bregand@.fr ;
# Created Date: June 2025
# Modif Date v1 : June 2025
# version ='1'
# ---------------------------------------------------------------------------
"""
Parametre l'ensemble du programme 

Note
----
Le programme comporte encore des parties en développement,
telles que l'appel à l'API OSM, qui n'a pas encore été implémenté.
"""
# ---------------------------------------------------------------------------

# Import
import os
from llama_index.core import Settings

# Emplacement des fichier python 
emplacement_code: str  = "C:/ConveraSD_kaggle/" # pour Kaggle : /kaggle/

# Emplacement fichier zip
zip_file: str  = emplacement_code+"data_zip/Supports programmation.zip" 
# IMPORTANT Pour Unzip : Config -> --unzipfil = True
# pour Kaggle :"input/zip-supports-programmation/Supports programmation.zip"
# dans le cas ou le fichier ne serai pas deja UnZip

# Emplacement Ficher apres unzip
if emplacement_code == "/kaggle/":
    # /kaggle/input : chemin kaggle des fichier importer
    # zip-supports-programmation : nom données au moment de l'import
    extract_folder: str = "/kaggle/input/"+"zip-supports-programmation"
else : 
    extract_folder: str = emplacement_code+"extract_file/"

# Fichier 'concerne'
# Les fichier contenue dans l'extraction zip
## Solution potentiellement à changer si le nombre de fichier est trop important
## pour ne pas avoir a tous remplire à la main
folders = [
    "Supports programmation/S1",
    "Supports programmation/S2"
] 

# Ajout du chemin 'Ficher concerner apres unzip'
full_paths: list = [os.path.join(extract_folder, folder.strip("/\\").strip()) for folder in folders]


# Emplacement fichier stockage transformation des fichier en txt
if emplacement_code == "/kaggle/":
    # /kaggle/working : chemin kaggle des fichier cree
    output_txt_file: str = "/kaggle/working/"+"data/"
else :
    output_txt_file: str  = emplacement_code+ "data/extracted_data/" 
# IMPORTANT Pour transformation en txt pour traitement : Config -> --extractetxt = True  

# ORC (reconnaissance optique de caractères) open source 
# Si texte contenue dans image d'un pdf extrait les information
#https://sourceforge.net/projects/tesseract-ocr.mirror/
if emplacement_code == "/kaggle/":
    file_pytesseract: str = "tesseract" # 
else :
    file_pytesseract:str = "C:/Program Files/Tesseract-OCR/tesseract.exe"
lang_orc: str ="fra+eng"

# Embedding model
#https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# input text, return list numbers (used to capture the semantics)
embed_model_name: str = 'all-MiniLM-L6-v2'

# Generative model
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
gen_model: str = "mistralai/Mistral-7B-Instruct-v0.3" 

# Taille en token 
chunk_size_default: int = 256
# Config : --chunk_size = x  

# Nombre tokens pouvant etre partages entre 2 chunk consécutifs
chunk_overlap_default: int = 64
# Config : --chunk_overlap = x  

# Max token return par chat bot
max_tokens_default: int = 128
# Config : --max_tokens = x  

# Nombre chunk pertinent 
topk_default: int = 1
# Config : --topk = x



index_path: str = output_txt_file + "/faiss.index"

mapping_path: str = output_txt_file + 'chunks.pkl'
