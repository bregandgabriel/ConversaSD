-------------------------------------------------------------------------------------------
 ▄▀▄▄▄▄   ▄▀▀▀▀▄   ▄▀▀▄ ▀▄  ▄▀▀▄ ▄▀▀▄  ▄▀▀█▄▄▄▄  ▄▀▀▄▀▀▀▄  ▄▀▀▀▀▄  ▄▀▀█▄   ▄▀▀▀▀▄  ▄▀▀█▄▄  
█ █    ▌ █      █ █  █ █ █ █   █    █ ▐  ▄▀   ▐ █   █   █ █ █   ▐ ▐ ▄▀ ▀▄ █ █   ▐ █ ▄▀   █ 
▐ █      █      █ ▐  █  ▀█ ▐  █    █    █▄▄▄▄▄  ▐  █▀▀█▀     ▀▄     █▄▄▄█    ▀▄   ▐ █    █ 
  █      ▀▄    ▄▀   █   █     █   ▄▀    █    ▌   ▄▀    █  ▀▄   █   ▄▀   █ ▀▄   █    █    █ 
 ▄▀▄▄▄▄▀   ▀▀▀▀   ▄▀   █       ▀▄▀     ▄▀▄▄▄▄   █     █    █▀▀▀   █   ▄▀   █▀▀▀    ▄▀▄▄▄▄▀ 
█     ▐           █    ▐               █    ▐   ▐     ▐    ▐      ▐   ▐    ▐      █     ▐  
▐                 ▐                    ▐                                          ▐        
-------------------------------------------------------------------------------------------              
# ConversaSD
> Chatbot développé dans le cadre du BUT Science des Données (SAÉ 6.EMS.01 - Modélisation statistique pour les données complexes et le Big Data)

> **Encadrant :** M. Adrien Guille  
## Informations générales

- **Créé par :**
  - Gabriel BREGAND – [gabriel.bregand@gmail.com](mailto:gabriel.bregand@gmail.com)
  - Zouhour ABASSI – [Zouhour.Kamtaoui@gmail.com](mailto:Zouhour.Kamtaoui@gmail.com)
  - Cordelia GUY – [guy.cordelia@gmail.com](mailto:guy.cordelia@gmail.com)
  - Najoua AJOUIRJA – [najoua.ajouirja@gmail.com](mailto:najoua.ajouirja@gmail.com)
- **Date de création :** Juin 2025  
- **Dernière modification (v1) :** Juin 2025  
- **Version :** 1  

## Description du projet


L’objectif principal était de concevoir un chatbot (IA) capable de répondre à des questions posées en francer, en s’appuyant sur un système de RAG (Retrieval-Augmented Generation) alimenté par les cours des deux premières années du BUT SD.

Ce projet a permis de mettre en œuvre des compétences en :
- Traitement automatique du langage naturel (NLP)
- Recherche d’information
- Intégration d'un RAG à une IA générative
- Travail collaboratif et gestion de projet

## Technologies utilisées

- Python 3.10
- Vectorisation de fichier
- Bibliothèques : ( `faiss`, `fitz`, `transformers`, etc.)
- LangChain / LLM (Mistral-7B-Instruct-v0.3)

## Structure du projet

conversaSD/
conversaSD/
│

├── **logs/ # Fichiers de log générés lors de l'exécution**

├── **scr/ # Scripts principaux du chatbot**

│ ├── data_zip/

│ │ └── Supports programmation.zip *# Corpus pédagogique utilisé pour l'entraînement*

│ ├── ConversaSD_chatbot.py *# Le chatbot*

│ ├── ConversaSD_clearfil.py *# Nettoyage et préparation des fichiers*

│ ├── ConversaSD_log.py *# Gestion des logs*

│ ├── ConversaSD_main.py *# Script de lancement principal*

│ ├── ConversaSD_parametre.py *# Paramètres et configuration*

│ └── ConversaSD_rag.py *# Composant RAG (vectorisation + recherc*
he)


├── **doc/ # Documentation du projet**

## Parametrage 

| Abrégé | Nom long       | Type | Valeur par défaut                      | Description                                                                 |
|--------|----------------|------|----------------------------------------|-----------------------------------------------------------------------------|
| -ll    | --log-level    | str  | "debug"                                | Niveau de journalisation à adopter. Choix possibles : debug, info, warning, error, critical. |
| -lf    | --log-file     | str  | "log_execution.log"                    | Nom du fichier dans lequel les logs seront enregistrés.                    |
| -uz    | --unzipfil     | bool | False                                  | Si True, décompresse automatiquement les fichiers ZIP dans le dossier prévu. |
| -et    | --extractetxt  | bool | True                                   | Si True, extrait et convertit les fichiers PDF, CSV, etc. en texte brut .txt. |
| -cs    | --chunk_size   | int  | parametre.chunk_size_default (256)     | Nombre de tokens par segment de texte (chunk).                             |
| -co    | --chunk_overlap| int  | parametre.chunk_overlap_default (64)   | Chevauchement en tokens entre deux segments consécutifs.                   |
| -tk    | --topk         | int  | parametre.topk_default (1)             | Nombre de segments (chunks) les plus pertinents à utiliser pour générer la réponse. |
| -mt    | --max_tokens   | int  | parametre.max_tokens_default (128)     | Nombre maximum de tokens autorisés dans une réponse générée par le chatbot. |


## Lancement du projet

### Prérequis

- Python 3.10
- Installer les dépendances :

```bash
C:/...python.exe -m pip install --quiet pytesseract pillow
C:/...python.exe -m pip install llama-index
C:/...python.exe -m pip install pymupdf
C:/...python.exe -m pip install ftfy
C:/...python.exe -m pip install -U llama-index
C:/...python.exe -m pip install -U llama-index-embeddings-huggingface
C:/...python.exe -m pip install nltk
C:/...python.exe -m pip install faiss-cpu --quiet
C:/...python.exe -m pip install sentencepiece
C:/...python.exe -m pip install huggingface_hub --upgrade
C:/...python.exe -m pip install langchain
C:/...python.exe -m pip install spacy
C:/...python.exe -m spacy download fr_core_news_md
C:/...python.exe -m pip install protobuf
C:/...python.exe -m pip install accelerate
```

### Si besoin de changer de token 

```powershell
$env:HF_TOKEN = "token"
```