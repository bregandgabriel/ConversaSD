
import zipfile
import os
import json
import csv
import unicodedata
import re
from PIL import Image
import io
import pytesseract
import fitz #python3.12.exe -m pip install PyMuPDF
from ftfy import fix_text # latin1 top utf-8 reparation

import ConversaSD_parametre as parametre

# Path to the zip file and the extraction folder

def unzip(zip_file, extract_folder, logger):
    """
    Décompresse le fichier ZIP specifie dans le dossier d'extraction.
    """
    try:
        # Vérifie si le fichier ZIP existe
        if not os.path.exists(zip_file):
            msg = f"Le fichier ZIP {zip_file} n'existe pas."
            logger.error(msg)
            raise FileNotFoundError(msg) 

        # Vérifie si le dossier d'extraction existe, sinon on le crée
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)
            logger.info(f"Le dossier d'extraction {extract_folder} a été créé.")

        # Extraction des fichiers
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
            logger.info(f"Fichiers extraits avec succès dans {extract_folder}.")  
    except zipfile.BadZipFile:
        msg = f"Le fichier {zip_file} n'est pas un fichier ZIP valide."
        logger.error(msg)
        raise FileNotFoundError(msg)  
    except Exception as e:
        msg = f"Une erreur s'est produite lors de l'extraction du fichier ZIP : {str(e)}"
        logger.error(msg)
        raise FileNotFoundError(msg)   


def clean_filename(name):
    """
    Transformation du nom du fichier pour les rendre plus lisible
    """
    # Normalise les accents et caractères Unicode vers ASCII simple
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    # Remplace les caractères non alphanumériques (sauf ._- ) par _
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    return name

def converte_fil_to_individual_texts(output_folder, folders, logger):
    """
    Transformation du nom du fichier pour les rendre plus lisible
    """
    logger.info(f"output_folder : {output_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Dossier créé : {output_folder}")
    else:
        logger.debug(f"Le dossier existe déjà : {output_folder}")
    logger.debug(os.makedirs(output_folder, exist_ok=True))
    readers = {}
    logger.info("Début de la conversion vers fichiers texte individuels")

    # S'assurer que le dossier de sortie existe
    os.makedirs(output_folder, exist_ok=True)

    # Lister les fichiers déjà créés (insensible à la casse)
    existing_files_lower = {f.lower() for f in os.listdir(output_folder)}

    for folder in folders:
        last_folder_name = os.path.basename(folder) 
        logger.debug(f"last_folder_name: {last_folder_name}")
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            # Construire le nom de sortie
            output_txt_filename = f"{last_folder_name}_{filename}.txt"
            output_txt_path = os.path.join(output_folder, output_txt_filename)
            text = ""
            # Vérifier si ce fichier a déjà été créé (insensible à la casse)
            if output_txt_filename.lower() in existing_files_lower:
                logger.debug(f"Fichier déjà existant (ignoré) : {output_txt_filename}")
                continue
            try:
                #https://pymupdf.readthedocs.io/en/latest/tutorial.html
                if filename.lower().endswith(".pdf"):
                    # Spécifie le chemin d'installation de Tesseract
                    pytesseract.pytesseract.tesseract_cmd = parametre.file_pytesseract
                    # Ouvre le fichier PDF
                    doc = fitz.open(file_path)
                    for page_num in range(doc.page_count):
                        page = doc.load_page(page_num)
                        # Extrer du texte depuis la page
                        text_write = page.get_text()
                        text += text_write + "\n"  # Ajoute le texte de la page au fichier
                        # Extrer des images de la page et OCR -> https://stackoverflow.com/questions/78479905/extract-text-from-image-using-tessaract-and-opencv
                        for  img in page.get_images(full=True):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]  # Récupère les bytes de l'image
                            # Convertir les bytes en image PIL
                            image = Image.open(io.BytesIO(image_bytes))   
                            # Applique l'OCR pour extraire du texte de l'image 
                            ocr_txt = pytesseract.image_to_string(image, lang=parametre.lang_orc)
                            text += ocr_txt + "\n"  # Ajoute le texte OCR à la sortie
                elif filename.lower().endswith(".ipynb"):
                    file_content = open(file_path, "r", encoding="latin1").read()
                    notebook = json.loads(file_content)
                    code_lines = []
                    for cell in notebook.get("cells", []):
                        if cell.get("cell_type") == "code":
                            code_lines.extend(cell.get("source", []))
                            code_lines.append("\n")
                    text += "".join(code_lines)

                elif filename.lower().endswith(".csv"):
                    file_content = open(file_path, mode='r', encoding='latin1').read()
                    reader_csv = csv.reader(file_content.splitlines())
                    text = ""
                    for row in reader_csv:
                        text += ', '.join(row) + '\n'

                else:
                    logger.info(f"Fichier ignoré : {filename}")
                    logger.debug(f"Si information a extrere de {filename} programmer l'extraction")
                    continue
                

                clean_name = clean_filename(filename)
                output_txt_filename = output_folder + last_folder_name +"_" + clean_name +".txt"
                output_txt_path = os.path.join(output_folder, output_txt_filename)
                text = fix_text(text)
                text = re.sub(r'\s+', ' ', text)  # remplace tous les retours ligne, tabulations, etc. par un espace
                text = re.sub(r'([.?!])\s*', r'\1 ', text)  # s’assurer qu’il y a un espace après chaque ponctuation
                text = text.strip()    
                replacements = {
                    "•": "-",
                    "–": "-",
                    "—": "-",
                    "−": "-",
                    "◦": "-",
                    "▪": "-",
                } # ajouter si d'autre caractere avec meme signification trouver

                for char, repl in replacements.items():
                    text = text.replace(char, repl)

                with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
                logger.debug(f"{filename} converti et sauvegardé dans {output_txt_path}")
                existing_files_lower.add(output_txt_filename.lower())

            except Exception as e:
                msg = f"Erreur lors du traitement de {filename} : {e}"
                logger.error(msg)
                raise FileNotFoundError(msg)

    logger.info(f"Tous les fichiers ont été convertis dans {output_folder}")



