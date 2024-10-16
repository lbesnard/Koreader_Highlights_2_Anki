# Koreader_Highlights_2_Anki


## Installation with Poetry
Clone the repo
```bash
pip install poetry
poetry install
```

### Usage

```bash
poetry run koreader_highlights_2_anki -h
usage: koreader_highlights_2_anki [-h] --input-folder INPUT_FOLDER --output-folder OUTPUT_FOLDER [--deck-name DECK_NAME] [--no-ai] [--select-files]

Create Anki flashcards from Koreader highlights.

options:
  -h, --help            show this help message and exit
  --input-folder INPUT_FOLDER, -i INPUT_FOLDER
                        Path to the folder containing metadata.epub.lua files in KOReader.
  --output-folder OUTPUT_FOLDER, -o OUTPUT_FOLDER
                        Path to the folder where .apkg files will be saved.
  --deck-name DECK_NAME, -n DECK_NAME
                        Name of the Anki deck (e.g., 'deck::subdeck') (Default to Books Highlights 📚 .
  --no-ai               Decide if we don't use AI to find which part of the sentence to 'cloze'.
  --select-files        If set, allows you to select files interactively for processing.

example:
poetry run koreader_highlights_2_anki -i /media/KOBOeReader/Ebooks -o ~/Documents --select-files

```
