#!/usr/bin/env python3
"""
This script generates Anki flashcards from highlighted text using AI to create cloze deletions.
It utilizes the BERT model to assess word importance in sentences, allowing for the generation of effective
flashcards for study and revision.

Key Features:
- Parses highlights from a KOReader LUA file to extract bookmarks and book metadata.
- Uses BERT to determine word importance in sentences to create cloze deletions.
- Generates flashcards in the Anki format with customizable deck names and output paths.
- Supports multiple languages by using NLTK stopwords.
- Includes error handling and logging to aid in debugging.

Dependencies:
- genanki: For creating Anki flashcards.
- nltk: For natural language processing, including tokenization and stopword handling.
- torch and transformers: For utilizing the BERT model for masked language modeling.
- lupa: For executing Lua scripts.

Usage:
1. Parse a Lua file containing highlights using `parse_lua_highlights(filepath)`.
2. Generate Anki flashcards with `create_anki_flashcards(parsed_data, output_apkg_path, deck_name)`.
3. Optionally, create AI-assisted cloze deletions with `generate_cloze_with_ai(highlight_text, highlight_sentences, note_text, model, deck, highlight, language)`.
4. Save the generated .apkg file for import into Anki.

Example:
```python
./koreader_highlights_anki.py -i /media/KOBOeReader/Ebooks/ -o ~/Documents/AnkiHighlights
"""

import argparse
import fnmatch
import hashlib
import inspect
import logging
import os
import random
import re
import traceback
import warnings

import genanki
import inquirer
import nltk
import torch
from iterfzf import iterfzf
from lupa import LuaRuntime
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

# Ignore specific warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
# Suppress warnings from transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_word_importance(sentence, language="en"):
    """
    Use BERT to score word importance by masking each word and predicting it.
    Words that are harder for BERT to predict will be considered more important.
    """
    # Load the pretrained model and tokenizer

    logger.debug(f"Processing sentence: {sentence}")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.clean_up_tokenization_spaces = True
    model = BertForMaskedLM.from_pretrained(model_name)

    # Download NLTK stopwords if you haven't already
    import nltk

    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    stop_words_dict = {
        "en-US": set(stopwords.words("english")),
        "en": set(stopwords.words("english")),
        "fr": set(stopwords.words("french")),
        "de": set(stopwords.words("german")),
        "es": set(stopwords.words("spanish")),
        "it": set(stopwords.words("italian")),
    }
    stop_words = stop_words_dict.get(language, stop_words_dict["en-US"])

    words = sentence.split()
    word_scores = []

    for i, word in enumerate(words):
        # Skip punctuation if necessary
        if word.lower() in stop_words or not word.isalpha():
            continue

        # Tokenize and mask the current word
        masked_sentence = words[:i] + ["[MASK]"] + words[i + 1 :]
        inputs = tokenizer(
            " ".join(masked_sentence),
            return_tensors="pt",
            clean_up_tokenization_spaces=True,
        )

        # Get BERT predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        # Get the token ID for the masked word
        masked_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(
            as_tuple=True
        )[1]

        predicted_token_id = torch.argmax(predictions[0, masked_index, :], dim=-1)

        # Score based on the predicted token's probability
        predicted_prob = torch.softmax(predictions[0, masked_index, :], dim=-1)[
            0, predicted_token_id
        ].item()
        importance_score = 1 - predicted_prob  # Lower probability means more important
        word_scores.append((word, importance_score))

    # Sort words by their importance score (higher is more important)
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return word_scores


def generate_cloze_with_ai(
    highlight_text,
    highlight_sentences,
    note_text,
    model,
    deck,
    highlight,
    language="en-US",
):
    """
    Generates cloze deletions using AI to select the most important words while keeping the full highlight text.

    Args:
        highlight_text (str): The full highlight text containing all sentences.
        highlight_sentences (list): A list of sentences to process for cloze deletions.
        note_text (str): Text to include in the note.
        model: The Anki model to use.
        deck: The Anki deck to add notes to.
        highlight (dict): Additional highlight information including datetime.

    Returns:
        deck: The updated Anki deck with added cloze notes.
    """
    # Ensure NLTK punkt tokenizer models are downloaded
    nltk.download("punkt", quiet=True)

    # Create a copy of the original highlight text for replacement
    full_highlight_cloze = highlight_text
    logger.debug(
        f"Starting to process {len(highlight_sentences)} highlight sentences..."
    )
    for sentence in highlight_sentences:
        words = sentence.split()

        if len(words) < 2:  # Skip sentences that are too short
            continue

        # Use AI to determine word importance in the sentence
        word_scores = get_word_importance(sentence, language=language)
        # Select the most important word
        most_important_word = word_scores[0][0] if word_scores else None

        if most_important_word:
            # Find the index of the most important word
            most_important_index = words.index(most_important_word)

            # Determine context words (2 to 3 surrounding words)
            context_words = []
            start_index = max(0, most_important_index - 2)  # 2 words before
            end_index = min(len(words), most_important_index + 3)  # 3 words after

            # Create a cloze text from the important word and context
            cloze_text = " ".join(words[start_index:end_index])

            # Replace the selected words with cloze syntax in the full highlight text
            full_highlight_cloze = full_highlight_cloze.replace(
                cloze_text, f"{{{{c1::{cloze_text}}}}}"
            )
    logger.debug("Finished processing all sentences.")
    # Create the datetime string for the bottom right
    datetime_info = f"<span style='color: lightgrey; font-size: small; font-style: italic; float: right;'>{highlight['datetime']}</span>"

    # Add the note to the deck with the full highlight text containing cloze deletions
    note = genanki.Note(
        model=model,
        fields=[
            note_text + full_highlight_cloze,
            datetime_info,
        ],  # Add datetime in Extra field
    )
    deck.add_note(note)

    return deck


def create_anki_flashcards_ai(
    parsed_data, output_apkg_path, deck_name="Books Highlights 📚"
):
    """
    Creates cloze flashcards for Anki from the highlights.

    Args:
        parsed_data (dict): Parsed highlights and book metadata (output of parse_lua_highlights).
        deck_name (str): Name of the Anki deck in the format 'Books Highlights::[book_title]-[Author_name]'.
        output_apkg_path (str): Path where the .apkg file should be saved. Can be a folder or a full path.

    Returns:
        None
    """
    # Download NLTK punkt tokenizer models if you haven't already
    nltk.download("punkt_tab")
    nltk.download("punkt")

    language = parsed_data["language"]

    book_title = parsed_data["title"].replace(" ", "_")
    author_name = parsed_data["authors"].replace(" ", "_")

    # Set deck name in format 'Books Highlights::[book_title]-[author_name]'
    deck_full_name = (
        f"{deck_name}::{book_title.replace('_', ' ')}-{author_name.replace('_', ' ')}"
    )

    # Generate deck ID using a hash of the deck_full_name
    deck_id = int(hashlib.md5(deck_full_name.encode("utf-8")).hexdigest(), 16) % (
        10**10
    )

    deck = genanki.Deck(deck_id=deck_id, name=deck_full_name)

    model = genanki.CLOZE_MODEL

    # Generate cards for each highlight
    for highlight in tqdm(
        parsed_data["entries"], desc="Processing book highlights", unit="highlight"
    ):
        highlight_text = highlight["notes"]

        # Use regex to split sentences by both '.' and ';'
        highlight_sentences = re.split(
            r"(?<=[.!?;]) +", highlight_text
        )  # Tokenize sentences
        page_number = highlight["page"]
        chapter_info = f"Chapter: {highlight['chapter']} - Page: {page_number}"

        # Keep your original formatting for title and note_text
        title = f"<b style='font-size: 24px;'>{book_title.replace('_', ' ')} - {author_name.replace('_', ' ')}</b><br><br>"  # Increase font size of the title
        note_text = f"<span style='font-size: 14px;'>{title}{chapter_info}</span><hr>"  # Decrease font size of the note text

        # Generate cloze deletions using AI
        deck = generate_cloze_with_ai(
            highlight_text,
            highlight_sentences,
            note_text,
            model,
            deck,
            highlight,
            language=language,
        )

    # Check if output_apkg_path is a folder, and if so, append a default file name
    if os.path.isdir(output_apkg_path):
        default_filename = f"{book_title}_{author_name}.apkg".replace("/", ".")
        output_apkg_path = os.path.join(output_apkg_path, default_filename)

    # Create the Anki package and save it
    package = genanki.Package(deck)
    package.write_to_file(output_apkg_path)


def create_anki_flashcards(
    parsed_data, output_apkg_path, deck_name="Books Highlights 📚"
):
    """
    Creates cloze flashcards for Anki from the highlights.

    Args:
        parsed_data (dict): Parsed highlights and book metadata (output of parse_lua_highlights).
        deck_name (str): Name of the Anki deck in the format 'Books Highlights::[book_title]-[Author_name]'.
        output_apkg_path (str): Path where the .apkg file should be saved. Can be a folder or a full path.

    Returns:
        None
    """
    # Download NLTK punkt tokenizer models if you haven't already
    nltk.download("punkt_tab")
    nltk.download("punkt")

    book_title = parsed_data["title"].replace(" ", "_")
    author_name = parsed_data["authors"].replace(" ", "_")

    # Set deck name in format 'Books Highlights::[book_title]-[author_name]'
    deck_full_name = (
        f"{deck_name}::{book_title.replace('_', ' ')}-{author_name.replace('_', ' ')}"
    )

    # Generate deck ID using a hash of the deck_full_name
    deck_id = int(hashlib.md5(deck_full_name.encode("utf-8")).hexdigest(), 16) % (
        10**10
    )

    deck = genanki.Deck(deck_id=deck_id, name=deck_full_name)

    model = genanki.CLOZE_MODEL

    # Generate cards for each highlight
    for highlight in parsed_data["entries"]:
        highlight_text = highlight["notes"]

        # Use regex to split sentences by both '.' and ';'
        highlight_sentences = re.split(
            r"(?<=[.!?;]) +", highlight_text
        )  # Tokenize sentences
        page_number = highlight["page"]
        chapter_info = f"Chapter: {highlight['chapter']} - Page: {page_number}"

        # Keep your original formatting for title and note_text
        title = f"<b style='font-size: 24px;'>{book_title.replace('_', ' ')} - {author_name.replace('_', ' ')}</b><br><br>"  # Increase font size of the title
        note_text = f"<span style='font-size: 14px;'>{title}{chapter_info}</span><hr>"  # Decrease font size of the note text

        # For each sentence, create a separate card but display the full highlight text
        for sentence in highlight_sentences:
            words = sentence.split()
            if len(words) < 2:  # Skip sentences that are too short
                continue

            # Determine how many words to cloze (between 2 and 4)
            cloze_count = random.randint(
                2, min(4, len(words))
            )  # Ensure we don't exceed the sentence length

            # Randomly select a starting index for cloze deletion
            start_index = random.randint(0, len(words) - cloze_count)
            cloze_text = " ".join(words[start_index : start_index + cloze_count])

            # Replace the selected words with cloze syntax
            cloze_sentence = sentence.replace(cloze_text, f"{{{{c1::{cloze_text}}}}}")

            # Replace the sentence in the full highlight text with the cloze sentence
            full_highlight_cloze = highlight_text.replace(sentence, cloze_sentence)

            # Create the datetime string for the bottom right
            datetime_info = f"<span style='color: lightgrey; font-size: small; font-style: italic; float: right;'>{highlight['datetime']}</span>"

            note = genanki.Note(
                model=model,
                fields=[
                    note_text + full_highlight_cloze,
                    datetime_info,
                ],  # Add datetime in Extra field
            )
            deck.add_note(note)

    # Check if output_apkg_path is a folder, and if so, append a default file name
    if os.path.isdir(output_apkg_path):
        default_filename = f"{book_title}_{author_name}.apkg".replace("/", ".")
        output_apkg_path = os.path.join(output_apkg_path, default_filename)

    # Create the Anki package and save it
    package = genanki.Package(deck)
    package.write_to_file(output_apkg_path)


def parse_lua_highlights_annotations(filepath):
    """
    Parses a Lua file to extract highlighted bookmarks or annotations and book metadata.

    Args:
        filepath (str): Path to the Lua file.

    Returns:
        dict: Dictionary containing the highlighted entries (bookmarks or annotations),
              book title, author, language, and optionally page numbers.
              Returns None if no highlights are found.
    """
    # Create a Lua runtime environment
    lua = LuaRuntime(unpack_returned_tuples=True)

    # Load the Lua file content
    with open(filepath, "r", encoding="utf-8") as file:
        lua_content = file.read()

    # Wrap the Lua content in a function for execution
    lua_content_wrapped = f"function load_data() {lua_content} end"

    # Execute the wrapped content to define the function
    lua.execute(lua_content_wrapped)

    # Call the function to retrieve the Lua table
    lua_table = lua.globals().load_data()

    # Access the bookmarks and annotations
    annotations = lua_table["annotations"]

    highlighted_entries = []
    try:
        for annotation in annotations.values():
            if annotation["text"]:
                highlighted_entries.append(
                    {
                        "chapter": annotation["chapter"],
                        "datetime": annotation["datetime"],
                        "notes": annotation["text"],
                        "page": annotation["pageno"],
                    }
                )
    except Exception as e:
        logger.debug(
            f"Error parsing Lua file with {inspect.currentframe().f_code.co_name}"
        )
        return None

    # Extract book metadata (authors, title, language)
    metadata = {
        "title": lua_table["stats"]["title"],
        "authors": lua_table["stats"]["authors"],
        "language": lua_table["stats"]["language"],
        "entries": highlighted_entries,
    }

    # Return None if no highlighted entries found
    if not highlighted_entries:
        return None

    return metadata


def parse_lua_highlights_bookmarks(filepath):
    """
    Parses a Lua file to extract highlighted bookmarks or annotations and book metadata.

    Args:
        filepath (str): Path to the Lua file.

    Returns:
        dict: Dictionary containing the highlighted entries (bookmarks or annotations),
              book title, author, language, and optionally page numbers.
              Returns None if no highlights are found.
    """
    # Create a Lua runtime environment
    lua = LuaRuntime(unpack_returned_tuples=True)

    # Load the Lua file content
    with open(filepath, "r", encoding="utf-8") as file:
        lua_content = file.read()

    # Evaluate the Lua content
    lua_table = lua.execute(lua_content)

    bookmarks = lua_table["bookmarks"]

    highlighted_entries = []
    try:
        for bookmark in bookmarks.values():
            if bookmark["highlighted"]:
                page = bookmark["page"]

                # Regular expression to match the number after "DocFragment"
                match = re.search(r"DocFragment\[(\d+)\]", page)

                # If a match is found, extract the number
                if match:
                    page = match.group(1)
                else:
                    page = None

                highlighted_entries.append(
                    {
                        "chapter": bookmark["chapter"],
                        "datetime": bookmark["datetime"],
                        "notes": bookmark["notes"],
                        "page": page,
                    }
                )
    except Exception as e:
        logger.debug(
            f"Error parsing Lua file with {inspect.currentframe().f_code.co_name}"
        )
        return None

    # Extract book metadata (authors, title, language)
    metadata = {
        "title": lua_table["stats"]["title"],
        "authors": lua_table["stats"]["authors"],
        "language": lua_table["stats"]["language"],
        "entries": highlighted_entries,
    }

    # Return None if no highlighted entries found
    if not highlighted_entries:
        return None

    return metadata


def main():
    """
    Main function to scan directory for KOREADER metadata files and create Anki flashcards .

    Args:
        None (argparse used for input)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Create Anki flashcards from Koreader highlights."
    )
    parser.add_argument(
        "--input-folder",
        "-i",
        required=True,
        help="Path to the folder containing metadata.epub.lua files in KOReader.",
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        required=True,
        help="Path to the folder where .apkg files will be saved.",
    )
    parser.add_argument(
        "--deck-name",
        "-n",
        default="Books Highlights 📚",
        required=False,
        help="Name of the Anki deck (e.g., 'deck::subdeck') (Default to Books Highlights 📚 .",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Decide if we don't use AI to find which part of the sentence to 'cloze'.",
    )
    parser.add_argument(
        "--select-files",
        "-s",
        action="store_true",
        help="If set, allows you to select files interactively for processing.",
    )
    args = parser.parse_args()

    # Check if the output folder exists, create it if it doesn't
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logger.info(f"Created output folder: {args.output_folder}")

    # Recursively find all metadata.epub.lua files in the input folder
    lua_files = []
    for root, _, files in os.walk(args.input_folder):
        for file in files:
            if fnmatch.fnmatch(file, "metadata.*.lua"):
                file_path = os.path.join(root, file)
                lua_files.append(file_path)

    if not lua_files:
        logger.error("No metadata.*.lua files found in the input folder.")
        return
    else:
        nltk.download = lambda *args, **kwargs: None

    # If --select-files is passed, present a list for the user to select files
    if args.select_files:
        #         questions = [
        # inquirer.Checkbox(
        # "selected_files",
        # message="Select the files to process (use space to select, arrows to navigate):",
        # choices=lua_files,
        # )
        # ]
        # answers = inquirer.prompt(questions)
        #         selected_files = answers.get("selected_files", [])
        prompt_message = "Select the files to process with TAB (You can type characters of a book to filter out):"
        selected_files = list(iterfzf(lua_files, multi=True, prompt=prompt_message))

    else:
        selected_files = lua_files

    for file_path in selected_files:
        logger.info(f"Processing {file_path}")

        try:
            clipped_highlights = parse_lua_highlights_bookmarks(file_path)
            if clipped_highlights is None:
                logger.debug(
                    f"Error parsing Bookmarks from Lua file, try using Annotations instead: {file_path}"
                )
                clipped_highlights = parse_lua_highlights_annotations(file_path)

            if clipped_highlights:
                # Create Anki flashcards
                if args.no_ai is True:
                    create_anki_flashcards(
                        clipped_highlights, args.output_folder, deck_name=args.deck_name
                    )
                else:
                    create_anki_flashcards_ai(
                        clipped_highlights, args.output_folder, deck_name=args.deck_name
                    )
            else:
                logger.warning(f"No highlights found for {file_path}")

        except Exception as e:
            logger.error(
                f"An error occurred while processing the highlights from {file_path}: {e}\n {traceback.print_exc()}"
            )


if __name__ == "__main__":
    main()
