from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter


# This was the article that we used:
#     https://www.espncricinfo.com/story/mohammed-siraj-finds-his-rhythm-to-lead-rcb-bowlers-rout-of-gujarat-titans-1432331

example_text=""

with open('example_data/wc.txt', 'r') as file:
    example_text = file.read()


character_text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=600,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)
character_split_chunks = character_text_splitter.split_text(example_text)

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
)
recursive_split_chunks = recursive_text_splitter.split_text(example_text)

print("----Character Split Chunks----")
print(character_split_chunks[0])
print("--------")
print("----Recursive Character Split Chunks----")
print(recursive_split_chunks[0])



