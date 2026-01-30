import os
import re
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TransformComponent


class BronzeToSilverCleaner(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        for node in nodes:
            text = node.get_content()

            # 1. Separator Destroyer to remove the " ------ " lines
            text = re.sub(r"(?m)^\s*[-—=_]{3,}\s*$", "", text)

            # 2. Remove Image Refs
            text = re.sub(r"!\[.*?\](?:\[.*?\]|\(.*?\))", "", text)
            text = re.sub(r"(?m)^\[image.*?\]:\s*<data:image[^>]+>", "", text)

            # 3. Universal Un-escaper
            text = re.sub(r"\\([!\[\]().*+=\-_#&|<>])", r"\1", text)

            # 4. Fix Links
            text = re.sub(r"\[(https?://[^\]]+)\]\(\1\)", r"\1", text)
            text = text.replace("[]", "")

            # 5. Remove Emojis
            emoji_pattern = r"[\U00010000-\U0010ffff\u2700-\u27bf\u2600-\u26ff\ufe0f]"
            text = re.sub(emoji_pattern, "", text)

            # 6. Header Promotion Logic
            pattern = r"(?m)^\s*([\*\-]?)\s*(\*{2,3})(.*?)\2(.*)$"

            def header_replacement(match):
                prefix = match.group(1)
                header_text = match.group(3)
                trailing_text = match.group(4)
                clean_text = header_text.lower().strip()

                is_role_marker = any(
                    k in clean_text
                    for k in [
                        "engineer",
                        "designer",
                        "manager",
                        "all",
                        "data scientist",
                    ]
                )

                if is_role_marker:
                    return f"\n* **{header_text}**{trailing_text}\n"
                elif prefix and prefix.strip():
                    return f"{prefix} **{header_text}**{trailing_text}"
                else:
                    return f"\n## {header_text}{trailing_text}\n"

            text = re.sub(pattern, header_replacement, text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            node.set_content(text)

        return nodes


def run_cleaning_pipeline(input_dir: str, output_dir: str):
    """Reads Raw data, cleans it, and saves to Silver."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"🧹 Cleaning Data: {input_dir} -> {output_dir}")

    reader = SimpleDirectoryReader(input_dir=input_dir, recursive=True)
    pipeline = IngestionPipeline(transformations=[BronzeToSilverCleaner()])

    for docs in reader.iter_data():
        clean_docs = pipeline.run(documents=docs)
        for doc in clean_docs:
            filename = doc.metadata.get("file_name")
            out_path = os.path.join(output_dir, f"SILVER_{filename}")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(doc.text)
            print(f"   ✨ Saved: {out_path}")
