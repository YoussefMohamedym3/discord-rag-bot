import os

import pandas as pd
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser

from src.config.settings import AppSettings


def inspect_chunks(input_dir: str, output_csv: str):
    print(f"🕵️  Inspecting data in: {input_dir}")

    # 1. Load Documents
    reader = SimpleDirectoryReader(input_dir=input_dir, recursive=True)
    documents = reader.load_data()
    print(f"📄 Loaded {len(documents)} source files.")

    # 2. Run ONLY the Parser (No Embeddings)
    # This simulates exactly how the Indexer will chop up your text
    parser = MarkdownNodeParser(include_metadata=True)
    nodes = parser.get_nodes_from_documents(documents)

    print(f"✂️  Parsed into {len(nodes)} chunks.")

    # 3. Convert to DataFrame for Inspection
    data = []
    for node in nodes:
        data.append(
            {
                "File Name": node.metadata.get("file_name", "Unknown"),
                # This captures the # Header > ## Subheader path
                "Header Path": node.metadata.get("header_path", "None"),
                "Character Count": len(node.text),
                # Preview the text to ensure it looks clean
                "Content Preview": node.text[:100].replace("\n", " ") + "...",
                "Full Content": node.text,
            }
        )

    df = pd.DataFrame(data)

    # 4. Save to CSV
    # Using '~' as separator prevents issues if your text contains commas
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, sep="~", index=False, encoding="utf-8")

    # 5. Print Summary Statistics
    print("\n📊 --- CHUNK STATISTICS ---")
    print(f"Total Chunks: {len(nodes)}")
    if not df.empty:
        print(f"Smallest Chunk: {df['Character Count'].min()} chars")
        print(f"Largest Chunk:  {df['Character Count'].max()} chars")
        print(f"Average Size:   {int(df['Character Count'].mean())} chars")

    print(f"\n💾 Detailed report saved to: {output_csv}")
    print("👉 Open in Excel -> Data -> Text to Columns -> Delimiter: '~'")


if __name__ == "__main__":
    # By default, inspect the 'Silver' folder (Cleaned Data)
    # This helps you verify if your cleaning script actually worked
    in_dir = AppSettings.DATA_SILVER_DIR

    # Save the report to a 'debug_reports' folder in the root
    out_file = os.path.join("debug_reports", "chunks_report.csv")

    inspect_chunks(in_dir, out_file)
