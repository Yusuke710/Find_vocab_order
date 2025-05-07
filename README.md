# JLPT Vocabulary Ordering via Greedy Min Surprise Topological Sort

## Project Goal

This project aims to find an optimized learning order for Japanese Language Proficiency Test (JLPT) N5 to N1 vocabulary. The optimization strategy is based on a "minimum surprise" principle: at each step, introduce the word whose example sentence contains the fewest *unknown* tokens, leveraging previously learned vocabulary. The goal is to create a smoother learning curve by minimizing cognitive load.

## How to Run

### Prerequisites

*   Python 3.x
*   A Gemini API Key (for sentence generation)

### Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Yusuke710/Find_vocab_order.git
    cd https://github.com/Yusuke710/Find_vocab_order.git
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment. You can create and activate one using `uv`:
    ```bash
    uv sync
    . .venv/bin/activate
    ```
    The script requires the following Python libraries:
    *   `pandas`
    *   `requests`
    *   `fugashi`
    *   `unidic-lite` (or another UniDic dictionary for `fugashi`)
    *   `networkx`
    *   `google-generativeai` (for LLM-based sentence generation)
    *   `python-dotenv` (for managing API keys)
    *   `matplotlib` (for plotting)
    *   `scipy` (for Kendall's Tau rank correlation)
    *   `tqdm` (for progress bars)

3.  **Set up Gemini API Key:**
    Create a file named `.env` in the root directory of the project and add your Gemini API key:
    ```
    GEMINI_API_KEY=YOUR_API_KEY_HERE
    ```

4.  **Prepare Seed Words:**
    Create a file named `seed_words.txt` in the root directory. Add one Japanese seed word per line. These are words assumed to be known from the start. Example:
    ```
    私
    は
    です
    ます
    ```
    If this file is not present or is empty, the script will run with no initial seed words (which might affect the ordering quality).

### Running the Script

Execute the main script from the project's root directory:
```bash
python vocab_order_min_surprise.py
```

## Methodology

The script aims to find a learning order for a given vocabulary list that minimizes "cognitive load" or "surprise" at each step. The core idea is to introduce new words whose example sentences contain the fewest *unknown* words at that point in the learning sequence.

1.  **Data Loading**: Loads vocabulary (word, reading, meaning, JLPT level) from the [elzup/jlpt-word-list](https://github.com/elzup/jlpt-word-list) repository (N5-N1 levels by default).
2.  **Sentence Generation**: For each vocabulary word, it generates a simple, common example sentence.
    *   Uses the Gemini LLM (`gemini-2.0-flash` by default) via the Google Generative AI SDK.
    *   Generated sentences are cached in `sentence_cache.csv` to avoid redundant API calls across runs. This file is loaded at the start and updated after sentence generation.
3.  **Tokenization**: Uses `fugashi` with `unidic-lite` to tokenize all generated sentences into individual words/tokens (surface forms).
4.  **Dependency Graph**: Builds a directed graph (`networkx.DiGraph`) where nodes are words (vocabulary + initial `SEED_WORDS`). An edge `(A, B)` exists if word `A` appears in the example sentence for word `B`. This represents that `A` is conceptually a prerequisite for understanding `B`'s example sentence.
5.  **Optimal Ordering**: This is the key step.
    *   **Initialization**: Starts with a set of known words (`SEED_WORDS`). A queue is initialized with all vocabulary words whose example sentences *only* contain words from the `SEED_WORDS` set (i.e., their prerequisites are already met).
    *   **Iteration**: While the queue is not empty:
        *   It selects the "best" word from the queue. The "best" word is defined as the one whose example sentence contains the minimum number of tokens *not* currently in the set of known words. **If multiple words have the same minimum cost, the tie is broken by selecting the word with the lower original index (i.e., the one appearing earlier in the initial N5-N1 list).**
        *   This "best" word is added to the `optimal_order` list and the set of known words.
        *   The algorithm then checks all vocabulary words that have the newly learned word as a prerequisite (its successors in the graph). For each successor, it checks if *all* of its prerequisites are now in the known set. If so, the successor is added to the queue, making it a candidate for the next learning step.
    *   **Output**: This process continues until the queue is empty, resulting in the `optimal_vocab_list`.
6.  **Analysis & Output**:
    *   Compares the `optimal_vocab_list` to the original list, calculating Kendall's Tau for rank correlation (if `scipy` is installed).
    *   Saves detailed results and comparisons to CSV files.
    *   Generates plots visualizing the cognitive load (if `matplotlib` is installed).

## Output Files

The script generates the following key files in the project's root directory:

*   `vocab_order_comparison.csv`: Detailed comparison of original vs. optimal order, including sentences, costs, and token information.
*   `sentence_cache.csv`: Cache of LLM-generated sentences to speed up subsequent runs.
*   `cognitive_load_plot.png`: (Optional) Plot showing the number of unknown words per sentence in the optimal order.

## Configuration

*   **Vocabulary Levels & Source:** Defined by `LEVELS` and `BASE_URL` constants in `vocab_order_min_surprise.py`.
*   **Seed Words:** Provided in `seed_words.txt`.
*   **Gemini API Key:** Set in the `.env` file.
*   **Output File Paths:** Defined as constants (e.g., `OUTPUT_CSV_PATH`) in `vocab_order_min_surprise.py`.