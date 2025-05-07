"""
jlpt_order_min_surprise.py

Finds an optimized learning order for JLPT vocabulary based on minimizing
the introduction of unknown words in example sentences.

Requirements:
- pandas
- requests
- fugashi
- unidic-lite
- networkx
- matplotlib (optional for plotting)
- scipy (optional for rank correlation)
- tqdm (optional for progress bars)
"""

import io
import sys
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import os
from typing import Tuple # Added for type hinting

# Third-party imports
import networkx as nx
import pandas as pd
import requests
import fugashi

# Optional imports
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    from scipy.stats import kendalltau
except ImportError:
    kendalltau = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ---------------- Constants & Configuration ----------------
VOCAB_COLUMN = "expression" # Column name in CSV containing the vocabulary word
BASE_URL = "https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/"
LEVELS = ["n5.csv", "n4.csv", "n3.csv", "n2.csv", "n1.csv"] # Ensure N5 is present to be used as seeds
OUTPUT_CSV_PATH = "vocab_order_comparison.csv"
SENTENCE_CACHE_CSV_PATH = "sentence_cache.csv" # New path for sentence cache

# Initialize tokenizer globally (or pass it around if preferred)
tagger = fugashi.Tagger()

def tokenize(text: str) -> list[str]:
    """Tokenizes Japanese text using fugashi."""
    # Consider adding error handling or filtering (e.g., for punctuation)
    return [token.surface for token in tagger(text)]

# ---------------- 1. Data Loading ----------------
def load_vocabulary(base_url: str, levels: list[str], vocab_col: str) -> Tuple[pd.DataFrame, set[str]]:
    """Loads vocabulary lists, adds level info, keeps essential columns,
    removes duplicates, and returns the full DataFrame and the set of N5 words.

    Returns:
        A tuple containing:
        - vocab_df: DataFrame with all unique vocabulary words.
        - n5_words: Set of words from the N5 level.
    """
    print(f"Loading vocabulary from {len(levels)} levels...")
    dfs = []
    n5_words_list = [] # List to collect N5 words before deduplication
    required_cols = {vocab_col, 'reading', 'meaning'} # Columns we need

    for level_file in levels:
        url = base_url + level_file
        level_name = level_file.split('.')[0].upper() # Extract N5, N4 etc.
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            df = pd.read_csv(io.StringIO(response.text))

            # Check if required columns exist
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                print(f"Warning: File for level '{level_name}' ({level_file}) is missing columns: {missing_cols}. Skipping this level.", file=sys.stderr)
                continue

            # Add the level information
            df['jlpt_level'] = level_name

            # Select only the columns we need (including the new level)
            cols_to_keep = list(required_cols) + ['jlpt_level']
            dfs.append(df[cols_to_keep])

            # --- Extract N5 words before deduplication --- 
            if level_name == 'N5' and vocab_col in df.columns:
                n5_words_list.extend(df[vocab_col].astype(str).tolist()) # Add all N5 words from this file
            # ---------------------------------------------

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}", file=sys.stderr)
        except pd.errors.ParserError as e:
            print(f"Error parsing {url}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred processing {url}: {e}", file=sys.stderr)

    if not dfs:
        print("Error: No valid vocabulary data loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Concatenate and drop duplicates based on the main vocabulary word
    # Keep the first occurrence to retain its reading/meaning/level
    vocab_df = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=[vocab_col], keep='first')
        .reset_index(drop=True)
    )
    # Ensure reading/meaning/level are strings, fill NaNs
    vocab_df['reading'] = vocab_df['reading'].fillna('').astype(str)
    vocab_df['meaning'] = vocab_df['meaning'].fillna('').astype(str)
    vocab_df['jlpt_level'] = vocab_df['jlpt_level'].fillna('Unknown').astype(str) # Handle potential rare cases

    # Create the final set of unique N5 words
    n5_words = set(n5_words_list)

    print(f"Loaded {len(vocab_df)} unique words with reading, meaning, and level.")
    print(f"Extracted {len(n5_words)} N5 words to be used as seeds.")
    return vocab_df, n5_words

# ---------------- 2. Sentence Generation ----------------
class SentenceGenerator(ABC):
    """Abstract base class for sentence generators."""
    @abstractmethod
    def generate(self, word: str) -> str:
        pass

class NaiveSentenceGenerator(SentenceGenerator):
    """Generates very simple, template-based sentences."""
    def generate(self, word: str) -> str:
        """Very naive sentence generator."""
        # Consider using POS tagging for better template selection
        if word.endswith("い"):             # Adjective guess
            return f"{word} 日 です。"
        elif word.endswith(("る", "う")):   # Verb-ish guess
            return f"私 は まいにち {word}。"
        else:                           # Noun/other default
            return f"{word} は 便利 な もの です。"

# TODO: Implement LLMSentenceGenerator(SentenceGenerator) later
class LLMSentenceGenerator(SentenceGenerator):
    """Generates sentences using the Gemini LLM."""
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """Initializes the Gemini client.

        Args:
            api_key: Your Gemini API key. (Best practice: load from env variable)
            model_name: The specific Gemini model to use.
        """
        print(f"Initializing LLM Generator with model: {model_name}")
        self.model_name = model_name
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # Initialize the generative model
            self.model = genai.GenerativeModel(self.model_name)
            print("Gemini client configured successfully.")
        except ImportError:
            print("Error: google-generativeai library not found.", file=sys.stderr)
            print("Please install it: `uv add google-generativeai`", file=sys.stderr)
            self.model = None
        except Exception as e:
            print(f"Error configuring Gemini client: {e}", file=sys.stderr)
            self.model = None

    def generate(self, word: str) -> str:
        """Generates a simple, common sentence for the word using the LLM."""
        if not self.model:
            print("Error: LLM model not initialized.", file=sys.stderr)
            return ""

        # Prompt designed to elicit a common, natural, and simple example sentence
        # This indirectly encourages sentences with higher likelihood based on training data.
        prompt = f"日本語の単語「{word}」を使った、一般的で自然な短い例文を1つだけ作ってください。"
        # Translation: "Please create just one short, common, and natural example sentence using the Japanese word '{word}'."

        try:
            # Make the actual API call to the Gemini model
            response = self.model.generate_content(
                prompt,
                # Optional: Add safety settings or generation config if needed
                # safety_settings={...},
                # generation_config=genai.types.GenerationConfig(...)
            )

            # Extract the text, handling potential issues
            if response.parts:
                generated_sentence = response.text.strip()
            else:
                # Handle cases where the model might refuse to answer (e.g., safety filters)
                print(f"Warning: LLM response for '{word}' was empty or blocked. Prompt: {prompt}. Reason: {response.prompt_feedback.block_reason}", file=sys.stderr)
                generated_sentence = ""

            # Basic post-processing (optional)
            generated_sentence = generated_sentence.replace("\n", " ").strip()
            # Remove potential markdown like backticks if the model adds them
            generated_sentence = generated_sentence.strip("`")

            # Optional: Add a small delay to avoid hitting rate limits if making many calls rapidly
            # import time
            # time.sleep(0.5) 

            return generated_sentence

        except Exception as e:
            print(f"Error during LLM API call for '{word}': {e}", file=sys.stderr)
            # Consider more specific error handling based on potential API errors
            return ""

def generate_all_sentences(vocab_list: list[str], generator: SentenceGenerator, cache: dict[str, str]) -> dict[str, str]:
    """Generates a sentence for each word, using cache if available."""
    print(f"Generating sentences for {len(vocab_list)} words (using cache where possible)...")
    sentences = {}
    cached_count = 0
    generated_count = 0
    iterator = tqdm(vocab_list) if tqdm else vocab_list

    for word in iterator:
        # Check cache first
        cached_sentence = cache.get(word)
        if cached_sentence: # Check if sentence exists and is not empty
            sentences[word] = cached_sentence
            cached_count += 1
        else:
            # If not in cache or sentence is empty, generate anew
            try:
                sentences[word] = generator.generate(word)
                generated_count += 1
            except Exception as e:
                print(f"Error generating sentence for '{word}': {e}", file=sys.stderr)
                sentences[word] = ""

    print(f"Sentence generation complete: {cached_count} loaded from cache, {generated_count} newly generated.")
    return sentences

# ---------------- 3. Dependency Graph Building ----------------
def build_dependency_graph(words_to_order: list[str], seed_words: set[str], tokenized_sentences: dict[str, list[str]]) -> nx.DiGraph:
    """Builds a directed graph where edges represent word prerequisites.

    Args:
        words_to_order: List of words to be included in the graph (excluding seeds).
        seed_words: Set of initial seed words.
        tokenized_sentences: Dictionary mapping words to their tokenized sentences.

    Returns:
        A directed graph.
    """
    print("Building dependency graph...")
    graph = nx.DiGraph()
    # Start with known words = seeds
    known_words_for_graph = set(seed_words)

    # Add all potential nodes first (seeds + words to order)
    all_nodes = set(words_to_order).union(seed_words)
    graph.add_nodes_from(all_nodes)

    # Iterate only through the words we need to order
    iterator = tqdm(words_to_order) if tqdm else words_to_order
    for word in iterator:
        tokens = tokenized_sentences.get(word) # Use pre-tokenized list
        if not tokens: continue # Skip if no tokens (empty/failed sentence)

        # Find dependencies within the current set of known words (including seeds and previously added words_to_order)
        # Only add edges from nodes that actually exist in our graph (all_nodes)
        dependencies = {token for token in tokens if token in known_words_for_graph and token in all_nodes}

        for dep in dependencies:
            if dep != word: # Avoid self-loops
                graph.add_edge(dep, word) # Edge: prerequisite -> new word

        # Add the new word itself to known set *after* processing its deps
        known_words_for_graph.add(word)

    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

# ---------------- 4. Optimal Ordering Algorithm ----------------
def find_min_surprise_order(
    words_to_order: list[str], 
    sentences_map: dict[str, str], 
    prereq_graph: nx.DiGraph, 
    seed_vocab: set[str], 
    tokenized_sentences: dict[str, list[str]],
    original_index_map: dict[str, int] # Add parameter for original index
) -> list[str]:
    """Performs a sort prioritizing words whose sentences introduce fewest new tokens,
    ensuring words are added once all prerequisites (including seeds) are met.
    Tie-breaking: prioritizes words with lower original index.

    Args:
        words_to_order: List of vocabulary words to sort.
        sentences_map: Dictionary mapping words to their generated sentences.
        prereq_graph: Directed graph where edges (u, v) mean u is a prerequisite for v.
                      Includes edges from seed words and between vocab words.
        seed_vocab: Set of initial known words (not part of words_to_order).
        tokenized_sentences: Dictionary mapping words to their tokenized sentences.
        original_index_map: Dictionary mapping words to their original 0-based index.

    Returns:
        A list of words from words_to_order in the calculated optimal order.
    """
    print("Sorting words by minimum surprise...")
    # Build adjacency list (successors): u -> list of v where (u, v) is an edge
    adj_list = defaultdict(list)
    for u, v in prereq_graph.edges:
        # We only care about successors that are actually in the list we need to order
        if v in words_to_order:
            adj_list[u].append(v)

    current_known = set(seed_vocab)
    optimal_order = []
    words_remaining = set(words_to_order) # Keep track of words not yet ordered
    queue = deque() # Words whose prerequisites are met

    # --- Initial Queue Population ---
    # Find words whose prerequisites are *already* fully met by seed_vocab
    print("Initializing queue with words learnable from seeds...")
    initially_learnable = []
    for word in words_to_order:
        # Get all predecessors that are actually in the graph
        prereqs = {p for p in prereq_graph.predecessors(word) if prereq_graph.has_node(p)}
        # Check if all prerequisites are within the initial seed set
        if prereqs.issubset(seed_vocab):
            initially_learnable.append(word)

    for word in initially_learnable:
        queue.append(word)
        if word in words_remaining: # Avoid removing if somehow not there
             words_remaining.remove(word)
    print(f"Initialized queue with {len(queue)} words.")
    # --------------------------------

    num_words = len(words_to_order)
    pbar = tqdm(total=num_words) if tqdm else None

    while queue:
        # Find word in queue whose sentence adds the fewest unknown tokens
        # Use pre-tokenized sentences here for cost calculation
        # Tie-breaking: use original index (lower is better)
        best_word = min(
            queue,
            key=lambda w: (
                sum(token not in current_known for token in tokenized_sentences.get(w, [])),
                original_index_map.get(w, float('inf')) # Use original index for tie-breaking
            )
        )

        queue.remove(best_word)
        optimal_order.append(best_word)
        current_known.add(best_word) # Add the newly learned word
        # words_remaining should already have 'best_word' removed when added to queue

        if pbar: pbar.update(1)

        # --- Update Neighbors ---
        # Check words that depend on best_word (its successors)
        for neighbor in adj_list.get(best_word, []):
            if neighbor in words_remaining: # Only check words not yet ordered/queued
                # Get all predecessors that are actually in the graph
                neighbor_prereqs = {p for p in prereq_graph.predecessors(neighbor) if prereq_graph.has_node(p)}
                # Check if *all* prerequisites for this neighbor are now known
                if neighbor_prereqs.issubset(current_known):
                    queue.append(neighbor)
                    words_remaining.remove(neighbor) # Remove from remaining *now*
        # ----------------------

    if pbar: pbar.close()

    if len(optimal_order) != num_words:
        print(f"Warning: Output order ({len(optimal_order)}) doesn't match input size ({num_words}). {len(words_remaining)} words could not be ordered. This may indicate cycles involving only unordered words, or words whose prerequisite sentences failed/were empty.", file=sys.stderr)
        # Optionally: list the remaining words
        # print(f"Unordered words: {words_remaining}")

    print(f"Sorted {len(optimal_order)} words.")
    return optimal_order

# ---------------- 5. Results Processing & Saving ----------------
def save_comparison_csv(
    output_path: str,
    original_list: list[str],
    optimal_list: list[str],
    sentences_map: dict[str, str],
    reading_map: dict[str, str],
    meaning_map: dict[str, str],
    level_map: dict[str, str],
    tokenized_sentences: dict[str, list[str]],
    seed_words: set[str]
):
    """Saves the comparison including cost, reading, meaning, level, tokenization (surface), and unknown tokens."""
    print(f"\n--- Saving Comparison to CSV ({output_path}) ---")

    original_indices = {word: i for i, word in enumerate(original_list)}
    results_data = []
    processed_words_optimal_order = set(optimal_list)

    # --- Calculate costs and unknown tokens for optimally placed words --- 
    word_costs = {}
    unknown_tokens_map = {} # Store the list of unknown tokens (surface forms)
    current_known_for_cost = set(seed_words)
    for word in optimal_list:
        tokens = tokenized_sentences.get(word, []) # Use surface forms for cost calc
        unknown_tokens_at_step = [token for token in tokens if token not in current_known_for_cost]
        cost = len(unknown_tokens_at_step)
        word_costs[word] = cost
        unknown_tokens_map[word] = unknown_tokens_at_step # Store the list of surface forms
        current_known_for_cost.add(word) # Add word after calculating its cost
    # -----------------------------------------------------------------

    # Add words that ARE in the optimal order
    for i, word in enumerate(optimal_list):
        results_data.append({
            "word": word,
            "sentence": sentences_map.get(word, ""),
            "jlpt_level": level_map.get(word, ""),
            "original_index": original_indices.get(word, -1),
            "optimal_index": i,
            "number_of_unknown_tokens": word_costs.get(word, -1),
            "unknown_tokens_in_sentence": ', '.join(unknown_tokens_map.get(word, [])),
            "tokenized_sentence_surface": ', '.join(tokenized_sentences.get(word, [])),
            "reading": reading_map.get(word, ""),
            "meaning": meaning_map.get(word, ""),
        })

    # Add words that were NOT placed in the optimal order
    unprocessed_count = 0
    for i, word in enumerate(original_list):
        if word not in processed_words_optimal_order:
            results_data.append({
                "word": word,
                "sentence": sentences_map.get(word, ""),
                "jlpt_level": level_map.get(word, ""),
                "original_index": i,
                "optimal_index": -1,
                "number_of_unknown_tokens": -1,
                "unknown_tokens_in_sentence": "",
                "tokenized_sentence_surface": ', '.join(tokenized_sentences.get(word, [])),
                "reading": reading_map.get(word, ""),
                "meaning": meaning_map.get(word, ""),
            })
            unprocessed_count += 1

    if unprocessed_count > 0:
        print(f"Note: {unprocessed_count} words were not included in the optimal order and are marked with index -1.")

    try:
        # Define final columns in desired order
        final_columns = [
            'word', 'sentence', 'jlpt_level', 'original_index', 'optimal_index',
            'number_of_unknown_tokens', 'unknown_tokens_in_sentence', 
            'tokenized_sentence_surface', 
            'reading', 'meaning'
        ]
        results_df = pd.DataFrame(results_data, columns=final_columns)

        # Sort primarily by optimal_index (putting -1 at the end), then original_index
        results_df['sort_key'] = results_df['optimal_index'].apply(lambda x: float('inf') if x == -1 else x)
        results_df = results_df.sort_values(by=['sort_key', 'original_index']).drop(columns=['sort_key'])

        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Comparison results saved successfully ({len(results_df)} words).")
    except Exception as e:
        print(f"Error saving CSV file to {output_path}: {e}", file=sys.stderr)

def analyze_order_similarity(original_list: list[str], optimal_list: list[str]):
    """Prints analysis comparing the original and optimal lists."""
    print("\n--- Order Analysis ---")
    N_COMPARE = 30
    print(f"\nFirst {N_COMPARE} words (Original): {original_list[:N_COMPARE]}")
    print(f"First {N_COMPARE} words (Optimal):  {optimal_list[:N_COMPARE]}")

    print("\nWord Movement Examples:")
    sample_indices = [i for i in [0, 10, 50, 100, 200, 500] if i < len(original_list)]
    optimal_index_map = {word: i for i, word in enumerate(optimal_list)}
    for original_idx in sample_indices:
        word = original_list[original_idx]
        optimal_idx = optimal_index_map.get(word, "Not found")
        print(f"- '{word}': Original index {original_idx} -> Optimal index {optimal_idx}")

    if kendalltau:
        common_words = [w for w in original_list if w in optimal_index_map]
        if len(common_words) > 1:
            original_rank_map = {word: i for i, word in enumerate(original_list)}
            original_ranks = [original_rank_map[w] for w in common_words]
            optimal_ranks = [optimal_index_map[w] for w in common_words]
            tau, p_value = kendalltau(original_ranks, optimal_ranks)
            print(f"\nKendall's Tau rank correlation: {tau:.4f}")
        else:
            print("\nNot enough common words (>1) to calculate Kendall's Tau.")
    else:
        print("\nScipy not found. Skipping Kendall's Tau calculation.")

# ---------------- 6. Plotting ----------------
def plot_cognitive_load(optimal_list: list[str], sentences_map: dict[str, str], seed_words: set[str], tokenized_sentences: dict[str, list[str]]):
    """Plots the number of unknown tokens encountered at each step and saves data."""
    print("\n--- Cognitive Load Plot & Data --- ") # Modified title
    if not optimal_list:
        print("No words in optimal order to plot.")
        return

    print("Calculating cognitive load curve...")
    known_plot = set(seed_words)
    costs = []
    for word in optimal_list:
        tokens = tokenized_sentences.get(word, []) # Use pre-tokenized
        cost = sum(token not in known_plot for token in tokens)
        costs.append(cost)
        known_plot.add(word) # Add after cost calculation

    # --- Calculate Cumulative Cognitive Load ---
    cumulative_costs = []
    current_sum = 0
    for cost in costs:
        current_sum += cost
        cumulative_costs.append(current_sum)
    # -----------------------------------------

    # --- Save Cumulative Cognitive Load Data ---
    if cumulative_costs:
        cumulative_df = pd.DataFrame({
            'learning_step': range(1, len(cumulative_costs) + 1),
            'cumulative_cognitive_load': cumulative_costs
        })
        cumulative_csv_filename = "cumulative_cognitive_load_data.csv"
        try:
            print(f"Saving cumulative cognitive load data to {cumulative_csv_filename}...")
            cumulative_df.to_csv(cumulative_csv_filename, index=False)
            print("Cumulative cognitive load data saved successfully.")
        except Exception as e:
            print(f"Error saving cumulative cognitive load data: {e}", file=sys.stderr)
    # -----------------------------------------

    if not plt: # Check if plotting is possible
        print("Matplotlib not found. Skipping plot generation.")
        return

    # --- Plot and Save Per-Step Cognitive Load ---
    plt.figure(figsize=(10, 5))
    plt.plot(costs)
    plt.xlabel("Vocabulary Learning Step")
    plt.ylabel("Number of Unknown Tokens in Sentence")
    plt.title("Cognitive Load per Step (Min-Surprise Order)")
    plt.grid(True)
    plot_filename = "cognitive_load_plot.png"
    try:
        print(f"Saving per-step cognitive load plot to {plot_filename}...")
        plt.savefig(plot_filename)
        print(f"Per-step cognitive load plot saved to {plot_filename}.") # Added confirmation
    except Exception as e:
        print(f"Error saving per-step plot: {e}", file=sys.stderr)
    plt.close() # Close the per-step plot figure
    # -------------------------------------------

    # --- Plot and Save Cumulative Cognitive Load ---
    if cumulative_costs:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(cumulative_costs) + 1), cumulative_costs)
        plt.xlabel("Vocabulary Learning Step")
        plt.ylabel("Cumulative Number of Unknown Tokens")
        plt.title("Cumulative Cognitive Load Curve (Min-Surprise Order)")
        plt.grid(True)
        cumulative_plot_filename = "cumulative_cognitive_load_plot.png"
        try:
            print(f"Saving cumulative cognitive load plot to {cumulative_plot_filename}...")
            plt.savefig(cumulative_plot_filename)
            print(f"Cumulative cognitive load plot saved to {cumulative_plot_filename}.") # Added confirmation
        except Exception as e:
            print(f"Error saving cumulative plot: {e}", file=sys.stderr)
        plt.close() # Close the cumulative plot figure
    # --------------------------------------------

# ---------------- Main Execution ----------------
def main():
    """Main function to orchestrate the vocabulary ordering process."""
    # --- Load Environment Variables from .env file --- 
    if load_dotenv:
        print("Loading environment variables from .env file...")
        load_dotenv() # Load variables from .env into environment
    else:
        print("python-dotenv not found. Cannot load .env file.", file=sys.stderr)
    # ----------------------------------------------------

    # --- Load API Key from Environment Variable --- 
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        print("Please create a .env file with GEMINI_API_KEY=YOUR_KEY", file=sys.stderr)
        sys.exit(1)
    # ---------------------------------------------

    # --- Load Existing Sentences Cache (Conditional Update) ---
    sentence_cache = {}
    if os.path.exists(SENTENCE_CACHE_CSV_PATH): # Use new cache path
        print(f"Loading existing sentences from {SENTENCE_CACHE_CSV_PATH}...")
        try:
            cache_df = pd.read_csv(SENTENCE_CACHE_CSV_PATH)
            # Ensure required columns exist
            if 'word' in cache_df.columns and 'sentence' in cache_df.columns:
                # Fill NaN sentences with empty strings before creating dict
                cache_df['sentence'] = cache_df['sentence'].fillna('')
                sentence_cache = pd.Series(cache_df.sentence.values, index=cache_df.word).to_dict()
                print(f"Loaded {len(sentence_cache)} sentences into cache.")
            else:
                print(f"Warning: CSV file {SENTENCE_CACHE_CSV_PATH} missing required 'word' or 'sentence' columns. Cache not loaded.", file=sys.stderr)
        except Exception as e:
            print(f"Error loading sentence cache from {SENTENCE_CACHE_CSV_PATH}: {e}", file=sys.stderr)
    else:
        print(f"No existing sentence cache file found at {SENTENCE_CACHE_CSV_PATH}.")
    # ---------------------------------------

    # 1. Load Vocabulary and N5 Seed Words
    vocab_df, n5_seed_words = load_vocabulary(BASE_URL, LEVELS, VOCAB_COLUMN) # Capture N5 words to use as seeds
    original_vocab_list = list(vocab_df[VOCAB_COLUMN])

    if not n5_seed_words:
        print("Warning: No N5 words were loaded to be used as seeds. This might be because 'n5.csv' is not in LEVELS or the N5 file was empty/not found. Proceeding with an empty seed set.", file=sys.stderr)

    # Filter original list to exclude N5 seed words
    words_to_actually_order = [w for w in original_vocab_list if w not in n5_seed_words]

    # Create lookup maps for reading, meaning, and level
    reading_map = pd.Series(vocab_df.reading.values, index=vocab_df[VOCAB_COLUMN]).to_dict()
    meaning_map = pd.Series(vocab_df.meaning.values, index=vocab_df[VOCAB_COLUMN]).to_dict()
    level_map = pd.Series(vocab_df.jlpt_level.values, index=vocab_df[VOCAB_COLUMN]).to_dict() # Add level map

    # 2. Generate Sentences
    # generator = NaiveSentenceGenerator()
    generator = LLMSentenceGenerator(api_key=api_key)

    if isinstance(generator, LLMSentenceGenerator) and not generator.model:
         print("LLM Generator failed to initialize. Exiting.", file=sys.stderr)
         sys.exit(1)

    sentences = generate_all_sentences(words_to_actually_order, generator, sentence_cache)

    # --- Save Updated Sentences to Cache (Conditional Update) ---
    newly_generated_or_updated_items = []
    for word, current_sentence in sentences.items():
        if current_sentence: # Only consider non-empty generated sentences
            # If word wasn't in initial cache, or was in cache but empty, it's new/updated
            if word not in sentence_cache or not sentence_cache.get(word):
                newly_generated_or_updated_items.append({'word': word, 'sentence': current_sentence})

    if newly_generated_or_updated_items:
        print(f"Found {len(newly_generated_or_updated_items)} new/updated sentences to add to cache at {SENTENCE_CACHE_CSV_PATH}...")
        new_sentences_df = pd.DataFrame(newly_generated_or_updated_items)
        
        final_df_to_save = new_sentences_df
        if os.path.exists(SENTENCE_CACHE_CSV_PATH):
            try:
                existing_cache_df = pd.read_csv(SENTENCE_CACHE_CSV_PATH)
                # Ensure columns are consistent for concatenation, especially if existing_cache_df is empty
                if not existing_cache_df.empty and not all(col in existing_cache_df.columns for col in ['word', 'sentence']):
                    print(f"Warning: Existing cache file {SENTENCE_CACHE_CSV_PATH} has unexpected columns. Overwriting with new sentences.", file=sys.stderr)
                elif not existing_cache_df.empty:
                    combined_df = pd.concat([existing_cache_df, new_sentences_df], ignore_index=True)
                    final_df_to_save = combined_df.drop_duplicates(subset=['word'], keep='last').reset_index(drop=True)
                # If existing_cache_df is empty or had wrong columns, final_df_to_save is already new_sentences_df
            except pd.errors.EmptyDataError:
                print(f"Existing cache file {SENTENCE_CACHE_CSV_PATH} is empty. Initializing with new sentences.")
                # final_df_to_save is already new_sentences_df
            except Exception as e:
                print(f"Error reading existing cache file {SENTENCE_CACHE_CSV_PATH}: {e}. Overwriting with new sentences.", file=sys.stderr)
                # final_df_to_save is already new_sentences_df

        try:
            final_df_to_save.to_csv(SENTENCE_CACHE_CSV_PATH, index=False, encoding='utf-8-sig')
            print(f"Saved {len(final_df_to_save)} total sentences to cache at {SENTENCE_CACHE_CSV_PATH}.")
        except Exception as e:
            print(f"Error saving sentence cache to {SENTENCE_CACHE_CSV_PATH}: {e}", file=sys.stderr)
    else:
        print(f"Sentence cache at {SENTENCE_CACHE_CSV_PATH} is already up-to-date. No new sentences generated.")
    # ----------------------------------------------------------------

    # --- Pre-tokenize all sentences (surface forms only) --- 
    print("Tokenizing all generated sentences...")
    tokenized_sentences = {} # word -> [surface1, surface2, ...]
    # tagged_sentences = {}    # word -> [(surface1, pos1), (surface2, pos2), ...] # REMOVED
    # Use tqdm directly on items if available
    items_iterator = tqdm(sentences.items()) if tqdm else sentences.items()
    for word, sent in items_iterator:
        if sent:
            try:
                # Only need surface forms now
                tokenized_sentences[word] = tokenize(sent)
                # tagged_sentences[word] = [(token.surface, token.pos) for token in tagger(sent)] # REMOVED
            except Exception as e:
                 print(f"Error tokenizing sentence for '{word}': {sent}. Error: {e}", file=sys.stderr)
                 tokenized_sentences[word] = []
                 # tagged_sentences[word] = [] # REMOVED
        else:
            tokenized_sentences[word] = []
            # tagged_sentences[word] = [] # REMOVED
    print(f"Tokenized {len(tokenized_sentences)} sentences.")
    # ------------------------------------------------------

    # 3. Build Dependency Graph (pass only words_to_order, N5 seeds, and their tokenized sentences)
    dependency_graph = build_dependency_graph(words_to_actually_order, n5_seed_words, tokenized_sentences)

    # 4. Find Optimal Order (pass tokenized surface forms, N5 seeds, and original index map)
    # Note: words_to_actually_order ensures we don't try to order the N5 seeds themselves
    original_index_map = {word: i for i, word in enumerate(original_vocab_list)}
    optimal_vocab_list = find_min_surprise_order(words_to_actually_order, sentences, dependency_graph, n5_seed_words, tokenized_sentences, original_index_map)

    # 5. Analyze and Save Results
    # Pass the original list for reporting, but optimal list excludes N5 seeds
    analyze_order_similarity(original_vocab_list, optimal_vocab_list) # Compare against original
    save_comparison_csv(
        OUTPUT_CSV_PATH,
        original_vocab_list, # Full original list for context
        optimal_vocab_list,  # The ordered list (excluding N5 seeds)
        sentences,
        reading_map,
        meaning_map,
        level_map,
        tokenized_sentences,
        n5_seed_words # Pass N5 seeds
    )

    # 6. Plot (Optional) (pass tokenized surface forms and N5 seeds)
    plot_cognitive_load(optimal_vocab_list, sentences, n5_seed_words, tokenized_sentences)

if __name__ == "__main__":
    main() 