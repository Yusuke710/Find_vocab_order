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
from functools import partial
import itertools # Added for SCC processing

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
LEVELS = ["n5.csv", "n4.csv", "n3.csv", "n2.csv", "n1.csv"] # Add more levels if needed
SEED_WORDS_PATH = "seed_words.txt" # Path to the seed words file
OUTPUT_CSV_PATH = "vocab_order_comparison_scc.csv" # Changed output filename
CACHE_READ_PATH = "vocab_order_comparison.csv" # Read from original cache if exists

# Initialize tokenizer globally (or pass it around if preferred)
tagger = fugashi.Tagger()

def tokenize(text: str) -> list[str]:
    """Tokenizes Japanese text using fugashi."""
    # Consider adding error handling or filtering (e.g., for punctuation)
    return [token.surface for token in tagger(text)]

# --- New Function to Load Seed Words ---
def load_seed_words(filepath: str) -> set[str]:
    """Loads seed words from a file, one word per line."""
    seed_words = set()
    print(f"Loading seed words from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'): # Ignore empty lines and comments
                    seed_words.add(word)
        print(f"Loaded {len(seed_words)} seed words.")
    except FileNotFoundError:
        print(f"Warning: Seed words file not found at {filepath}. Using an empty set.", file=sys.stderr)
        # Optionally, return a minimal default set here if preferred
        # return {'は', 'が', 'です'}
    except Exception as e:
        print(f"Error loading seed words from {filepath}: {e}", file=sys.stderr)
        print("Using an empty set of seed words.", file=sys.stderr)
    return seed_words

# Helper function to calculate cost
def calculate_word_cost(word: str, known_set: set[str], sentences_map: dict[str, str], tokenized_sentences: dict[str, list[str]]) -> int:
    """Calculates the 'surprise' cost of a word given a set of known tokens."""
    tokens = tokenized_sentences.get(word)
    if tokens is None: # Sentence might be missing or empty
        return float('inf') # Assign high cost if sentence/tokens are missing
    return sum(token not in known_set for token in tokens)

# ---------------- 1. Data Loading ----------------
def load_vocabulary(base_url: str, levels: list[str], column: str) -> pd.DataFrame:
    """Loads vocabulary lists from URLs, concatenates, and removes duplicates."""
    print(f"Loading vocabulary from {len(levels)} levels...")
    dfs = []
    for level in levels:
        url = base_url + level
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            dfs.append(pd.read_csv(io.StringIO(response.text)))
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}", file=sys.stderr)
        except pd.errors.ParserError as e:
            print(f"Error parsing {url}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred processing {url}: {e}", file=sys.stderr)

    if not dfs:
        print("Error: No vocabulary data loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    vocab_df = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=[column])
        .reset_index(drop=True)
    )
    print(f"Loaded {len(vocab_df)} unique words.")
    return vocab_df

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
            # Suppress API key warning if using env var
            genai.configure(api_key=api_key, transport='rest')
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

            # Robust text extraction
            generated_sentence = ""
            if response.parts:
                 # Check candidates for safety reasons
                if response.candidates and response.candidates[0].finish_reason == 'SAFETY':
                     print(f"Warning: LLM response for '{word}' blocked due to SAFETY. Prompt: {prompt}.", file=sys.stderr)
                elif response.candidates and response.candidates[0].finish_reason == 'OTHER':
                     print(f"Warning: LLM response for '{word}' stopped for OTHER reason. Prompt: {prompt}.", file=sys.stderr)
                else:
                     generated_sentence = response.text # Assuming .text gives combined parts

            else: # Handle cases where response might be empty or blocked without parts
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    print(f"Warning: LLM response for '{word}' was empty or blocked. Prompt: {prompt}. Reason: {response.prompt_feedback.block_reason}", file=sys.stderr)
                else:
                    print(f"Warning: LLM response for '{word}' was empty for unknown reason. Prompt: {prompt}.", file=sys.stderr)

            # Basic post-processing (optional)
            generated_sentence = generated_sentence.replace("\n", " ").strip()
            # Remove potential markdown like backticks if the model adds them
            generated_sentence = generated_sentence.strip("`")

            # Optional: Add a small delay to avoid hitting rate limits if making many calls rapidly
            # import time
            # time.sleep(0.5) 

            return generated_sentence

        except Exception as e:
            # Catch more specific errors if possible (e.g., google.api_core.exceptions.GoogleAPIError)
            print(f"Error during LLM API call for '{word}': {e}", file=sys.stderr)
            # Consider more specific error handling based on potential API errors
            return ""

def generate_all_sentences(vocab_list: list[str], generator: SentenceGenerator, cache: dict[str, str]) -> dict[str, str]:
    """Generates a sentence for each word, using cache if available."""
    print(f"Generating sentences for {len(vocab_list)} words (using cache where possible)...")
    sentences = {}
    cached_count = 0
    generated_count = 0
    failed_count = 0
    iterator = tqdm(vocab_list) if tqdm else vocab_list

    for word in iterator:
        # Check cache first
        cached_sentence = cache.get(word)
        # Use cache only if it's not None and not an empty string
        if cached_sentence: # Check if sentence exists and is not empty
            sentences[word] = cached_sentence
            cached_count += 1
        else:
            # If not in cache or sentence is empty, generate anew
            try:
                generated = generator.generate(word)
                if generated: # Check if LLM returned a non-empty sentence
                    sentences[word] = generated
                    generated_count += 1
                else:
                    # LLM returned empty or failed internally, but no exception
                    sentences[word] = "" # Store empty string to avoid regenerating
                    failed_count += 1
            except Exception as e:
                print(f"Error generating sentence for '{word}': {e}", file=sys.stderr)
                sentences[word] = "" # Store empty string on exception
                failed_count += 1

    print(f"Sentence generation complete: {cached_count} loaded, {generated_count} generated, {failed_count} failed/empty.")
    return sentences

# ---------------- 3. Dependency Graph Building ----------------
def build_dependency_graph(words: list[str], sentences_map: dict[str, str], seed_words: set[str], tokenized_sentences: dict[str, list[str]]) -> nx.DiGraph:
    """Builds a directed graph where edges represent word prerequisites."""
    print("Building dependency graph...")
    graph = nx.DiGraph()
    all_potential_nodes = set(words).union(seed_words)
    graph.add_nodes_from(all_potential_nodes) # Ensure all words are nodes, even if disconnected

    iterator = tqdm(words) if tqdm else words
    for word in iterator:
        tokens = tokenized_sentences.get(word)
        if not tokens: continue # Skip if no tokens (empty/failed sentence)

        # An edge u -> word means u must be known before word
        for token in tokens:
            # Only add edge if the token is a known seed word OR another vocab word
            if token in all_potential_nodes and token != word:
                 # Check if the token is actually in our list of nodes (it should be)
                 if graph.has_node(token):
                    graph.add_edge(token, word)

    # Remove self-loops if any were accidentally created (though logic above prevents it)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

# ---------------- 4. Optimal Ordering Algorithm (SCC Version) ----------------

def build_condensed_graph(graph: nx.DiGraph, vocab_words: set[str]) -> tuple[nx.DiGraph, dict, dict]:
    """
    Builds a condensed graph where SCCs are single nodes.

    Returns:
        - Condensed DiGraph
        - scc_map: Maps original word -> SCC tuple identifier (or the word itself if size 1)
        - component_nodes: Maps SCC tuple identifier -> list of words in that SCC
    """
    print("Finding strongly connected components...")
    sccs = list(nx.strongly_connected_components(graph))
    print(f"Found {len(sccs)} strongly connected components.")

    # Map original node to its SCC identifier (tuple for multi-node SCCs, word itself for singletons)
    scc_map = {}
    component_nodes = {}
    scc_identifiers = []

    multi_node_scc_count = 0
    for component in sccs:
        # Use frozenset for hashable representation of the component
        component_set = frozenset(component)
        if len(component_set) > 1:
            # Filter to only include vocab words in the SCC identifier if needed
            vocab_in_component = tuple(sorted([word for word in component_set if word in vocab_words]))
            if not vocab_in_component: continue # Skip if SCC contains only non-vocab (e.g., seed) words

            scc_id = vocab_in_component # Use tuple of vocab words as ID
            component_nodes[scc_id] = list(vocab_in_component) # Store list of words
            scc_identifiers.append(scc_id)
            multi_node_scc_count += 1
            for node in component: # Map all original nodes in SCC
                 if node in vocab_words: scc_map[node] = scc_id
        else:
            # Singleton component
            node = next(iter(component_set)) # Get the single node
            if node in vocab_words: # Only add vocab words as nodes in condensed graph
                scc_id = node # Use the word itself as ID
                component_nodes[scc_id] = [node]
                scc_map[node] = scc_id
                scc_identifiers.append(scc_id)

    print(f"Identified {multi_node_scc_count} multi-node SCCs involving vocabulary words.")

    print("Building condensed graph...")
    condensed_graph = nx.DiGraph()
    condensed_graph.add_nodes_from(scc_identifiers)

    # Add edges between SCCs/nodes in the condensed graph
    for u, v in graph.edges():
         # Only consider edges where the target is a vocab word
         if v not in vocab_words: continue
         
         u_scc = scc_map.get(u) # Source component ID (could be a seed word or vocab)
         v_scc = scc_map.get(v) # Target component ID (must be vocab)

         # Add edge if both nodes are in the condensed graph mapping
         # and the edge is between *different* components
         if u_scc is not None and v_scc is not None and u_scc != v_scc:
             # Ensure nodes exist in condensed graph before adding edge
             if condensed_graph.has_node(u_scc) and condensed_graph.has_node(v_scc):
                 condensed_graph.add_edge(u_scc, v_scc)

    print(f"Condensed graph built with {condensed_graph.number_of_nodes()} nodes (SCCs/single words) and {condensed_graph.number_of_edges()} edges.")
    return condensed_graph, scc_map, component_nodes


def find_min_surprise_order_scc(
    condensed_graph: nx.DiGraph,
    component_nodes: dict,
    sentences_map: dict[str, str],
    seed_vocab: set[str],
    tokenized_sentences: dict[str, list[str]]
) -> list[str]:
    """
    Performs topological sort on the condensed graph (SCCs as nodes),
    prioritizing nodes/SCCs with the minimum combined surprise cost.
    """
    print("Sorting condensed graph by minimum surprise...")
    optimal_order = []
    current_known = set(seed_vocab)

    # Calculate in-degrees for the condensed graph nodes
    in_degree = defaultdict(int, {node: deg for node, deg in condensed_graph.in_degree()})
    adj_list = defaultdict(list, {node: list(neighbors) for node, neighbors in condensed_graph.adjacency()})

    # Initial queue: nodes in the condensed graph with in-degree 0
    # These are SCCs or single words whose prerequisites (if any) are outside the vocab set (e.g., seed words)
    # or within the same SCC (handled by condensation).
    queue = deque([node for node in condensed_graph.nodes() if in_degree[node] == 0])

    total_nodes_to_process = condensed_graph.number_of_nodes()
    pbar = tqdm(total=total_nodes_to_process) if tqdm else None
    processed_nodes_count = 0

    while queue:
        if not queue:
             # This might happen if the graph has parts not reachable from seeds,
             # even after condensation. We might need to handle disconnected components later if this occurs.
             print("Warning: Queue empty, but not all nodes processed. Possible disconnected components.", file=sys.stderr)
             break

        # Calculate costs for all components currently in the queue
        component_costs = {}
        for component_id in queue:
            cost = 0
            words_in_component = component_nodes[component_id]
            for word in words_in_component:
                 # Use the helper function, passing the pre-tokenized sentences
                 cost += calculate_word_cost(word, current_known, sentences_map, tokenized_sentences)
            component_costs[component_id] = cost

        # Find the component in the queue with the minimum total cost
        best_component_id = min(queue, key=lambda comp_id: component_costs[comp_id])
        best_cost = component_costs[best_component_id]

        queue.remove(best_component_id)
        processed_nodes_count += 1

        # Add words from the selected component to the optimal order
        # Order *within* the component: simple alphabetical sort for now
        words_to_add = sorted(component_nodes[best_component_id])
        optimal_order.extend(words_to_add)
        current_known.update(words_to_add)

        # Print info about the step
        component_repr = f"'{best_component_id}'" if isinstance(best_component_id, str) else f"SCC({len(words_to_add)} words)"
        # --- Reduced Verbosity Print Statement ---
        # Limit printing to every N steps or based on time to avoid flooding logs
        if processed_nodes_count % 50 == 0 or processed_nodes_count == 1 or processed_nodes_count == total_nodes_to_process :
             print(f"  - Step {processed_nodes_count}/{total_nodes_to_process}: Added {component_repr} (Cost: {best_cost}). Known: {len(current_known)}")
        # ---------------------------------------


        if pbar: pbar.update(1)

        # Update in-degrees of neighbors in the condensed graph
        for neighbor_id in adj_list[best_component_id]:
            in_degree[neighbor_id] -= 1
            if in_degree[neighbor_id] == 0:
                queue.append(neighbor_id)

    if pbar: pbar.close()

    # Check if all components were processed
    # Note: We check against processed_nodes_count, not len(optimal_order) directly
    if processed_nodes_count != total_nodes_to_process:
        num_missing = total_nodes_to_process - processed_nodes_count
        print(f"Warning: Processed {processed_nodes_count}/{total_nodes_to_process} components. {num_missing} components might be unreachable.", file=sys.stderr)
        
        # --- Handle Unreachable Components ---
        print(f"Attempting to add {num_missing} unreachable components/words...")
        unreachable_nodes = [node for node in condensed_graph.nodes() if node not in current_known and in_degree[node] > 0] 
        # Calculate cost for each unreachable node based on the final known set
        unreachable_costs = []
        for component_id in unreachable_nodes:
            cost = 0
            words_in_component = component_nodes.get(component_id, []) # Use .get for safety
            for word in words_in_component:
                cost += calculate_word_cost(word, current_known, sentences_map, tokenized_sentences)
            unreachable_costs.append((component_id, cost))
            
        # Sort unreachable components by cost
        sorted_unreachable = sorted(unreachable_costs, key=lambda item: item[1])
        
        # Append words from unreachable components to the optimal order
        for component_id, cost in sorted_unreachable:
             words_to_add = sorted(component_nodes.get(component_id, []))
             optimal_order.extend(words_to_add)
             current_known.update(words_to_add) # Keep known set updated
             print(f"  - Added unreachable component {component_repr} (Cost: {cost})")
        print(f"Added words from {len(sorted_unreachable)} unreachable components.")
        # -------------------------------------


    print(f"Sorted {len(optimal_order)} words via condensed graph.")
    return optimal_order


# ---------------- 5. Results Processing & Saving (Unchanged, uses final optimal_list) ----------------
def save_comparison_csv(output_path: str, original_list: list[str], optimal_list: list[str], sentences_map: dict[str, str]):
    """Saves the original vs final optimal order comparison to a CSV file."""
    print(f"\n--- Saving Comparison to CSV ({output_path}) ---")

    original_indices = {word: i for i, word in enumerate(original_list)}
    optimal_indices = {word: i for i, word in enumerate(optimal_list)}
    results_data = []

    # Ensure all words from original list are accounted for, even if ordering failed for some
    processed_optimal = set(optimal_list)
    missing_words = [w for w in original_list if w not in processed_optimal]

    word_count = 0
    for i, word in enumerate(optimal_list):
        results_data.append({
            "word": word,
            "sentence": sentences_map.get(word, ""),
            "original_index": original_indices.get(word, -1),
            "optimal_index": i
        })
        word_count += 1
    
    # Add any missing words at the end with optimal_index = -1
    if missing_words:
        print(f"Warning: {len(missing_words)} words from original list were not placed in the optimal order. Appending to CSV.", file=sys.stderr)
        for word in missing_words:
             results_data.append({
                "word": word,
                "sentence": sentences_map.get(word, ""),
                "original_index": original_indices.get(word, -1),
                "optimal_index": -1 # Indicate they weren't placed by the sort
            })


    try:
        results_df = pd.DataFrame(results_data, columns=['word', 'sentence', 'original_index', 'optimal_index'])
        # Sort by optimal index, placing -1 (missing) at the end
        results_df['sort_key'] = results_df['optimal_index'].apply(lambda x: float('inf') if x == -1 else x)
        results_df = results_df.sort_values(by=['sort_key', 'original_index']).drop(columns=['sort_key']).reset_index(drop=True) 
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Comparison results saved successfully ({len(results_df)} words).")
    except Exception as e:
        print(f"Error saving CSV file to {output_path}: {e}", file=sys.stderr)


def analyze_order_similarity(original_list: list[str], optimal_list: list[str]):
    """Prints analysis comparing the original and final optimal lists."""
    print("\n--- Order Analysis ---")
    if not optimal_list:
         print("Optimal list is empty, cannot perform analysis.")
         return
         
    N_COMPARE = 30
    print(f"\nFirst {N_COMPARE} words (Original): {original_list[:N_COMPARE]}")
    print(f"First {N_COMPARE} words (Optimal):  {optimal_list[:N_COMPARE]}")

    print("\nWord Movement Examples:")
    sample_indices = [i for i in [0, 10, 50, 100, 200, 500, 1000, 2000] if i < len(original_list)]
    optimal_index_map = {word: i for i, word in enumerate(optimal_list)}

    not_found_count = 0
    for original_idx in sample_indices:
        word = original_list[original_idx]
        optimal_idx = optimal_index_map.get(word)
        if optimal_idx is not None:
            print(f"- '{word}': Original index {original_idx} -> Final index {optimal_idx}")
        else:
            print(f"- '{word}': Original index {original_idx} -> Not found in final optimal list.")
            not_found_count += 1
    
    if not_found_count > 0:
         print(f"({not_found_count} sample words were not found in the final list)")


    if kendalltau:
        # Filter original list to only words present in the optimal list for fair comparison
        common_words = [w for w in original_list if w in optimal_index_map]

        if len(common_words) > 1:
            original_rank_map = {word: i for i, word in enumerate(original_list)}

            # Get ranks based on the common words only
            original_ranks = [original_rank_map[word] for word in common_words]
            optimal_ranks = [optimal_index_map[word] for word in common_words]

            try:
                 tau, p_value = kendalltau(original_ranks, optimal_ranks)
                 print(f"\nKendall's Tau rank correlation (comparing {len(common_words)} common words): {tau:.4f}")
            except ValueError as e:
                 print(f"\nCould not calculate Kendall's Tau. Error: {e}")
            except Exception as e:
                 print(f"\nAn unexpected error occurred during Kendall's Tau calculation: {e}")
        else:
            print("\nCould not calculate Kendall's Tau. Not enough common words or list too short.")
    else:
        print("\nScipy not found. Skipping Kendall's Tau calculation.")


# ---------------- 6. Plotting (Modified to also save data) ----------------
def plot_cognitive_load(optimal_list: list[str], sentences_map: dict[str, str], seed_words: set[str], tokenized_sentences: dict[str, list[str]]):
    """
    Calculates cognitive load, plots it, and saves the raw data to a CSV file.
    """
    print("\n--- Cognitive Load Analysis ---")
    if not optimal_list:
        print("No words in optimal order to analyze.")
        return

    print("Calculating cognitive load curve...")
    known_plot = set(seed_words)
    costs = []
    # Use the pre-tokenized sentences for efficiency
    cost_calculator_plot = partial(calculate_word_cost, known_set=known_plot, sentences_map=sentences_map, tokenized_sentences=tokenized_sentences)

    # Corrected loop: calculate cost BEFORE adding word to known_plot
    for word in optimal_list: # Iterate through the final ordered list
        # Calculate cost based on the known set BEFORE this word is added
        cost = calculate_word_cost(word, known_plot, sentences_map, tokenized_sentences)
        costs.append(cost if cost != float('inf') else -1) # Append cost (use -1 for inf/missing sentence)
        # Add the word to known set *after* calculating its cost
        known_plot.add(word) 

    # --- Generate Plot Image ---
    if plt:
        plt.figure(figsize=(12, 6))
        # Plot only non-negative costs (ignore the -1 placeholders)
        steps_with_valid_cost = [i for i, c in enumerate(costs) if c >= 0]
        valid_costs = [c for c in costs if c >= 0]
        if steps_with_valid_cost: # Ensure there's data to plot
             plt.plot(steps_with_valid_cost, valid_costs)
        plt.xlabel("Vocabulary Learning Step")
        plt.ylabel("Number of Unknown Tokens in Sentence")
        plt.title("Cognitive Load per Step (SCC Min-Surprise Order)")
        plt.grid(True)
        plot_filename = "cognitive_load_plot_scc.png" # Changed plot filename
        print(f"Saving cognitive load plot to {plot_filename}...")
        try:
            plt.savefig(plot_filename)
            plt.close() # Close the plot figure to free memory
        except Exception as e:
            print(f"Error saving plot: {e}", file=sys.stderr)
    else:
        print("Matplotlib not found. Skipping plot generation.")

    # --- Generate and Save Data String ---
    data_filename = "cognitive_load_data_scc.csv"
    print(f"Saving cognitive load data to {data_filename}...")
    try:
        with open(data_filename, 'w', encoding='utf-8') as f:
            f.write("step,cost\n") # Write header
            for i, cost_value in enumerate(costs):
                f.write(f"{i},{cost_value}\n") # Write step and cost
        print("Cognitive load data saved successfully.")
    except Exception as e:
        print(f"Error saving cognitive load data: {e}", file=sys.stderr)


# ---------------- Main Execution (Modified for SCC) ----------------
def main():
    """Main function to orchestrate the vocabulary ordering process using SCCs."""
    if load_dotenv:
        print("Loading environment variables from .env file...")
        load_dotenv()
    else:
        print("python-dotenv not found. Cannot load .env file.", file=sys.stderr)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # --- Load Seed Words from File ---
    seed_vocab = load_seed_words(SEED_WORDS_PATH)
    if not seed_vocab:
         print("Warning: No seed words loaded. The process might struggle to start.", file=sys.stderr)
         # Consider exiting if seed words are crucial and none were loaded:
         # sys.exit(1)
    # ---------------------------------

    # --- Load Existing Sentences Cache ---
    sentence_cache = {}
    cache_file_to_read = CACHE_READ_PATH if os.path.exists(CACHE_READ_PATH) else OUTPUT_CSV_PATH
    if os.path.exists(cache_file_to_read):
        print(f"Loading existing sentences from {cache_file_to_read}...")
        try:
            cache_df = pd.read_csv(cache_file_to_read)
            if 'word' in cache_df.columns and 'sentence' in cache_df.columns:
                cache_df['sentence'] = cache_df['sentence'].fillna('')
                sentence_cache = pd.Series(cache_df.sentence.values, index=cache_df.word).to_dict()
                print(f"Loaded {len(sentence_cache)} sentences into cache.")
            else:
                print(f"Warning: CSV file {cache_file_to_read} missing 'word' or 'sentence' columns. Cache not loaded.", file=sys.stderr)
        except Exception as e:
            print(f"Error loading sentence cache from {cache_file_to_read}: {e}", file=sys.stderr)
    else:
        print(f"No existing sentence cache file found ({CACHE_READ_PATH} or {OUTPUT_CSV_PATH}).")
    # ---------------------------------------

    # 1. Load Vocabulary
    vocab_df = load_vocabulary(BASE_URL, LEVELS, VOCAB_COLUMN)
    original_vocab_list = list(vocab_df[VOCAB_COLUMN])
    vocab_set = set(original_vocab_list) # Use set for faster lookups

    # 2. Generate/Load Sentences
    generator = LLMSentenceGenerator(api_key=api_key)
    if isinstance(generator, LLMSentenceGenerator) and not generator.model:
         print("LLM Generator failed to initialize. Exiting.", file=sys.stderr)
         sys.exit(1)
    sentences = generate_all_sentences(original_vocab_list, generator, sentence_cache)

    # Pre-tokenize all sentences for efficiency
    print("Tokenizing all sentences...")
    tokenized_sentences = {word: tokenize(sent) for word, sent in tqdm(sentences.items()) if sent}
    print(f"Tokenized {len(tokenized_sentences)} sentences.")

    # 3. Build Full Dependency Graph (Pass loaded seed_vocab)
    dependency_graph = build_dependency_graph(original_vocab_list, sentences, seed_vocab, tokenized_sentences)

    # 4. Build Condensed Graph (SCCs as nodes)
    condensed_graph, scc_map, component_nodes = build_condensed_graph(dependency_graph, vocab_set)

    # 5. Find Optimal Order using SCC Condensed Graph (Pass loaded seed_vocab)
    final_optimal_list = find_min_surprise_order_scc(
        condensed_graph,
        component_nodes,
        sentences,
        seed_vocab, # Pass the loaded set
        tokenized_sentences
    )

    # 6. Analyze and Save Results
    analyze_order_similarity(original_vocab_list, final_optimal_list)
    save_comparison_csv(OUTPUT_CSV_PATH, original_vocab_list, final_optimal_list, sentences)

    # 7. Plot (Optional) (Pass loaded seed_vocab)
    plot_cognitive_load(final_optimal_list, sentences, seed_vocab, tokenized_sentences)

if __name__ == "__main__":
    main() 