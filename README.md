# JLPT Vocabulary Minimum Surprise Ordering

## Project Goal

This project aims to find an optimized learning order for Japanese Language Proficiency Test (JLPT) N5 and N4 vocabulary. The optimization strategy is based on a "minimum surprise" principle: at each step, introduce the word whose example sentence contains the fewest *unknown* tokens, leveraging previously learned vocabulary. The goal is to create a smoother learning curve by minimizing cognitive load.

## Process Overview

1.  **Fetch Vocabulary:** Download raw JLPT N5 and N4 vocabulary lists (CSV format) from the [elzup/jlpt-word-list](https://github.com/elzup/jlpt-word-list) GitHub repository.
2.  **Generate Sentences:** For each vocabulary word, automatically generate a simple example sentence. (Current implementation uses basic templates).
3.  **Build Dependency Graph:** Construct a directed acyclic graph (DAG) where nodes are vocabulary words (plus a set of predefined `SEED_WORDS`). An edge exists from word `A` to word `B` if word `A` appears in the example sentence generated for word `B`.
4.  **Topological Sort (Min-Surprise):** Perform a modified topological sort on the graph. Prioritize nodes (words) with an in-degree of zero (relative to other vocabulary words). Among these candidates, select the word whose sentence introduces the minimum number of tokens not yet encountered (i.e., not in `SEED_WORDS` or previously selected words).
5.  **Output Results:**
    *   Generate a CSV file (`vocab_order_comparison.csv`) comparing the original vocabulary order with the newly calculated optimal order. This file includes the word, its generated sentence, its original index, and its optimal index.
    *   Print key statistics and comparisons to the console.
    *   Optionally, display a plot visualizing the "cognitive load" (number of unknown tokens per sentence) at each step of the optimal order using `matplotlib`.

## Minimum Surprise Sorting Algorithm

The core of the optimization lies in a greedy algorithm that aims to minimize the introduction of unknown words at each step:

1.  **Initialization:**
    *   Start with a set of `known_words` containing only the predefined `SEED_WORDS`.
    *   Identify an initial `queue` of candidate vocabulary words. These are words whose generated sentences *only* contain dependencies found within the `SEED_WORDS` (i.e., they have an in-degree of 0 when considering dependencies *only* from other vocabulary words, not seeds).
2.  **Iteration:**
    *   While the `queue` of candidate words is not empty:
        *   **Calculate Cost:** For each `word` in the `queue`, calculate its "surprise cost". This cost is the number of tokens in the `word`'s generated sentence that are *not* currently in the `known_words` set.
        *   **Select Best:** Choose the `best_word` from the `queue` that has the minimum surprise cost. (Tie-breaking defaults to the order words appeared in the queue).
        *   **Update:**
            *   Remove `best_word` from the `queue`.
            *   Append `best_word` to the `optimal_order` list.
            *   Add `best_word` to the `known_words` set.
            *   **Unlock Neighbors:** For any other vocabulary word (`neighbor`) that had `best_word` as a prerequisite, decrease its count of unmet prerequisites. If a `neighbor`'s count reaches zero, add it to the `queue` as it is now available to be selected.
3.  **Termination:** The loop continues until the `queue` is empty, meaning no more words can be placed according to the dependency rules.

This process ensures that, at each step, the word chosen is the one predicted to be easiest to understand based on the vocabulary learned so far.

## Current Implementation

*   **Script:** `jlpt_order_min_surprise.py`
*   **Language:** Python 3
*   **Core Libraries:**
    *   `pandas`: Data manipulation (loading CSVs, creating DataFrames).
    *   `requests`: Fetching vocabulary lists from URLs.
    *   `fugashi` / `unidic-lite`: Japanese word tokenization.
    *   `networkx`: Building and analyzing the dependency graph.
*   **Optional Libraries:**
    *   `matplotlib`: Plotting the cognitive load curve.
    *   `scipy`: Calculating Kendall's Tau rank correlation for order comparison.
    *   `tqdm`: Displaying progress bars for long operations.
*   **Structure:** The code is refactored into functions and classes for modularity (loading, sentence generation, graph building, sorting, analysis, saving, plotting) orchestrated by a `main()` function.
*   **Sentence Generation:** Currently uses a `NaiveSentenceGenerator` class with very basic templates based on word endings (い-adjectives, る/う-verbs, nouns).
*   **Output File:** `vocab_order_comparison.csv`

## Findings & Observations

*   The script successfully generates an alternative vocabulary order aiming to minimize surprise.
*   **Tie-Breaking Behavior:** When multiple words have the same minimum cost (number of new tokens in their sentence), the default `min()` function behavior leads to selection based on the order in the queue. This results in clusters of similar words (e.g., い-adjectives) appearing in Japanese alphabetical order in the initial part of the "optimal" list, as they often share the same low cost initially.
*   **Rank Correlation:** Kendall's Tau rank correlation between the original and optimal order is typically around 0.7, indicating a significant reordering but retaining some similarity to the original sequence.
*   **Incomplete Ordering:** The script currently prints a warning as not all words from the input lists (~222 out of ~1373) are included in the final `optimal_order`. These words are marked with an `optimal_index` of -1 in the output CSV. This is likely due to cycles in the dependency graph (which can happen with simple templates) or words whose sentences contain *only* tokens not reachable from the `SEED_WORDS` via the generated dependencies (disconnected components).

## Potential Future Directions

*   **LLM Sentence Generation:** Implement an `LLMSentenceGenerator` class (inheriting from `SentenceGenerator`) to create more realistic and varied example sentences, potentially leading to more nuanced dependency costs and better ordering.
*   **Improved Tie-Breaking:** Modify the `find_min_surprise_order` function to use a secondary key for tie-breaking when costs are equal (e.g., prioritize words that unlock more future words, or use a random shuffle).
*   **Investigate Disconnected Components/Cycles:** Analyze the dependency graph further to understand why some words are not included in the final order and refine the `SEED_WORDS` or sentence generation logic accordingly.
*   **Refine SEED_WORDS:** Expand the `SEED_WORDS` set with more common grammatical components or high-frequency words.
*   **POS Tagging for Templates:** Enhance the `NaiveSentenceGenerator` to use Part-of-Speech tagging (available via `fugashi`) to select more appropriate sentence templates.
*   **Configuration:** Move settings like `LEVELS`, `SEED_WORDS`, etc., to a separate configuration file or command-line arguments.