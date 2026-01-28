"""
Optimized Trie Data Structure for Real-Time Search Suggestions
Author: DSA Project
Purpose: Efficient prefix-based search with frequency ranking

Time Complexity Summary:
- Insert: O(m) where m = word length
- Search: O(p + k*log(k)) where p = prefix length, k = top results
- Update: O(m + log(k)) where m = word length, k = heap size
- Space: O(ALPHABET_SIZE * N * M) where N = words, M = avg length
"""

import heapq
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SearchResult:
    """
    Data class to store search results with metadata.
    
    Attributes:
        word (str): The suggested word
        frequency (int): Number of times this word has been selected
        rank (int): Ranking position (1-indexed)
        color (str): Color code based on popularity (green/yellow/orange)
    """
    word: str
    frequency: int
    rank: int
    color: str


class TrieNode:
    """
    Represents a single node in the Trie data structure.
    
    Each node stores:
    - Children mapping (character -> TrieNode)
    - Word completion flag
    - Priority queue of top-k most frequent words passing through this node
    - Complete word if this is a terminal node
    
    Space Complexity: O(ALPHABET_SIZE + k) where k is max heap size
    """
    
    def __init__(self, max_suggestions: int = 10):
        """
        Initialize a Trie node.
        
        Args:
            max_suggestions (int): Maximum number of suggestions to maintain at this node
        
        Time Complexity: O(1)
        """
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False
        self.word: Optional[str] = None  # Store complete word at terminal nodes
        self.frequency: int = 0  # Frequency if this is a word ending
        
        # Min-heap to maintain top-k frequent words
        # Stores tuples: (frequency, word)
        # Min-heap allows efficient removal of least frequent item
        self.top_suggestions: List[Tuple[int, str]] = []
        self.max_suggestions: int = max_suggestions
        
        # Quick lookup for words in heap
        self.suggestion_set: set = set()
    
    def add_to_suggestions(self, word: str, frequency: int) -> None:
        """
        Add or update a word in this node's top suggestions.
        
        Strategy: Maintain a min-heap of size k. If heap is full and new word
        has higher frequency than minimum, replace the minimum.
        
        Args:
            word (str): The word to add
            frequency (int): Frequency of the word
        
        Time Complexity: O(log k) where k = max_suggestions
        Space Complexity: O(k)
        """
        # If word already in suggestions, remove old entry
        if word in self.suggestion_set:
            self.top_suggestions = [(f, w) for f, w in self.top_suggestions if w != word]
            heapq.heapify(self.top_suggestions)
            self.suggestion_set.remove(word)
        
        # If heap not full, just add
        if len(self.top_suggestions) < self.max_suggestions:
            heapq.heappush(self.top_suggestions, (frequency, word))
            self.suggestion_set.add(word)
        else:
            # If new word has higher frequency than minimum, replace
            if frequency > self.top_suggestions[0][0]:
                removed = heapq.heapreplace(self.top_suggestions, (frequency, word))
                self.suggestion_set.remove(removed[1])
                self.suggestion_set.add(word)
    
    def get_top_suggestions(self) -> List[Tuple[int, str]]:
        """
        Get top suggestions sorted by frequency (highest first).
        
        Returns:
            List of (frequency, word) tuples sorted in descending order
        
        Time Complexity: O(k log k) where k = max_suggestions
        """
        return sorted(self.top_suggestions, reverse=True)


class Trie:
    """
    Trie (Prefix Tree) data structure optimized for autocomplete suggestions.
    
    Key Features:
    - Efficient prefix-based search: O(p) where p = prefix length
    - Maintains top-k suggestions at each node for fast retrieval
    - Dynamic frequency updates without rebuilding structure
    - Memory-efficient shared prefix storage
    
    Performance Characteristics:
    - Search is independent of dataset size (only depends on prefix length)
    - Updates are localized to the path of the word
    - Space trade-off: stores top-k suggestions at nodes for speed
    """
    
    def __init__(self, max_suggestions: int = 10):
        """
        Initialize the Trie with an empty root node.
        
        Args:
            max_suggestions (int): Maximum suggestions to maintain per node
        
        Time Complexity: O(1)
        """
        self.root = TrieNode(max_suggestions)
        self.max_suggestions = max_suggestions
        self.total_words = 0
        self.word_frequencies: Dict[str, int] = {}  # Global word frequency map
        
        # Statistics for analysis
        self.total_nodes = 1
        self.max_depth = 0
    
    def insert(self, word: str, frequency: int = 1) -> None:
        """
        Insert a word into the Trie with given frequency.
        
        Process:
        1. Traverse/create nodes for each character
        2. Mark final node as word ending
        3. Update top suggestions along the path
        
        Args:
            word (str): Word to insert (case-insensitive)
            frequency (int): Initial frequency (default 1)
        
        Time Complexity: O(m + m*log(k)) = O(m*log(k))
            where m = word length, k = max_suggestions
            - O(m) to traverse/create nodes
            - O(log k) per node to update heap (m nodes total)
        
        Space Complexity: O(m) for new nodes if word doesn't exist
        """
        word = word.lower().strip()
        if not word:
            return
        
        node = self.root
        depth = 0
        
        # Traverse and create nodes as needed
        for char in word:
            depth += 1
            if char not in node.children:
                node.children[char] = TrieNode(self.max_suggestions)
                self.total_nodes += 1
            node = node.children[char]
            
            # Update top suggestions at this node
            node.add_to_suggestions(word, frequency)
        
        # Mark as end of word
        if not node.is_end_of_word:
            self.total_words += 1
        
        node.is_end_of_word = True
        node.word = word
        node.frequency = frequency
        
        # Update global frequency map
        self.word_frequencies[word] = frequency
        
        # Update max depth
        self.max_depth = max(self.max_depth, depth)
    
    def search(self, prefix: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for words matching the given prefix, ranked by frequency.
        
        Strategy:
        1. Navigate to prefix node: O(p)
        2. Retrieve pre-computed top suggestions: O(k log k)
        3. Much faster than scanning entire dataset: O(n)
        
        Args:
            prefix (str): Search prefix (case-insensitive)
            limit (int): Maximum number of results to return
        
        Returns:
            List of SearchResult objects sorted by frequency (descending)
        
        Time Complexity: O(p + k*log(k))
            where p = prefix length, k = min(limit, max_suggestions)
            - O(p) to navigate to prefix node
            - O(k log k) to sort top suggestions
        
        Space Complexity: O(k) for results list
        
        Comparison with Naive Approach:
        - Naive: O(n*m) where n = total words, m = avg word length
        - Trie: O(p + k*log(k)) - independent of dataset size!
        """
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        
        # Navigate to prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # Prefix doesn't exist
            node = node.children[char]
        
        # Get top suggestions from this node
        suggestions = node.get_top_suggestions()
        
        # Limit results and calculate color codes
        results = []
        for rank, (freq, word) in enumerate(suggestions[:limit], 1):
            color = self._get_color_code(freq)
            results.append(SearchResult(
                word=word,
                frequency=freq,
                rank=rank,
                color=color
            ))
        
        return results
    
    def update_frequency(self, word: str) -> bool:
        """
        Increment frequency of a word when user selects it.
        
        Process:
        1. Increment global frequency
        2. Traverse word path
        3. Update top suggestions at each node along path
        
        Args:
            word (str): Word to update (case-insensitive)
        
        Returns:
            bool: True if word exists and was updated, False otherwise
        
        Time Complexity: O(m + m*log(k)) = O(m*log(k))
            where m = word length, k = max_suggestions
            - O(m) to traverse path
            - O(log k) per node to update heap
        
        Space Complexity: O(1) - in-place update
        """
        word = word.lower().strip()
        if word not in self.word_frequencies:
            return False
        
        # Increment frequency
        self.word_frequencies[word] += 1
        new_freq = self.word_frequencies[word]
        
        # Update along the path
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
            # Update this node's top suggestions with new frequency
            node.add_to_suggestions(word, new_freq)
        
        # Update terminal node
        node.frequency = new_freq
        
        return True
    
    def _get_color_code(self, frequency: int) -> str:
        """
        Determine color code based on frequency percentile.
        
        Green: Top 33% (highly popular)
        Yellow: Middle 33% (moderately popular)
        Orange: Bottom 33% (less popular)
        
        Args:
            frequency (int): Word frequency
        
        Returns:
            str: Color code ('green', 'yellow', or 'orange')
        
        Time Complexity: O(1)
        """
        if not self.word_frequencies:
            return 'orange'
        
        # Calculate percentile thresholds
        freqs = sorted(self.word_frequencies.values(), reverse=True)
        n = len(freqs)
        
        if n == 0:
            return 'orange'
        
        top_33_threshold = freqs[min(n // 3, n - 1)]
        top_66_threshold = freqs[min(2 * n // 3, n - 1)]
        
        if frequency >= top_33_threshold:
            return 'green'
        elif frequency >= top_66_threshold:
            return 'yellow'
        else:
            return 'orange'
    
    def get_all_words(self) -> List[Dict[str, any]]:
        """
        Get all words in the Trie with their frequencies.
        
        Returns:
            List of dictionaries containing word, frequency, and color
        
        Time Complexity: O(n) where n = total words
        Space Complexity: O(n)
        """
        words = []
        for word, freq in self.word_frequencies.items():
            color = self._get_color_code(freq)
            words.append({
                'word': word,
                'frequency': freq,
                'color': color
            })
        
        return sorted(words, key=lambda x: x['frequency'], reverse=True)
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get Trie statistics for dashboard display.
        
        Returns:
            Dictionary containing various statistics
        
        Time Complexity: O(1)
        """
        return {
            'total_words': self.total_words,
            'total_nodes': self.total_nodes,
            'max_depth': self.max_depth,
            'avg_word_frequency': sum(self.word_frequencies.values()) / max(len(self.word_frequencies), 1),
            'max_frequency': max(self.word_frequencies.values()) if self.word_frequencies else 0,
            'min_frequency': min(self.word_frequencies.values()) if self.word_frequencies else 0
        }
    
    def get_visualization_data(self, prefix: str) -> Dict[str, any]:
        """
        Get Trie structure data for visualization.
        
        Returns path from root to prefix node with node information.
        
        Args:
            prefix (str): Prefix to visualize path for
        
        Returns:
            Dictionary containing path information and node details
        
        Time Complexity: O(p) where p = prefix length
        """
        prefix = prefix.lower().strip()
        path = []
        node = self.root
        
        # Traverse and collect path information
        for i, char in enumerate(prefix):
            current_prefix = prefix[:i+1]
            
            if char not in node.children:
                break
            
            node = node.children[char]
            
            # Collect node information
            node_info = {
                'char': char,
                'prefix': current_prefix,
                'is_word': node.is_end_of_word,
                'frequency': node.frequency if node.is_end_of_word else 0,
                'children_count': len(node.children),
                'top_suggestions': [
                    {'word': w, 'freq': f} 
                    for f, w in node.get_top_suggestions()[:5]
                ]
            }
            path.append(node_info)
        
        return {
            'prefix': prefix,
            'path': path,
            'path_length': len(path),
            'found': len(path) == len(prefix)
        }


# Example usage and testing
if __name__ == "__main__":
    # Create Trie instance
    trie = Trie(max_suggestions=10)
    
    # Sample data - common words with realistic frequencies
    sample_words = [
        ("apple", 1523), ("application", 892), ("app", 445), ("appreciate", 234),
        ("apply", 678), ("approach", 445), ("appropriate", 334),
        ("banana", 234), ("band", 567), ("bank", 890), ("ball", 445),
        ("cat", 1234), ("car", 2345), ("card", 1567), ("care", 890),
        ("case", 678), ("call", 1890),
        ("dog", 987), ("door", 654), ("down", 1234), ("do", 2345),
        ("tree", 456), ("true", 789), ("try", 1234), ("trip", 567),
        ("python", 1890), ("programming", 1456), ("project", 1234),
        ("computer", 2134), ("code", 1987), ("algorithm", 1345)
    ]
    
    # Insert words
    print("Inserting words into Trie...")
    for word, freq in sample_words:
        trie.insert(word, freq)
    
    print(f"\nTrie Statistics:")
    stats = trie.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search
    print("\n" + "="*60)
    print("Testing Search Functionality")
    print("="*60)
    
    test_prefixes = ["app", "ca", "pro", "tr", "do"]
    
    for prefix in test_prefixes:
        print(f"\nSearch for '{prefix}':")
        results = trie.search(prefix, limit=5)
        
        if results:
            for result in results:
                color_emoji = {"green": "🟢", "yellow": "🟡", "orange": "🟠"}
                print(f"  {color_emoji[result.color]} {result.rank}. {result.word:15} (freq: {result.frequency:4})")
        else:
            print("  No results found")
    
    # Test frequency update
    print("\n" + "="*60)
    print("Testing Dynamic Frequency Update")
    print("="*60)
    
    print("\nBefore update - Search for 'app':")
    results_before = trie.search("app", limit=3)
    for r in results_before:
        print(f"  {r.word}: {r.frequency}")
    
    # User selects "appreciate" multiple times
    print("\nSimulating user selecting 'appreciate' 1500 times...")
    for _ in range(1500):
        trie.update_frequency("appreciate")
    
    print("\nAfter update - Search for 'app':")
    results_after = trie.search("app", limit=3)
    for r in results_after:
        print(f"  {r.word}: {r.frequency}")
    
    # Test visualization data
    print("\n" + "="*60)
    print("Testing Visualization Data")
    print("="*60)
    
    viz_data = trie.get_visualization_data("app")
    print(f"\nPath for 'app':")
    for node in viz_data['path']:
        print(f"  '{node['char']}' -> {node['prefix']}")
        print(f"    Children: {node['children_count']}")
        print(f"    Top suggestions: {[s['word'] for s in node['top_suggestions'][:3]]}")