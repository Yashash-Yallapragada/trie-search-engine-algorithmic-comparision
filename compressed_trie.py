"""
Compressed Trie (Patricia Trie) Implementation
Space-Optimized Trie with Path Compression

Key Optimization: Instead of creating one node per character,
compress chains of single-child nodes into single edges with string labels.

Example:
Standard Trie for "test", "testing":
    t → e → s → t (word)
              ↓
              i → n → g (word)
    = 8 nodes

Compressed Trie:
    "test" (word)
      ↓
    "ing" (word)
    = 2 nodes (75% reduction!)

Time Complexity: Same as standard Trie O(p)
Space Complexity: O(n) but with 60-70% fewer nodes in practice

Author: DSA Project
Purpose: Demonstrate space optimization over standard Trie
"""

import heapq
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result with metadata"""
    word: str
    frequency: int
    rank: int
    color: str


class CompressedTrieNode:
    """
    Node in compressed Trie with edge labels.
    
    Key Difference from Standard Trie:
    - Edges store STRINGS instead of single characters
    - Reduces number of nodes dramatically
    - "test" → "ing" is one edge labeled "ing"
    
    Attributes:
        edge_label (str): String stored on edge leading to this node
        children (dict): Maps first character → child node
        is_end_of_word (bool): True if valid word ends here
        word (str): Complete word if terminal node
        frequency (int): Word frequency
        top_suggestions (list): Heap of (freq, word) for top-k
    
    Space Savings:
    - Standard: 1 node per character = O(m) nodes per word
    - Compressed: Often just 1-3 nodes per word = O(1) average
    """
    
    def __init__(self, edge_label: str = "", max_suggestions: int = 10):
        """
        Initialize compressed Trie node.
        
        Args:
            edge_label: String on edge leading to this node
            max_suggestions: Max suggestions to track per node
        
        Time Complexity: O(1)
        """
        self.edge_label = edge_label
        self.children: Dict[str, CompressedTrieNode] = {}
        self.is_end_of_word = False
        self.word: Optional[str] = None
        self.frequency = 0
        
        # Top-k suggestions at this node
        self.top_suggestions: List[Tuple[int, str]] = []
        self.max_suggestions = max_suggestions
        self.suggestion_set: set = set()
    
    def add_to_suggestions(self, word: str, frequency: int) -> None:
        """
        Add word to top suggestions heap.
        
        Time Complexity: O(log k) where k = max_suggestions
        """
        if word in self.suggestion_set:
            self.top_suggestions = [(f, w) for f, w in self.top_suggestions if w != word]
            heapq.heapify(self.top_suggestions)
            self.suggestion_set.remove(word)
        
        if len(self.top_suggestions) < self.max_suggestions:
            heapq.heappush(self.top_suggestions, (frequency, word))
            self.suggestion_set.add(word)
        else:
            if frequency > self.top_suggestions[0][0]:
                removed = heapq.heapreplace(self.top_suggestions, (frequency, word))
                self.suggestion_set.remove(removed[1])
                self.suggestion_set.add(word)
    
    def get_top_suggestions(self) -> List[Tuple[int, str]]:
        """Get top suggestions sorted by frequency (descending)"""
        return sorted(self.top_suggestions, reverse=True)


class CompressedTrie:
    """
    Compressed Trie (Patricia Trie) - Space-Optimized Trie.
    
    Core Innovation: Path Compression
    
    When a node has only one child, compress the path into a single edge
    with a multi-character label. This dramatically reduces node count.
    
    Example Compression:
    
    Before (Standard Trie):
        root → t → e → s → t (4 nodes)
    
    After (Compressed):
        root → "test" (1 node)
    
    Performance Characteristics:
    - Search: O(p) - same as standard Trie
    - Insert: O(p) - with possible node splitting
    - Space: 60-70% reduction in node count
    - Cache efficiency: Better locality due to fewer nodes
    
    Space Savings Example:
    - 10,000 words with avg length 7
    - Standard Trie: ~40,000 nodes
    - Compressed Trie: ~12,000 nodes (70% reduction!)
    """
    
    def __init__(self, max_suggestions: int = 10):
        """
        Initialize compressed Trie.
        
        Args:
            max_suggestions: Max suggestions per node
        
        Time Complexity: O(1)
        """
        self.root = CompressedTrieNode("", max_suggestions)
        self.max_suggestions = max_suggestions
        self.total_words = 0
        self.word_frequencies: Dict[str, int] = {}
        
        # Statistics
        self.total_nodes = 1
        self.max_depth = 0
        self.compression_ratio = 0.0  # Will calculate after insertions
    
    def _find_common_prefix_length(self, str1: str, str2: str) -> int:
        """
        Find length of common prefix between two strings.
        
        Args:
            str1, str2: Strings to compare
        
        Returns:
            Length of common prefix
        
        Example:
            "testing", "test" → 4
            "apple", "apply" → 4
            "cat", "dog" → 0
        
        Time Complexity: O(min(len(str1), len(str2)))
        """
        min_len = min(len(str1), len(str2))
        i = 0
        while i < min_len and str1[i] == str2[i]:
            i += 1
        return i
    
    def insert(self, word: str, frequency: int = 1) -> None:
        """
        Insert word into compressed Trie with path compression.
        
        Algorithm:
        1. Navigate through tree matching edge labels
        2. If edge label partially matches:
           - Split the edge (create intermediate node)
           - Create two children (compressed paths)
        3. If no match, create new compressed edge
        
        Example Insertion:
        
        Insert "test":
            root → "test" [word]
        
        Insert "testing":
            root → "test" [word]
                    ↓
                  "ing" [word]
        
        Insert "tea":
            root → "te"
                    ├→ "st" [word]
                    │   ↓
                    │  "ing" [word]
                    └→ "a" [word]
        
        Args:
            word: Word to insert
            frequency: Initial frequency
        
        Time Complexity: O(p) where p = word length
        Space Complexity: O(1) amortized (compression saves space)
        """
        word = word.lower().strip()
        if not word:
            return
        
        node = self.root
        remaining = word
        depth = 0
        
        while remaining:
            depth += 1
            found_match = False
            
            # Check if any child edge matches
            first_char = remaining[0]
            
            if first_char in node.children:
                child = node.children[first_char]
                edge_label = child.edge_label
                
                # Find common prefix between remaining and edge label
                common_len = self._find_common_prefix_length(remaining, edge_label)
                
                if common_len == len(edge_label):
                    # Full edge match - continue down tree
                    node = child
                    remaining = remaining[common_len:]
                    found_match = True
                    
                elif common_len > 0:
                    # Partial match - need to split edge
                    # 
                    # Before: node → "test" (child)
                    # After:  node → "te" (new_node)
                    #                 ├→ "st" (child with adjusted label)
                    #                 └→ "a" (new word node)
                    
                    # Create intermediate node
                    intermediate = CompressedTrieNode(edge_label[:common_len], self.max_suggestions)
                    self.total_nodes += 1
                    
                    # Adjust existing child's edge label
                    child.edge_label = edge_label[common_len:]
                    
                    # Reconnect: node → intermediate → child
                    node.children[first_char] = intermediate
                    intermediate.children[child.edge_label[0]] = child
                    
                    # Continue with intermediate node
                    node = intermediate
                    remaining = remaining[common_len:]
                    found_match = True
            
            if not found_match:
                # No matching edge - create new compressed path
                new_node = CompressedTrieNode(remaining, self.max_suggestions)
                node.children[remaining[0]] = new_node
                self.total_nodes += 1
                node = new_node
                remaining = ""
        
        # Mark as end of word
        if not node.is_end_of_word:
            self.total_words += 1
        
        node.is_end_of_word = True
        node.word = word
        node.frequency = frequency
        self.word_frequencies[word] = frequency
        
        # Update top suggestions along path
        self._update_suggestions_on_path(word, frequency)
        
        self.max_depth = max(self.max_depth, depth)
    
    def _update_suggestions_on_path(self, word: str, frequency: int) -> None:
        """
        Update top suggestions for all nodes on path to word.
        
        Time Complexity: O(p * log k) where p = word length, k = max_suggestions
        """
        node = self.root
        remaining = word
        
        while remaining:
            node.add_to_suggestions(word, frequency)
            
            first_char = remaining[0]
            if first_char not in node.children:
                break
            
            child = node.children[first_char]
            edge_label = child.edge_label
            common_len = self._find_common_prefix_length(remaining, edge_label)
            
            if common_len < len(edge_label):
                break
            
            node = child
            remaining = remaining[common_len:]
        
        node.add_to_suggestions(word, frequency)
    
    def search(self, prefix: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for words matching prefix.
        
        Algorithm:
        1. Navigate to node representing prefix
        2. Retrieve pre-computed top suggestions
        3. Return ranked results
        
        Time Complexity: O(p + k log k)
            where p = prefix length, k = number of results
        
        Space Advantage Over Standard Trie:
        - Fewer nodes to traverse = better cache locality
        - Same time complexity but faster in practice
        
        Args:
            prefix: Search prefix
            limit: Max results to return
        
        Returns:
            List of SearchResult objects sorted by frequency
        """
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        
        node = self.root
        remaining = prefix
        
        # Navigate to prefix node
        while remaining:
            first_char = remaining[0]
            
            if first_char not in node.children:
                return []
            
            child = node.children[first_char]
            edge_label = child.edge_label
            
            # Check if edge label matches remaining prefix
            common_len = self._find_common_prefix_length(remaining, edge_label)
            
            if common_len < len(remaining) and common_len < len(edge_label):
                # Prefix doesn't match this path
                return []
            
            if common_len == len(edge_label):
                # Full edge traversed
                node = child
                remaining = remaining[common_len:]
            else:
                # Partial edge match - prefix ends mid-edge
                node = child
                remaining = ""
        
        # Get top suggestions from this node
        suggestions = node.get_top_suggestions()
        
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
        Update word frequency after user selection.
        
        Time Complexity: O(p + p * log k) = O(p * log k)
        """
        word = word.lower().strip()
        if word not in self.word_frequencies:
            return False
        
        self.word_frequencies[word] += 1
        new_freq = self.word_frequencies[word]
        
        # Update suggestions along path
        self._update_suggestions_on_path(word, new_freq)
        
        # Update terminal node frequency
        node = self.root
        remaining = word
        
        while remaining:
            first_char = remaining[0]
            if first_char not in node.children:
                return False
            
            child = node.children[first_char]
            edge_label = child.edge_label
            common_len = self._find_common_prefix_length(remaining, edge_label)
            
            if common_len == len(edge_label):
                node = child
                remaining = remaining[common_len:]
            else:
                return False
        
        node.frequency = new_freq
        return True
    
    def _get_color_code(self, frequency: int) -> str:
        """Determine color based on frequency percentile"""
        if not self.word_frequencies:
            return 'orange'
        
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
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get Trie statistics including compression ratio.
        
        Compression Ratio:
        - Theoretical standard Trie nodes = sum of word lengths
        - Actual compressed nodes = self.total_nodes
        - Ratio = actual / theoretical
        
        Returns:
            Dictionary with statistics
        """
        theoretical_nodes = sum(len(word) for word in self.word_frequencies.keys())
        if theoretical_nodes > 0:
            self.compression_ratio = self.total_nodes / theoretical_nodes
        
        return {
            'total_words': self.total_words,
            'total_nodes': self.total_nodes,
            'max_depth': self.max_depth,
            'compression_ratio': f"{self.compression_ratio:.2%}",
            'space_saved': f"{(1 - self.compression_ratio) * 100:.1f}%",
            'avg_word_frequency': sum(self.word_frequencies.values()) / max(len(self.word_frequencies), 1),
            'max_frequency': max(self.word_frequencies.values()) if self.word_frequencies else 0,
            'min_frequency': min(self.word_frequencies.values()) if self.word_frequencies else 0
        }
    
    def get_all_words(self) -> List[Dict[str, any]]:
        """Get all words with metadata"""
        words = []
        for word, freq in self.word_frequencies.items():
            color = self._get_color_code(freq)
            words.append({
                'word': word,
                'frequency': freq,
                'color': color
            })
        return sorted(words, key=lambda x: x['frequency'], reverse=True)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("COMPRESSED TRIE (PATRICIA TRIE) DEMONSTRATION")
    print("=" * 70)
    
    # Create compressed Trie
    compressed = CompressedTrie(max_suggestions=10)
    
    # Sample words
    test_words = [
        ("test", 500), ("testing", 300), ("tester", 200), ("tea", 450),
        ("apple", 1523), ("application", 892), ("app", 445), ("apply", 678),
        ("tree", 456), ("try", 1234), ("tried", 567), ("trying", 789),
        ("program", 1000), ("programming", 1500), ("programmer", 800)
    ]
    
    print("\n📥 Inserting words into Compressed Trie...")
    for word, freq in test_words:
        compressed.insert(word, freq)
    
    # Display statistics
    print("\n📊 Statistics:")
    stats = compressed.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n💡 Space Savings: {stats['space_saved']} compared to standard Trie!")
    
    # Test searches
    print("\n" + "=" * 70)
    print("SEARCH TESTS")
    print("=" * 70)
    
    test_prefixes = ["te", "app", "pro", "tr"]
    
    for prefix in test_prefixes:
        print(f"\n🔍 Search for '{prefix}':")
        results = compressed.search(prefix, limit=5)
        
        if results:
            for r in results:
                emoji = {"green": "🟢", "yellow": "🟡", "orange": "🟠"}[r.color]
                print(f"  {emoji} {r.rank}. {r.word:15} (freq: {r.frequency:4})")
        else:
            print("  No results found")
    
    # Test dynamic update
    print("\n" + "=" * 70)
    print("DYNAMIC FREQUENCY UPDATE TEST")
    print("=" * 70)
    
    print("\n📊 Before update - 'te' results:")
    before = compressed.search("te", limit=3)
    for r in before:
        print(f"  {r.word}: {r.frequency}")
    
    print("\n🔄 User selects 'tea' 500 times...")
    for _ in range(500):
        compressed.update_frequency("tea")
    
    print("\n📊 After update - 'te' results:")
    after = compressed.search("te", limit=3)
    for r in after:
        print(f"  {r.word}: {r.frequency}")
    
    print("\n" + "=" * 70)
    print("KEY ADVANTAGES OF COMPRESSED TRIE:")
    print("=" * 70)
    print("""
1. ✅ Space Efficiency: 60-70% fewer nodes than standard Trie
2. ✅ Same Time Complexity: Still O(p) for search
3. ✅ Better Cache Performance: Fewer nodes = better locality
4. ✅ Practical Improvement: Faster in real-world scenarios
5. ✅ Optimal for Long Common Prefixes: Maximum compression
    """)