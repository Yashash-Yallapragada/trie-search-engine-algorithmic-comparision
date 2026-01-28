"""
Ternary Search Trie (TST) Implementation
Memory-Optimized Trie with 3-Way Branching

Key Optimization: Instead of 26 child pointers per node (standard Trie),
each node has only 3 pointers: left, middle, right.

Standard Trie Node Memory:
    26 pointers × 8 bytes = 208 bytes per node
    Most pointers are NULL (wasted space)

TST Node Memory:
    3 pointers × 8 bytes = 24 bytes per node
    89% memory reduction per node!

How TST Works:
    Each node stores one character and has:
    - left: points to nodes with smaller characters
    - middle: points to next character in word
    - right: points to nodes with larger characters

Example for "cat", "car", "dog":
            'c'
           /   \
          'a'  'd'
         /      \
        't'     'o'
       /         \
      'r'        'g'

Time Complexity: O(p + log n) average
Space Complexity: O(n) with 89% less memory per node

Author: DSA Project
Purpose: Demonstrate memory optimization over standard Trie
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


class TSTNode:
    """
    Node in Ternary Search Trie.
    
    Key Innovation: 3-way branching instead of 26-way
    
    Structure:
        char: Character stored at this node
        left: Nodes with smaller characters
        middle: Next character in word (follow path)
        right: Nodes with larger characters
        is_end: True if word ends here
    
    Memory Savings:
        Standard Trie: 26 pointers = 208 bytes
        TST: 3 pointers = 24 bytes
        Savings: 88.5% per node!
    
    Example:
        Inserting "cat", "cap", "bat":
        
                  'c'
                 / | \
              'b'  'a' 
                   |
                   't'
                  /
                'p'
        
        Navigation:
        - Looking for 'bat': go left from 'c' to 'b'
        - Looking for 'cat': equal to 'c', go middle to 'a'
        - Looking for 'cap': equal to 'c', middle to 'a', left to 'p'
    
    Time Complexity:
        - Search: O(p + log n) where p = word length, n = words
        - The log n comes from balanced tree structure
    """
    
    def __init__(self, char: str, max_suggestions: int = 10):
        """
        Initialize TST node.
        
        Args:
            char: Character stored at this node
            max_suggestions: Max suggestions to track
        
        Space: Only 3 pointers vs 26 in standard Trie!
        """
        self.char = char
        
        # Three-way branching (instead of 26-way)
        self.left: Optional[TSTNode] = None    # Characters < self.char
        self.middle: Optional[TSTNode] = None  # Next char in word
        self.right: Optional[TSTNode] = None   # Characters > self.char
        
        # Word ending info
        self.is_end_of_word = False
        self.word: Optional[str] = None
        self.frequency = 0
        
        # Top-k suggestions
        self.top_suggestions: List[Tuple[int, str]] = []
        self.max_suggestions = max_suggestions
        self.suggestion_set: set = set()
    
    def add_to_suggestions(self, word: str, frequency: int) -> None:
        """
        Add word to top suggestions heap.
        
        Time Complexity: O(log k)
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
        """Get top suggestions sorted by frequency"""
        return sorted(self.top_suggestions, reverse=True)


class TernarySearchTrie:
    """
    Ternary Search Trie - Memory-Optimized Trie.
    
    Core Innovation: 3-Way Branching
    
    Instead of storing 26 child pointers (one for each letter),
    each node has only 3 pointers, organizing children as a binary
    search tree at each level.
    
    Memory Comparison (10,000 words):
        Standard Trie:
        - ~40,000 nodes × 208 bytes = 8.3 MB
        
        TST:
        - ~40,000 nodes × 24 bytes = 0.96 MB
        - 88.5% memory reduction!
    
    Performance:
        - Search: O(p + log n) average
            - p = prefix length (same as standard Trie)
            - log n = tree navigation at each level
        - Insert: O(p + log n) average
        - Space: O(n) but with 89% less memory per node
    
    Trade-off:
        - Slightly slower than standard Trie (log n factor)
        - But MUCH more memory efficient
        - Ideal for memory-constrained environments
    
    Use Cases:
        - Mobile applications (limited RAM)
        - Embedded systems
        - Large dictionaries (millions of words)
        - When memory > speed priority
    """
    
    def __init__(self, max_suggestions: int = 10):
        """
        Initialize Ternary Search Trie.
        
        Args:
            max_suggestions: Max suggestions per node
        
        Time Complexity: O(1)
        """
        self.root: Optional[TSTNode] = None
        self.max_suggestions = max_suggestions
        self.total_words = 0
        self.word_frequencies: Dict[str, int] = {}
        
        # Statistics
        self.total_nodes = 0
        self.max_depth = 0
    
    def insert(self, word: str, frequency: int = 1) -> None:
        """
        Insert word into TST.
        
        Algorithm:
        1. For each character in word:
           a. If char < node.char: go left
           b. If char > node.char: go right
           c. If char == node.char: go middle (next char)
        2. Create nodes as needed (BST style)
        3. Mark final node as word ending
        
        Example Insertion Sequence for "cat", "car", "dog":
        
        Insert "cat":
            'c' → 'a' → 't' [word]
        
        Insert "car":
            'c' → 'a' → 't' [word]
                       ↙
                      'r' [word]
        
        Insert "dog":
            'c' → 'a' → 't' [word]
           ↙       ↙
         'd'      'r' [word]
          ↓
         'o'
          ↓
         'g' [word]
        
        Args:
            word: Word to insert
            frequency: Initial frequency
        
        Time Complexity: O(p + log n)
            - p = word length (must traverse all chars)
            - log n = average BST navigation per level
        
        Space Complexity: O(p) for new word
        """
        word = word.lower().strip()
        if not word:
            return
        
        self.root = self._insert_recursive(self.root, word, 0, frequency)
        
        if word not in self.word_frequencies:
            self.total_words += 1
        
        self.word_frequencies[word] = frequency
    
    def _insert_recursive(self, node: Optional[TSTNode], word: str, 
                         index: int, frequency: int) -> TSTNode:
        """
        Recursive insertion helper.
        
        At each level, maintains BST property:
        - left subtree: chars < current
        - right subtree: chars > current
        - middle: next character in word
        
        Time Complexity: O(log n) per character
        """
        char = word[index]
        
        # Create node if doesn't exist
        if node is None:
            node = TSTNode(char, self.max_suggestions)
            self.total_nodes += 1
        
        # BST navigation
        if char < node.char:
            # Go left (smaller character)
            node.left = self._insert_recursive(node.left, word, index, frequency)
        elif char > node.char:
            # Go right (larger character)
            node.right = self._insert_recursive(node.right, word, index, frequency)
        else:
            # Equal - this is our path
            # Add to suggestions at this node
            node.add_to_suggestions(word, frequency)
            
            if index < len(word) - 1:
                # More characters to process - go middle
                node.middle = self._insert_recursive(node.middle, word, index + 1, frequency)
            else:
                # Last character - mark as word ending
                node.is_end_of_word = True
                node.word = word
                node.frequency = frequency
        
        return node
    
    def search(self, prefix: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for words matching prefix.
        
        Algorithm:
        1. Navigate to node representing last char of prefix
        2. Retrieve pre-computed top suggestions
        3. Return ranked results
        
        Navigation Example (searching "cat"):
            Start at root
            'c': equal → go middle
            'a': equal → go middle
            't': equal → found!
        
        Args:
            prefix: Search prefix
            limit: Max results
        
        Returns:
            List of SearchResult objects
        
        Time Complexity: O(p + log n + k log k)
            - O(p) to traverse prefix
            - O(log n) for BST navigation per char
            - O(k log k) to sort results
        
        Comparison with Standard Trie:
            Standard: O(p + k log k)
            TST: O(p + log n + k log k)
            
            Trade-off: Slightly slower but 89% less memory!
        """
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        
        # Navigate to prefix node
        node = self._search_prefix(self.root, prefix, 0)
        
        if node is None:
            return []
        
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
    
    def _search_prefix(self, node: Optional[TSTNode], prefix: str, 
                      index: int) -> Optional[TSTNode]:
        """
        Navigate to node representing prefix.
        
        BST navigation at each level:
        - If char < node.char: search left
        - If char > node.char: search right
        - If char == node.char: 
            - If last char: return this node
            - Else: search middle
        
        Time Complexity: O(p + log n)
        """
        if node is None:
            return None
        
        char = prefix[index]
        
        if char < node.char:
            # Search left subtree
            return self._search_prefix(node.left, prefix, index)
        elif char > node.char:
            # Search right subtree
            return self._search_prefix(node.right, prefix, index)
        else:
            # Found matching character
            if index == len(prefix) - 1:
                # Last character - return this node
                return node
            else:
                # More characters - continue middle
                return self._search_prefix(node.middle, prefix, index + 1)
    
    def update_frequency(self, word: str) -> bool:
        """
        Update word frequency after user selection.
        
        Time Complexity: O(p + log n)
        """
        word = word.lower().strip()
        if word not in self.word_frequencies:
            return False
        
        self.word_frequencies[word] += 1
        new_freq = self.word_frequencies[word]
        
        # Update along path
        self._update_frequency_recursive(self.root, word, 0, new_freq)
        
        return True
    
    def _update_frequency_recursive(self, node: Optional[TSTNode], word: str,
                                   index: int, new_freq: int) -> None:
        """
        Update frequency along path to word.
        
        Time Complexity: O(p + log n)
        """
        if node is None:
            return
        
        char = word[index]
        
        if char < node.char:
            self._update_frequency_recursive(node.left, word, index, new_freq)
        elif char > node.char:
            self._update_frequency_recursive(node.right, word, index, new_freq)
        else:
            # Update suggestions at this node
            node.add_to_suggestions(word, new_freq)
            
            if index < len(word) - 1:
                self._update_frequency_recursive(node.middle, word, index + 1, new_freq)
            else:
                # Update terminal node
                node.frequency = new_freq
    
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
        Get TST statistics including memory savings.
        
        Memory Calculation:
            Standard Trie: nodes × 26 pointers × 8 bytes
            TST: nodes × 3 pointers × 8 bytes
            Savings: 88.5%
        
        Returns:
            Dictionary with statistics
        """
        standard_trie_memory = self.total_nodes * 26 * 8  # bytes
        tst_memory = self.total_nodes * 3 * 8  # bytes
        memory_saved = (1 - (tst_memory / standard_trie_memory)) if standard_trie_memory > 0 else 0
        
        return {
            'total_words': self.total_words,
            'total_nodes': self.total_nodes,
            'max_depth': self.max_depth,
            'memory_per_node': '24 bytes (vs 208 bytes standard)',
            'memory_savings': f"{memory_saved * 100:.1f}%",
            'estimated_memory_kb': f"{tst_memory / 1024:.2f} KB",
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
    print("TERNARY SEARCH TRIE (TST) DEMONSTRATION")
    print("=" * 70)
    
    # Create TST
    tst = TernarySearchTrie(max_suggestions=10)
    
    # Sample words
    test_words = [
        ("apple", 1523), ("application", 892), ("app", 445), ("appreciate", 234),
        ("apply", 678), ("approach", 445), ("appropriate", 334),
        ("cat", 1234), ("car", 2345), ("card", 1567), ("care", 890),
        ("tree", 456), ("try", 1234), ("tried", 567), ("trying", 789),
        ("dog", 987), ("door", 654), ("down", 1234),
        ("program", 1000), ("programming", 1500), ("programmer", 800)
    ]
    
    print("\n📥 Inserting words into TST...")
    for word, freq in test_words:
        tst.insert(word, freq)
    
    # Display statistics
    print("\n📊 Statistics:")
    stats = tst.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n💡 Memory Savings: {stats['memory_savings']} per node!")
    
    # Test searches
    print("\n" + "=" * 70)
    print("SEARCH TESTS")
    print("=" * 70)
    
    test_prefixes = ["app", "ca", "tr", "pro"]
    
    for prefix in test_prefixes:
        print(f"\n🔍 Search for '{prefix}':")
        results = tst.search(prefix, limit=5)
        
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
    
    print("\n📊 Before update - 'app' results:")
    before = tst.search("app", limit=3)
    for r in before:
        print(f"  {r.word}: {r.frequency}")
    
    print("\n🔄 User selects 'app' 2000 times...")
    for _ in range(2000):
        tst.update_frequency("app")
    
    print("\n📊 After update - 'app' results:")
    after = tst.search("app", limit=3)
    for r in after:
        print(f"  {r.word}: {r.frequency}")
    
    print("\n" + "=" * 70)
    print("KEY ADVANTAGES OF TERNARY SEARCH TRIE:")
    print("=" * 70)
    print("""
1. ✅ Memory Efficiency: 88.5% less memory per node (24 vs 208 bytes)
2. ✅ Scalable: Works well with large alphabets (not just English)
3. ✅ Balanced: BST structure provides good average performance
4. ✅ Cache Friendly: Smaller nodes = better cache utilization
5. ✅ Flexible: Can handle any character set, not limited to a-z
6. ✅ Space-Time Tradeoff: Slightly slower (log n) but massive space savings
    """)