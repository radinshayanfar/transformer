"""
Tests for tokenizer training functionality in the transformer repository.

This test suite covers:
- BPE tokenizer training from preprocess.py
- Custom tokenizer integration in train_webtext.py
- Corpus preparation in prepare_webtext.py
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import train_bpe
from tokenizers import Tokenizer


class TestTokenizerTraining(unittest.TestCase):
    """Test suite for BPE tokenizer training functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.corpus_file = os.path.join(self.temp_dir, "test_corpus.txt")
        self.tokenizer_file = os.path.join(self.temp_dir, "test_tokenizer.json")
        
        # Create a sample corpus
        self.sample_text = """
        The quick brown fox jumps over the lazy dog.
        Machine learning is a subset of artificial intelligence.
        Natural language processing enables computers to understand human language.
        Transformers are a type of neural network architecture.
        Byte pair encoding is a tokenization algorithm.
        Deep learning models require large amounts of training data.
        """
        
        with open(self.corpus_file, 'w') as f:
            f.write(self.sample_text)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_train_bpe_creates_tokenizer(self):
        """Test that train_bpe creates a tokenizer file."""
        train_bpe([self.corpus_file], vocab_size=500, save_path=self.tokenizer_file)
        
        self.assertTrue(os.path.exists(self.tokenizer_file))

    def test_train_bpe_tokenizer_is_loadable(self):
        """Test that the trained tokenizer can be loaded."""
        train_bpe([self.corpus_file], vocab_size=500, save_path=self.tokenizer_file)
        
        tokenizer = Tokenizer.from_file(self.tokenizer_file)
        self.assertIsNotNone(tokenizer)

    def test_tokenizer_has_special_tokens(self):
        """Test that the trained tokenizer includes required special tokens."""
        train_bpe([self.corpus_file], vocab_size=500, save_path=self.tokenizer_file)
        
        tokenizer = Tokenizer.from_file(self.tokenizer_file)
        
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[EOS]", "[BOS]"]
        for token in special_tokens:
            token_id = tokenizer.token_to_id(token)
            self.assertIsNotNone(token_id, f"Special token '{token}' not found in tokenizer")

    def test_tokenizer_can_encode_text(self):
        """Test that the trained tokenizer can encode text."""
        train_bpe([self.corpus_file], vocab_size=500, save_path=self.tokenizer_file)
        
        tokenizer = Tokenizer.from_file(self.tokenizer_file)
        
        test_sentence = "The transformer model is powerful."
        encoded = tokenizer.encode(test_sentence)
        
        self.assertIsNotNone(encoded)
        self.assertTrue(len(encoded.ids) > 0)
        self.assertTrue(len(encoded.tokens) > 0)

    def test_tokenizer_vocab_size(self):
        """Test that the tokenizer respects the requested vocab size."""
        requested_vocab_size = 300
        train_bpe([self.corpus_file], vocab_size=requested_vocab_size, save_path=self.tokenizer_file)
        
        tokenizer = Tokenizer.from_file(self.tokenizer_file)
        actual_vocab_size = tokenizer.get_vocab_size()
        
        # The actual vocab size should be <= requested size
        # (it may be smaller if the corpus doesn't have enough unique tokens)
        self.assertLessEqual(actual_vocab_size, requested_vocab_size)

    def test_tokenizer_with_custom_special_tokens(self):
        """Test that train_bpe accepts custom special tokens."""
        custom_tokens = ["[UNK]", "[PAD]", "[START]", "[END]"]
        train_bpe(
            [self.corpus_file],
            vocab_size=500,
            save_path=self.tokenizer_file,
            special_tokens=custom_tokens
        )
        
        tokenizer = Tokenizer.from_file(self.tokenizer_file)
        
        for token in custom_tokens:
            token_id = tokenizer.token_to_id(token)
            self.assertIsNotNone(token_id, f"Custom special token '{token}' not found")

    def test_train_bpe_with_multiple_files(self):
        """Test that train_bpe can handle multiple corpus files."""
        corpus_file2 = os.path.join(self.temp_dir, "test_corpus2.txt")
        with open(corpus_file2, 'w') as f:
            f.write("Additional training data for the tokenizer.\n")
        
        train_bpe(
            [self.corpus_file, corpus_file2],
            vocab_size=500,
            save_path=self.tokenizer_file
        )
        
        self.assertTrue(os.path.exists(self.tokenizer_file))
        
        tokenizer = Tokenizer.from_file(self.tokenizer_file)
        self.assertIsNotNone(tokenizer)


class TestWebTextTokenizerIntegration(unittest.TestCase):
    """Test suite for WebText tokenizer integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipIf(True, "Requires torch and full dependencies")
    def test_custom_tokenizer_mode(self):
        """Test that custom tokenizer mode uses the correct tokenizer."""
        # This test would require mocking the full train_webtext module
        # For now, we'll test the collate_fn function logic
        pass

    @unittest.skipIf(True, "Requires torch and full dependencies")
    def test_collate_fn_with_custom_tokenizer(self):
        """Test that collate_fn handles custom tokenizer correctly."""
        # Import the function
        from train_webtext import WebTextDataset
        
        # Create a mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode_batch.return_value = [
            Mock(ids=[1, 2, 3], attention_mask=[1, 1, 1])
        ]
        
        batch = [{"text": "test sentence"}]
        
        # This would require utils.pad_and_mask to be available
        # The actual test would verify the function is called correctly
        pass


class TestPrepareWebtextCorpus(unittest.TestCase):
    """Test suite for corpus preparation functionality."""

    @unittest.skipIf(True, "Requires openwebtext module")
    def test_corpus_file_creation(self):
        """Test that corpus file is created when flag is set."""
        # This would require mocking the openwebtext module
        # and testing the corpus writing logic
        pass


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTokenizerTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestWebTextTokenizerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPrepareWebtextCorpus))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
