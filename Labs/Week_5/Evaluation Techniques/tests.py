def test_dataset_loading(train_data, test_data):
    """Test if the dataset is loaded correctly."""
    assert train_data is not None, "Train data is not loaded."
    assert test_data is not None, "Test data is not loaded."
    print("test_dataset_loading passed.")

def test_vocab_creation(vocab, word_to_idx):
    """Test if the vocabulary and word-to-index mappings are created."""
    assert len(vocab) > 0, "Vocabulary is empty."
    assert len(word_to_idx) > 0, "Word-to-index mapping is empty."
    print("test_vocab_creation passed.")

def test_sequence(train_sequences, test_sequences):
    """Test if sequences are created properly."""
    assert len(train_sequences) > 0, "Train sequences are empty."
    assert len(test_sequences) > 0, "Test sequences are empty."
    assert isinstance(train_sequences[0], int), "Train sequences should contain integers."
    print("test_sequences passed.")

def test_model_building(model):
    """Test if the model is built and compiled correctly."""
    assert model is not None, "Model is not built."
    assert len(model.layers) > 0, "Model has no layers."
    print("test_model_building passed.")

def test_training_output(history):
    """Test if the training history is valid."""
    assert "loss" in history.history, "Loss not found in training history."
    assert "accuracy" in history.history, "Accuracy not found in training history."
    assert len(history.history["loss"]) > 0, "No training history for loss."
    print("test_training_output passed.")

def test_perplexity(perplexity):
    """Test if perplexity is computed correctly."""
    assert perplexity > 0, "Perplexity should be positive."
    print("test_perplexity passed.")
