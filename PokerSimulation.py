
## To run the program, run the following command:
## python3 PokerSimulation.py
## The program will prompt you to enter 5 cards in the format 'S5, H2, D10, CQ, HA'
## The program will then predict the poker hand class of the entered cards
## The program will then print the predicted poker hand class


import pickle
import numpy as np
import sys
import os

card_values = {
    "S": [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"],
    "H": [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"],
    "D": [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"],
    "C": [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"]
}

# Card value mapping for conversion to numeric values
card_value_mapping = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14
}

def parse_cards(card_input):
    """
    Parses a comma-separated string of card inputs and returns a feature list.
    
    Expected input format (for 5 cards):
      "H2, H3, H4, H5, H6"
    
    The function assumes the following suit mapping that matches the training data:
        Hearts   -> 1
        Diamonds -> 2
        Clubs    -> 3
        Spades   -> 4
    
    Returns:
        features (list): a list of integers representing the cards,
                         e.g., [1,2, 1,3, 1,4, 1,5, 1,6] for a flush in hearts.
    """
    # Define the correct suit mapping.
    suit_mapping = {'H': 1, 'D': 2, 'C': 3, 'S': 4}

    # Split the input by commas and strip whitespace.
    cards = [card.strip() for card in card_input.split(',')]
    
    # Check if exactly 5 cards have been entered.
    if len(cards) != 5:
        raise ValueError("Error: Exactly 5 cards are required (e.g., H2, H3, H4, H5, H6).")
    
    features = []
    for card in cards:
        # Ensure the card string is valid.
        if len(card) < 2:
            raise ValueError(f"Invalid card format: {card}")
        
        # The first character represents the suit.
        suit_char = card[0].upper()
        if suit_char not in suit_mapping:
            raise ValueError(f"Invalid suit: {suit_char} in card {card}")
        
        # The remaining part of the card is the rank
        rank_str = card[1:].upper()
        if rank_str not in card_value_mapping:
            raise ValueError(f"Invalid rank in card: {card}")
        
        rank = card_value_mapping[rank_str]
        features.append(suit_mapping[suit_char])
        features.append(rank)
        
    return features

def convert_cards_to_features(cards):
    """
    Converts a list of card strings to the feature format expected by the model.
    
    Parameters:
        cards (list): List of card strings in the format 'S5', 'H2', etc.
    
    Returns:
        list: Features in the format expected by the model
    """
    # This is a placeholder implementation - you'll need to adjust this
    # based on how your model expects the features to be formatted
    
    # For example, if your model expects 10 features representing 5 cards,
    # where each card is represented by 2 features (suit and value)
    features = []
    
    for card in cards:
        suit, value = parse_card_input(card)
        # Convert suit to numeric (S=1, H=2, D=3, C=4)
        suit_num = {"S": 1, "H": 2, "D": 3, "C": 4}[suit]
        # Convert value to numeric using the mapping
        value_num = card_value_mapping[value]
        
        features.append(suit_num)
        features.append(value_num)
    
    return features

def load_model(model_filename):
    """
    Loads a saved model from a pickle file.
    
    Parameters:
        model_filename (str): The filename of the saved model.
    
    Returns:
        model: The loaded model object.
    """
    try:
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Loaded model from {model_filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict_hand(model, features):
    """
    Predicts the poker hand class using the provided model and features.
    
    Parameters:
        model: A scikit-learn model that includes any needed preprocessing.
        features (list or array): The features representing a new poker hand.
                                  Must be in the same order and scale as the training data.
    
    Returns:
        predicted_class: The class predicted by the model.
    """
    # Convert the features to a numpy array and ensure the shape is (1, n_features)
    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

def get_user_input(expected_length):
    """
    Prompts the user to enter cards in the format 'S5, H2, D10, CQ, HA'.
    
    Parameters:
        expected_length (int): The expected number of features.
    
    Returns:
        features (list): A list of integer features.
    """
    while True:
        user_input = input(f"Please enter your 5 cards as a comma-separated list (e.g., S5, H2, D10, CQ, HA): ")
        try:
            # Split the input and strip whitespace
            cards = [card.strip() for card in user_input.split(',')]
            
            if len(cards) != 5:
                print(f"Error: Expected 5 cards, but got {len(cards)}. Try again.")
                continue
            
            # Convert cards to features
            features = convert_cards_to_features(cards)
            
            if len(features) != expected_length:
                print(f"Error: Expected {expected_length} features, but got {len(features)}. Try again.")
                continue
                
            return features
        except ValueError as e:
            print(f"Invalid input: {e}. Make sure to enter cards in the format S5, H2, etc.")
        except Exception as e:
            print(f"Error processing input: {e}. Try again.")

if __name__ == '__main__':

    def load_model(model_filename):
        try:
            with open(model_filename, 'rb') as file:
                model = pickle.load(file)
            print(f"Loaded model from {model_filename}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def predict_hand(model, features):
        # Ensure features is a numpy array of the appropriate shape.
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return prediction[0]

    # Choose your model type (example: 'gbt' for Gradient Boosting)
    model_type = 'gbt'  # Or 'svm', adjust as needed.
    if model_type == 'svm':
        model_filename = 'Models/svm_poker_model.pkl'
    elif model_type == 'gbt':
        model_filename = 'Models/gradient_boosting_weighted_poker_model_automatic.pkl'
    else:
        print("Invalid model type.")
        sys.exit(1)

    # Load the model.
    model = load_model(model_filename)

    # Get user input for cards.
    card_input = input("Please enter your 5 cards (e.g., H2, H3, H4, H5, H6): ")
    
    try:
        # Parse cards using the fixed mapping.
        user_features = parse_cards(card_input)
        print("Converted features:", user_features)
    except Exception as e:
        print(e)
        sys.exit(1)

    # Predict the poker hand class.
    predicted_class = predict_hand(model, user_features)
    print("\nUser's input:", user_features)
    print("Predicted poker hand class:", predicted_class)
