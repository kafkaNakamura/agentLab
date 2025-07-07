import random
from datasets import load_dataset

def prepare_uiim_training_data(dataset_id, split):
    print(f"Loading dataset: {dataset_id} with split: {split}")
    dataset = load_dataset(dataset_id, split=split)
    
    prepared_data = []
    intent_labels = ["Benign", "Probing/Exploratory", "Malicious (Subtle)", "Malicious (Overt)"]

    print(f"Processing {len(dataset)} examples...")
    for i, example in enumerate(dataset):
        # The 'dialogue' key holds a list of alternating conversation turns
        # e.g., ["user_utterance_1", "agent_utterance_1", "user_utterance_2", ...]
        # We assume turns alternate, starting with a user turn at index 0.
        conversation_turns = example['dialogue']
        
        current_history = ""
        for turn_index, turn_text in enumerate(conversation_turns):
            # User turns are at even indices (0, 2, 4...)
            is_user_turn = (turn_index % 2 == 0)
            
            # For a user turn, the current_history represents the context *before* this user turn.
            if is_user_turn:
                simulated_intent_label = random.choice(intent_labels)
                
                prepared_data_point = {
                    "conversational_history": current_history.strip(), # History *before* the current user turn
                    "user_turn_text": turn_text, # The text of the current user turn itself
                    "intent_label": simulated_intent_label
                }
                prepared_data.append(prepared_data_point)
                
            # Append the current turn's content to the history for the *next* turn's context
            # Label turns as "user" or "agent" in the history for clarity.
            role_prefix = "user: " if is_user_turn else "agent: "
            current_history += f"{role_prefix}{turn_text}\n"
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples.")

    print(f"Finished processing. Total prepared data points: {len(prepared_data)}")
    
    # Print a few sample data points to verify the structure
    print("\nSample prepared data points:")
    for j in range(min(5, len(prepared_data))):
        print(f"--- Sample {j+1} ---")
        print(f"Conversational History: '{prepared_data[j]['conversational_history']}'")
        print(f"User Turn Text: '{prepared_data[j]['user_turn_text']}'")
        print(f"Intent Label: '{prepared_data[j]['intent_label']}'")
        print("-" * 20)

    return prepared_data

# Example usage: This part will be executed to demonstrate the function.
dataset_id_to_use = "dream"
split_to_use = "train"

# Call the function to prepare the data
uiim_data = prepare_uiim_training_data(dataset_id_to_use, split_to_use)