import pandas as pd
import os

# --- Configuration ---
# Path to your captions file. Make sure it's correct.
CAPTIONS_FILE = './data/captions.txt' 
# The number of UNIQUE images you want for training.
TRAINING_SIZE = 20000

if not os.path.exists(CAPTIONS_FILE):
    print(f"Error: '{CAPTIONS_FILE}' not found. Please make sure the path is correct.")
else:
    # --- Step 1: Load the data and find ALL unique image names ---
    df = pd.read_csv(CAPTIONS_FILE, sep=',', header=None, names=['image', 'caption'])
    all_unique_images = df['image'].unique()
    total_unique_count = len(all_unique_images)
    
    print(f"Step 1 Complete: Found {total_unique_count} total unique images in the dataset.")

    # --- Step 2: Split the list of unique images into training and testing sets ---
    if TRAINING_SIZE > total_unique_count:
        print(f"Error: Your requested training size ({TRAINING_SIZE}) is larger than the total number of unique images available ({total_unique_count}).")
    else:
        training_images = all_unique_images[:TRAINING_SIZE]
        testing_images = all_unique_images[TRAINING_SIZE:]
        
        # --- Step 3: Save the lists to files ---
        with open('training_images.txt', 'w') as f:
            for img in sorted(training_images):
                f.write(img + '\n')
                
        with open('testing_images.txt', 'w') as f:
            for img in sorted(testing_images):
                f.write(img + '\n')
                
        # --- Final Summary ---
        print("\n✅ Success! Dataset has been split correctly.")
        print(f" -> {len(training_images)} unique images were selected for training (see 'training_images.txt')")
        print(f" -> {len(testing_images)} unique images are available for testing (see 'testing_images.txt')")
        print(f" -> Total unique images in the dataset: {total_unique_count}")