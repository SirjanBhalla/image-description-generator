# Image Description with BLIP on Flickr30k
A deep learning project for image description generation using a fine-tuned BLIP model on a subset of the Flickr30k dataset.
Features an interactive Streamlit app for generating descriptions, along with scripts for dataset splitting, EDA, and model evaluation.

## Project Structure
```markdown
my_captioning/
│
├── final_app_without_athentication.py  # Final Streamlit app with no authentication - for Streamlit Cloud
├── final_app_with_authentication.py    # Final Streamlit app with username/password
├── app_testing.py                      # Early version - allows model switching from UI
│
├── data/
│   ├── captions.txt            # Image name + caption pairs
│   ├── training_images.txt     # List of unique images used for training
│   ├── testing_images.txt      # List of unique images for testing
│   └── Images/                 # All dataset images (flat directory)
│
├── models/
│   └── blip-ft/
│       ├── final_blip_model/   # Trained on 1k images
│       ├── final_blip_model2/  # Trained on 5k images
│       └── final_blip_model3/  # Final model trained on 20k images
│
├── train_test_split.py                           # Creates training/testing image lists from captions
├── model_eval.py                                 # Calculates BLEU score on test set
├── image_caption_generator_model_training.ipynb  # Main notebook for model training
├── requirements.txt                              # Project dependencies
└── README.md               
```       

## Setup

#### 1. Clone the repository
```terminal
git clone <repo_url>
cd my_captioning
```

#### 2. Install dependencies
```terminal
pip install -r requirements.txt
```

#### 3. Folder setup
data/
├── captions.txt
├── training_images.txt
├── testing_images.txt
└── Images/ (all dataset images)

#### 4. Model setup
Place the fine-tuned model folder final_blip_model3 in models/blip-ft/.

## Running the App

#### Non-Authentication version:
```terminal
streamlit run final_app_without_athentication.py
```

#### Authentication version:
```terminal
streamlit run final_app_with_authentication.py 
```
*(Change username/password in the script.)*

#### Model testing version:
```terminal
streamlit run app_testing.py 
```

## Dataset Info

```markdown
Total unique images: ~31,784

Captions per image: 5 (avg.)

Training set: 20,000 unique images (~100k captions)

Test set: Remaining ~11,784 images

Frequent content:
Captions often describe people, clothing colors, pets, and outdoor scenes.
```

## Evaluation

Run *eda.ipynb* to check the BLEU score for the model.

```markdown
Result for final model: ~0.33 BLEU (reasonable for image captioning)
Uses testing_images.txt as unseen images.
```

## Scripts and notebooks
1. train_test_split.py - Creates training/testing image lists from captions
2. model_eval.py - Calculates BLEU score on test set
3. image_caption_generator_model_training.ipynb - Main notebook for model training (Suggested to run this on Colab for better GPU specifications than local machine)

## Notes
1. This repo contains both authenticated and non-authenticated app versions.

2. All images are stored flat in data/Images with names matching captions.txt.

3. Notebooks (.ipynb) are optional and for project analysis, not core workflow.

4. No database — authentication is session-based in Streamlit.

*See requirements.txt for libraries required.*