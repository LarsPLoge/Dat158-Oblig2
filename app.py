import gradio as gr
import pickle
import pandas as pd
from pathlib import Path

# Load the trained model and preprocessor
model_path = Path(__file__).parent / 'mushroom_model.pkl'
preprocessor_path = Path(__file__).parent / 'mushroom_preprocessor.pkl'
feature_names_path = Path(__file__).parent / 'mushroom_features.pkl'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Model files not found. Please run train_mushroom_model.py first.")
    model = None
    preprocessor = None
    feature_names = None

# Define feature options based on the dataset
feature_options = {
    'cap-shape': ['b (bell)', 'c (conical)', 'x (convex)', 'f (flat)', 'k (knobbed)', 's (sunken)'],
    'cap-surface': ['f (fibrous)', 'g (grooves)', 'y (scaly)', 's (smooth)'],
    'cap-color': ['n (brown)', 'b (buff)', 'c (cinnamon)', 'g (gray)', 'r (green)', 'p (pink)', 
                  'u (purple)', 'e (red)', 'w (white)', 'y (yellow)'],
    'bruises': ['t (bruises)', 'f (no)'],
    'odor': ['a (almond)', 'l (anise)', 'c (creosote)', 'y (fishy)', 'f (foul)', 
             'm (musty)', 'n (none)', 'p (pungent)', 's (spicy)'],
    'gill-attachment': ['a (attached)', 'f (free)'],
    'gill-spacing': ['c (close)', 'w (crowded)'],
    'gill-size': ['b (broad)', 'n (narrow)'],
    'gill-color': ['k (black)', 'n (brown)', 'b (buff)', 'h (chocolate)', 'g (gray)', 'r (green)',
                   'o (orange)', 'p (pink)', 'u (purple)', 'e (red)', 'w (white)', 'y (yellow)'],
    'stalk-shape': ['e (enlarging)', 't (tapering)'],
    'stalk-root': ['b (bulbous)', 'c (club)', 'e (equal)', 'r (rooted)'],
    'stalk-surface-above-ring': ['f (fibrous)', 'y (scaly)', 'k (silky)', 's (smooth)'],
    'stalk-surface-below-ring': ['f (fibrous)', 'y (scaly)', 'k (silky)', 's (smooth)'],
    'stalk-color-above-ring': ['n (brown)', 'b (buff)', 'c (cinnamon)', 'g (gray)', 'o (orange)',
                                'p (pink)', 'e (red)', 'w (white)', 'y (yellow)'],
    'stalk-color-below-ring': ['n (brown)', 'b (buff)', 'c (cinnamon)', 'g (gray)', 'o (orange)',
                                'p (pink)', 'e (red)', 'w (white)', 'y (yellow)'],
    'veil-type': ['p (partial)'],
    'veil-color': ['n (brown)', 'o (orange)', 'w (white)', 'y (yellow)'],
    'ring-number': ['n (none)', 'o (one)', 't (two)'],
    'ring-type': ['e (evanescent)', 'f (flaring)', 'l (large)', 'n (none)', 'p (pendant)'],
    'spore-print-color': ['k (black)', 'n (brown)', 'b (buff)', 'h (chocolate)', 'r (green)',
                          'o (orange)', 'u (purple)', 'w (white)', 'y (yellow)'],
    'population': ['a (abundant)', 'c (clustered)', 'n (numerous)', 's (scattered)',
                   'v (several)', 'y (solitary)'],
    'habitat': ['g (grasses)', 'l (leaves)', 'm (meadows)', 'p (paths)', 'u (urban)', 
                'w (waste)', 'd (woods)']
}

def predict_mushroom(cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment,
                    gill_spacing, gill_size, gill_color, stalk_shape, stalk_root,
                    stalk_surface_above_ring, stalk_surface_below_ring, 
                    stalk_color_above_ring, stalk_color_below_ring, veil_type,
                    veil_color, ring_number, ring_type, spore_print_color,
                    population, habitat):
    """
    Predict if mushroom is edible or poisonous based on input features
    """
    if model is None or preprocessor is None:
        return "❌ **Error:** Model not loaded. Please run `train_mushroom_model.py` first."
    
    # Extract the single letter code from the selection (format: "x (description)")
    def extract_code(value):
        return value.split(' ')[0] if value else value
    
    # Create input dataframe with proper feature names
    input_data = pd.DataFrame([[
        extract_code(cap_shape),
        extract_code(cap_surface),
        extract_code(cap_color),
        extract_code(bruises),
        extract_code(odor),
        extract_code(gill_attachment),
        extract_code(gill_spacing),
        extract_code(gill_size),
        extract_code(gill_color),
        extract_code(stalk_shape),
        extract_code(stalk_root),
        extract_code(stalk_surface_above_ring),
        extract_code(stalk_surface_below_ring),
        extract_code(stalk_color_above_ring),
        extract_code(stalk_color_below_ring),
        extract_code(veil_type),
        extract_code(veil_color),
        extract_code(ring_number),
        extract_code(ring_type),
        extract_code(spore_print_color),
        extract_code(population),
        extract_code(habitat)
    ]], columns=feature_names)
    
    # Preprocess and predict
    input_encoded = preprocessor.transform(input_data)
    prediction = model.predict(input_encoded)[0]
    
    # Format the result
    if prediction == 'e':
        result = """
        # 🍄 ✅ **EDIBLE**
        
        ### The mushroom is predicted to be **EDIBLE**
        
        ⚠️ **Important Disclaimer:** This is a machine learning prediction and should NOT be used as the sole 
        basis for consuming wild mushrooms. Always consult with a mycology expert before consuming any wild mushroom.
        """
        color = "green"
    else:
        result = """
        # ☠️ ⚠️ **POISONOUS**
        
        ### The mushroom is predicted to be **POISONOUS**
        
        🚨 **DO NOT EAT** this mushroom based on these characteristics!
        """
        color = "red"
    
    return result

# Create Gradio interface with all 22 features
demo = gr.Interface(
    fn=predict_mushroom,
    inputs=[
        gr.Dropdown(choices=feature_options['cap-shape'], label="Cap Shape", value='x (convex)'),
        gr.Dropdown(choices=feature_options['cap-surface'], label="Cap Surface", value='s (smooth)'),
        gr.Dropdown(choices=feature_options['cap-color'], label="Cap Color", value='n (brown)'),
        gr.Dropdown(choices=feature_options['bruises'], label="Bruises", value='t (bruises)'),
        gr.Dropdown(choices=feature_options['odor'], label="Odor", value='n (none)'),
        gr.Dropdown(choices=feature_options['gill-attachment'], label="Gill Attachment", value='f (free)'),
        gr.Dropdown(choices=feature_options['gill-spacing'], label="Gill Spacing", value='c (close)'),
        gr.Dropdown(choices=feature_options['gill-size'], label="Gill Size", value='n (narrow)'),
        gr.Dropdown(choices=feature_options['gill-color'], label="Gill Color", value='k (black)'),
        gr.Dropdown(choices=feature_options['stalk-shape'], label="Stalk Shape", value='e (enlarging)'),
        gr.Dropdown(choices=feature_options['stalk-root'], label="Stalk Root", value='e (equal)'),
        gr.Dropdown(choices=feature_options['stalk-surface-above-ring'], label="Stalk Surface Above Ring", value='s (smooth)'),
        gr.Dropdown(choices=feature_options['stalk-surface-below-ring'], label="Stalk Surface Below Ring", value='s (smooth)'),
        gr.Dropdown(choices=feature_options['stalk-color-above-ring'], label="Stalk Color Above Ring", value='w (white)'),
        gr.Dropdown(choices=feature_options['stalk-color-below-ring'], label="Stalk Color Below Ring", value='w (white)'),
        gr.Dropdown(choices=feature_options['veil-type'], label="Veil Type", value='p (partial)'),
        gr.Dropdown(choices=feature_options['veil-color'], label="Veil Color", value='w (white)'),
        gr.Dropdown(choices=feature_options['ring-number'], label="Ring Number", value='o (one)'),
        gr.Dropdown(choices=feature_options['ring-type'], label="Ring Type", value='p (pendant)'),
        gr.Dropdown(choices=feature_options['spore-print-color'], label="Spore Print Color", value='k (black)'),
        gr.Dropdown(choices=feature_options['population'], label="Population", value='s (scattered)'),
        gr.Dropdown(choices=feature_options['habitat'], label="Habitat", value='u (urban)'),
    ],
    outputs=gr.Markdown(label="Prediction Result"),
    title="🍄 Mushroom Edibility Classifier",
    description="""
    ### Predict if a mushroom is edible or poisonous based on its characteristics
    
    This classifier uses a trained SGD (Stochastic Gradient Descent) model on the UCI Mushroom dataset.
    Select the characteristics of the mushroom you want to classify.
    
    **⚠️ WARNING:** This is for educational purposes only. Never consume wild mushrooms based solely on machine learning predictions!
    
    **Note:** If you see an error, please run `python train_mushroom_model.py` first to train the model.
    """,
    examples=[
        # Example 1: Typical edible mushroom
        ['x (convex)', 's (smooth)', 'n (brown)', 't (bruises)', 'n (none)', 'f (free)', 
         'c (close)', 'b (broad)', 'k (black)', 'e (enlarging)', 'e (equal)', 's (smooth)',
         's (smooth)', 'w (white)', 'w (white)', 'p (partial)', 'w (white)', 'o (one)',
         'p (pendant)', 'k (black)', 's (scattered)', 'u (urban)'],
        # Example 2: Typical poisonous mushroom
        ['x (convex)', 'y (scaly)', 'w (white)', 't (bruises)', 'p (pungent)', 'f (free)',
         'c (close)', 'n (narrow)', 'k (black)', 'e (enlarging)', 'e (equal)', 's (smooth)',
         's (smooth)', 'w (white)', 'w (white)', 'p (partial)', 'w (white)', 'o (one)',
         'p (pendant)', 'w (white)', 's (scattered)', 'g (grasses)'],
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=False)
