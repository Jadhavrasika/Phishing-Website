# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from urllib.parse import urlparse, parse_qs
import socket
# Import necessary sklearn components even if loaded from pickle for type hints etc.
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# --- Set Page Config FIRST ---
st.set_page_config(page_title="Phishing Detector", layout="wide", initial_sidebar_state="expanded") # Added initial_sidebar_state

# --- Feature Extraction Functions ---
# (Keep your extract_url_features_simple and extract_all_features functions here)
# ... (your feature extraction functions as before) ...
def extract_url_features_simple(url):
    """
    Extracts basic features from a URL as done in the notebook.
    THIS IS A PLACEHOLDER - FULL FEATURE EXTRACTION NEEDED.
    """
    try:
        parsed = urlparse(url)
        features = {
            'scheme': parsed.scheme,
            'netloc': parsed.netloc,
            'path': parsed.path,
            'url_length': len(url), # Note: Notebook had 'length_url' separately
            'num_path_segments': len([seg for seg in parsed.path.split('/') if seg]),
            'num_query_params': len(parse_qs(parsed.query)),
        }
        # Check if domain is an IP address
        try:
            # Ensure netloc is not empty before attempting socket.inet_aton
            if parsed.netloc:
                socket.inet_aton(parsed.netloc)
                features['is_ip'] = 1
            else:
                features['is_ip'] = 0 # Handle case where netloc is empty
        except socket.error: # Catch specific socket errors
             features['is_ip'] = 0
        except Exception: # Catch any other potential errors during IP check
             features['is_ip'] = 0 # Default to not IP if error occurs
        return features
    except Exception as e:
        st.error(f"Error parsing URL: {e}") # st.error is fine inside functions called later
        return None

def extract_all_features(url):
    """
    Placeholder function for extracting ALL required features.
    This needs to replicate the feature engineering that created
    the original dataset_phishing.csv for the relevant columns.
    """
    # --- !!! IMPLEMENT FULL FEATURE EXTRACTION HERE !!! ---
    # (Keep the full implementation details or placeholder logic here)
    basic_features = extract_url_features_simple(url)
    if basic_features is None:
        return None

    required_numerical_features = [
        'ip', 'nb_at', 'nb_qm', 'nb_underscore', 'nb_tilde', 'nb_percent',
        'nb_star', 'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www',
        'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url',
        'ratio_digits_host', 'punycode', 'port', 'tld_in_path', 'tld_in_subdomain',
        'abnormal_subdomain', 'prefix_suffix', 'random_domain', 'shortening_service',
        'path_extension', 'nb_redirection', 'nb_external_redirection', 'char_repeat',
        'shortest_word_path', 'phish_hints', 'domain_in_brand', 'brand_in_subdomain',
        'brand_in_path', 'suspecious_tld', 'statistical_report', 'nb_hyperlinks',
        'ratio_extHyperlinks', 'nb_extCSS', 'ratio_extRedirection', 'ratio_extErrors',
        'login_form', 'external_favicon', 'links_in_tags', 'ratio_intMedia',
        'ratio_extMedia', 'iframe', 'popup_window', 'safe_anchor', 'onmouseover',
        'right_clic', 'empty_title', 'domain_in_title', 'domain_with_copyright',
        'whois_registered_domain', 'domain_registration_length', 'domain_age',
        'web_traffic', 'dns_record', 'google_index', 'page_rank', 'is_ip'
    ]
    all_features = {feat: 0 for feat in required_numerical_features}
    all_features['is_ip'] = basic_features.get('is_ip', 0)
    all_features['ip'] = all_features['is_ip']
    # Add calculations for all other required_numerical_features here...

    feature_df = pd.DataFrame([all_features], columns=required_numerical_features)
    return feature_df


# --- Load Pre-trained Objects ---
# Moved the loading function definition up, but calls remain below
@st.cache_resource
def load_object(path):
    """Loads a pickled object, handling potential errors."""
    try:
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    except FileNotFoundError:
        # Use st.exception here which is slightly better form for fatal errors
        # needed for the app to function, although st.error also works.
        st.exception(f"File not found: {path}. Ensure it's saved correctly during training.")
        # Alternatively, you could raise an error or exit if these are critical
        # raise FileNotFoundError(f"Critical file not found: {path}")
        return None
    except Exception as e:
        st.exception(f"Error loading {path}: {e}")
        return None

# Load the objects *after* st.set_page_config
model = load_object("best_rf_model.pkl")
scaler = load_object("scaler.pkl")
pca = load_object("pca.pkl")


# --- Streamlit App UI ---
# (Rest of your UI code: st.title, st.markdown, columns, button, logic, sidebar)
# ... (Keep the rest of your Streamlit UI code starting from st.title onwards) ...

st.title("🎣 Phishing Website Detector")
st.markdown(
    """
    Enter a URL below to check if it's likely a phishing attempt or a legitimate website.
    This tool uses a machine learning model trained on various URL and website features.
    """
)
st.info("**Disclaimer:** This prediction is based on a machine learning model and may not be 100% accurate. Always exercise caution online.")

col1, col2 = st.columns([3, 1]) # Make input column wider

with col1:
    url_input = st.text_input("Enter URL:", "http://www.example.com", label_visibility="collapsed", placeholder="Enter URL here e.g., http://www.google.com")

with col2:
    predict_button = st.button("Analyze URL", type="primary", use_container_width=True)


if predict_button:
    # ... (Keep the prediction logic inside the button block as before) ...
    if not url_input or not url_input.startswith(('http://', 'https://')):
        st.warning("Please enter a valid URL starting with http:// or https://")
    elif not model or not scaler or not pca:
        st.error("Model or preprocessing objects could not be loaded. Cannot predict.")
    else:
        with st.spinner("Analyzing URL features..."):
            # 1. Extract Features
            input_df = extract_all_features(url_input)

            if input_df is not None and not input_df.empty:
                try:
                    # 2. Get feature names scaler was trained on
                    required_features_list = scaler.feature_names_in_

                    # 3. Ensure input_df has the correct columns in the right order
                    input_df_reindexed = input_df.reindex(columns=required_features_list, fill_value=0)

                    # 4. Scale the Features
                    scaled_features = scaler.transform(input_df_reindexed)

                    # 5. Apply PCA Transformation
                    pca_features = pca.transform(scaled_features)

                    # 6. Make Prediction
                    prediction = model.predict(pca_features)
                    prediction_proba = model.predict_proba(pca_features)

                    # 7. Display Result
                    st.subheader("Prediction Result:")
                    proba_legitimate = prediction_proba[0][0]
                    proba_phishing = prediction_proba[0][1]

                    if prediction[0] == 1: # Phishing
                        st.error(f"**Status: Phishing** (Confidence: {proba_phishing:.1%})")
                        st.progress(proba_phishing)
                    else: # Legitimate
                        st.success(f"**Status: Legitimate** (Confidence: {proba_legitimate:.1%})")
                        st.progress(proba_legitimate)

                    with st.expander("See details"):
                        st.write("Prediction Probabilities:")
                        st.write(f"- Legitimate: {proba_legitimate:.2f}")
                        st.write(f"- Phishing: {proba_phishing:.2f}")

                except AttributeError as ae:
                    if 'feature_names_in_' in str(ae):
                        st.error("Error: The loaded scaler object is missing attribute 'feature_names_in_'. It might not have been fitted or saved correctly during training.")
                    else:
                        st.error(f"An AttributeError occurred during processing: {ae}")
                except ValueError as ve:
                     st.error(f"ValueError during processing: {ve}. This often means the number or order of features extracted doesn't match what the model expects.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during prediction: {e}")
            else:
                 st.error("Could not extract features from the URL.")


# --- Sidebar ---
# (Keep sidebar code as before)
st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a **Random Forest classifier** to predict if a URL corresponds to a phishing website.

    **Model Pipeline:**
    1.  URL Feature Extraction (Lexical, Host-based, etc.)
    2.  Data Scaling (RobustScaler)
    3.  Dimensionality Reduction (PCA - 10 components)
    4.  Classification (RandomForestClassifier)

    **Note:** The feature extraction used here is a simplified placeholder. For full accuracy replicating the original model, a comprehensive feature extraction process is required.
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Notebook analysis based on dataset from [GitHub](https://raw.githubusercontent.com/Jadhavrasika/Phishing-Website/refs/heads/main/dataset_phishing.csv)")