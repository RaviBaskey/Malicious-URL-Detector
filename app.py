import pandas as pd
import re
import joblib
from urllib.parse import urlparse
from tld import get_tld
from googlesearch import search
from flask import Flask, request, jsonify, render_template
import xgboost as xgb # It's good practice to have the library imported

# --- Feature Engineering Functions ---
# This entire section is identical to the previous version.

def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
    return 1 if match else 0

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    return 1 if match else 0

def google_index(url):
    try:
        site = search(url, 1)
        return 1 if site else 0
    except Exception as e:
        print(f"Skipping Google search for {url} due to error: {e}")
        return 0

def count_dot(url):
    return url.count('.')

def count_www(url):
    return url.count('www')

def count_atrate(url):
    return url.count('@')

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

def shortening_service(url):
    match = re.search(r'bit\\.ly|goo\\.gl|shorte\\.st|go2l\\.ink|x\\.co|ow\\.ly|t\\.co|tinyurl|tr\\.im|is\\.gd|cli\\.gs|'
                      r'yfrog\\.com|migre\\.me|ff\\.im|tiny\\.cc|url4\\.eu|twit\\.ac|su\\.pr|twurl\\.nl|snipurl\\.com|'
                      r'short\\.to|BudURL\\.com|ping\\.fm|post\\.ly|Just\\.as|bkite\\.com|snipr\\.com|fic\\.kr|loopt\\.us|'
                      r'doiop\\.com|short\\.ie|kl\\.am|wp\\.me|rubyurl\\.com|om\\.ly|to\\.ly|bit\\.do|t\\.co|lnkd\\.in|'
                      r'db\\.tt|qr\\.ae|adf\\.ly|goo\\.gl|bitly\\.com|cur\\.lv|tinyurl\\.com|ow\\.ly|bit\\.ly|ity\\.im|'
                      r'q\\.gs|is\\.gd|po\\.st|bc\\.vc|twitthis\\.com|u\\.to|j\\.mp|buzurl\\.com|cutt\\.us|u\\.bb|yourls\\.org|'
                      r'x\\.co|prettylinkpro\\.com|scrnch\\.me|filoops\\.info|vzturl\\.com|qr\\.net|1url\\.com|tweez\\.me|v\\.gd|'
                      r'tr\\.im|link\\.zip\\.net',
                      url)
    return 1 if match else 0

def count_https(url):
    return url.count('https')

def count_http(url):
    return url.count('http')

def count_per(url):
    return url.count("%")

def count_ques(url):
    return url.count('?')

def count_hyphen(url):
    return url.count('-')

def count_equal(url):
    return url.count('=')

def url_length(url):
    return len(str(url))

def hostname_length(url):
    return len(urlparse(url).netloc)

def sus_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url, re.IGNORECASE)
    return 1 if match else 0

def digit_count(url):
    return sum(c.isdigit() for c in url)

def letter_count(url):
    return sum(c.isalpha() for c in url)

def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

# --- Main Feature Extraction Pipeline ---
def extract_features(url):
    status = []
    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(google_index(url))
    status.append(count_dot(url))
    status.append(count_www(url))
    status.append(count_atrate(url))
    status.append(no_of_dir(url))
    status.append(no_of_embed(url))
    status.append(shortening_service(url))
    status.append(count_https(url))
    status.append(count_http(url))
    status.append(count_per(url))
    status.append(count_ques(url))
    status.append(count_hyphen(url))
    status.append(count_equal(url))
    status.append(url_length(url))
    status.append(hostname_length(url))
    status.append(sus_words(url))
    status.append(digit_count(url))
    status.append(letter_count(url))
    status.append(fd_length(url))
    tld = get_tld(url, fail_silently=True)
    status.append(tld_length(tld))
    return status

# --- Load Model and Define Feature Names ---
try:
    model = joblib.load('model.pkl')
    print("XGBoost model loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'model.pkl' not found. Please run your notebook to generate it.")
    model = None
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    model = None

feature_names = [
    'use_of_ip', 'abnormal_url', 'google_index', 'count.', 'count-www', 'count@',
    'count_dir', 'count_embed_domain', 'short_url', 'https_count', 'count_http',
    'count%', 'count?', 'count-', 'count=', 'url_length', 'hostname_length',
    'sus_url', 'count-digits', 'count-letters', 'fd_length', 'tld_length'
]

# --- Flask App Initialization ---
app = Flask(__name__)

# --- API Endpoints ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded, check server logs.'}), 500
        
    try:
        data = request.get_json(force=True)
        url_to_check = data['url']
        
        features_list = extract_features(url_to_check)
        features_df = pd.DataFrame([features_list], columns=feature_names)
        
        prediction = model.predict(features_df)
        label = int(prediction[0])
        
        if label == 0:
            result = "SAFE"
        elif label == 1:
            result = "DEFACEMENT"
        elif label == 2:
            result = "MALWARE"
        elif label == 3:
            result = "PHISHING"
        else:
            result = "UNKNOWN"
            
        return jsonify({'url': url_to_check, 'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Run the Application ---
if __name__ == '__main__':
    app.run(port=5000, debug=True)

