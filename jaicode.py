import subprocess
import sys

# Function to install a package via pip
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try importing packages and install if missing
try:
    import pandas as pd
except ImportError:
    print("pandas not found, installing...")
    install_package("pandas")
    import pandas as pd

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    print("scikit-learn not found, installing...")
    install_package("scikit-learn")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

import re

# Feature extraction function
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['count_dots'] = url.count('.')
    features['count_hyphens'] = url.count('-')
    features['count_at'] = url.count('@')
    features['count_question'] = url.count('?')
    features['count_equal'] = url.count('=')
    features['count_digits'] = sum(c.isdigit() for c in url)
    
    # Detect IP address in URL
    ip_pattern = re.compile(
        r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}'
    )
    features['has_ip'] = 1 if ip_pattern.search(url) else 0
    
    return features

# Sample data: list of tuples (url, label)
data = [
    ("http://example.com", 0),
    ("http://192.168.0.1/login", 1),
    ("http://free-money-now.com/login", 1),
    ("https://trustedsite.org", 0),
    ("http://malicious-site.com/steal-info", 1),
    ("https://secure.bank.com", 0)
]

# Create DataFrame and extract features
df = pd.DataFrame(data, columns=['url', 'label'])
features_list = df['url'].apply(extract_features)
features_df = pd.DataFrame(features_list.tolist())
dataset = pd.concat([features_df, df['label']], axis=1)

# Split dataset
X = dataset.drop('label', axis=1)
y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Test new URL
new_url = "http://suspicious-site.com/login"
new_features = pd.DataFrame([extract_features(new_url)])
prediction = clf.predict(new_features)[0]
print(f"URL: {new_url} is {'Malicious' if prediction == 1 else 'Benign'}")
