import re, math
from urllib.parse import urlparse
from pathlib import Path
import joblib

SUSPICIOUS_WORDS = ['login','verify','verification','secure','account','update','password','bank','payment','wallet','confirm','suspended','urgent','bonus','gift','claim','invoice','refund','signin']
SHORTENERS = {'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd','cutt.ly','rb.gy'}


def features(url):
    raw=url if re.match(r'^[a-zA-Z][\w+.-]*://',url) else 'http://'+url
    p=urlparse(raw); host=p.hostname or ''
    path=p.path or ''
    query=p.query or ''
    return [
        len(url), len(host), len(path), len(query),
        url.count('.'), url.count('-'), url.count('@'), url.count('?'), url.count('='), url.count('&'),
        int(p.scheme.lower()=='https'), int(re.match(r'^\d{1,3}(?:\.\d{1,3}){3}$',host or '') is not None),
        host.count('.'), len(host.split('.')), int(host.startswith('xn--') or '.xn--' in host),
        sum(url.lower().count(w) for w in SUSPICIOUS_WORDS), int(host.lower() in SHORTENERS),
        int('//' in path), int('@' in url), int(p.username is not None)
    ]

class PhishingDetector:
    def __init__(self, model_path: Path):
        self.model=None
        if model_path.exists(): self.model=joblib.load(model_path)

    def predict(self,url):
        normalized=url if re.match(r'^[a-zA-Z][\w+.-]*://',url) else 'https://'+url
        p=urlparse(normalized); host=(p.hostname or '').lower(); low=url.lower()
        reasons=[]; score=0
        if p.scheme!='https': score+=18; reasons.append('Connection does not use HTTPS')
        if '@' in url: score+=25; reasons.append('URL contains @, which can obscure the real destination')
        if re.match(r'^\d{1,3}(?:\.\d{1,3}){3}$',host): score+=25; reasons.append('Hostname is an IP address')
        if any(w in low for w in SUSPICIOUS_WORDS): score+=18; reasons.append('Contains common phishing-related terms')
        if host in SHORTENERS: score+=15; reasons.append('Uses a URL-shortening service')
        if host.count('.')>=3: score+=10; reasons.append('Unusually deep subdomain structure')
        if len(url)>100: score+=8; reasons.append('Unusually long URL')
        if host.startswith('xn--') or '.xn--' in host: score+=15; reasons.append('Punycode/internationalized hostname detected')
        score=min(score,100)
        ml_conf=None
        if self.model:
            try:
                proba=self.model.predict_proba([features(url)])[0]
                # label convention: 1 = phishing
                classes=list(self.model.classes_); phishing_idx=classes.index(1) if 1 in classes else -1
                ml_conf=float(proba[phishing_idx]) if phishing_idx>=0 else None
            except Exception: ml_conf=None
        if ml_conf is not None:
            final=round(100*ml_conf)
            score=round((score+final)/2)
            reasons.insert(0,f'ML model phishing probability: {final}%')
        if score>=70: verdict='PHISHING'
        elif score>=40: verdict='SUSPICIOUS'
        else: verdict='SAFE'
        confidence=round(max(score,100-score)/100*100,1)
        if ml_conf is not None: confidence=round(max(ml_conf,1-ml_conf)*100,1)
        return {'url':url,'normalized_url':normalized,'verdict':verdict,'risk_score':score,'confidence':confidence,'reasons':reasons or ['No major phishing indicators found']}
