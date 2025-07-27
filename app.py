import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import joblib
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor
from skimage.morphology import skeletonize
from skimage.util import invert

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'signature_system.pkl'
SCALER_PATH = 'scaler_signature.pkl'    

app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    SECRET_KEY='replace-with-secure-key'
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

clf    = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256,256))
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    feats = []

    hu = cv2.HuMoments(cv2.moments(bw)).flatten()
    feats.extend(np.log1p(np.abs(hu)))

    hog_fd = hog(bw, pixels_per_cell=(16,16), cells_per_block=(2,2),feature_vector=True)
    feats.extend(hog_fd)

    lbp = local_binary_pattern(bw, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(),bins=np.arange(0,11), range=(0,10))
    feats.extend((hist/(hist.sum()+1e-6)).tolist())

    try:
        skel = cv2.ximgproc.thinning(bw)
        feats.append(float(np.sum(skel>0)))
    except:
        feats.append(0.0)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area    = cv2.contourArea(c)
        perim   = cv2.arcLength(c, True)
        x,y,w,h = cv2.boundingRect(c)
        aspect  = float(w)/h if h>0 else 0.0
        extent  = float(area)/(w*h) if w*h>0 else 0.0
        hull    = cv2.convexHull(c)
        hull_A  = cv2.contourArea(hull)
        solidity= float(area)/hull_A if hull_A>0 else 0.0
    else:
        area=perim=aspect=extent=solidity=0.0

    feats.extend([area, perim, aspect, extent, solidity])
    return np.array(feats, dtype=np.float32)

@app.route('/', methods=['GET','POST'])
def index():
    result, filename, confidence = None, None, None

    if request.method == 'POST':
        file = request.files.get('signature')
        if not file or file.filename == '':
            flash('Please select a signature image.')
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            feats = np.array([extract_features(save_path)] )

            feats = scaler.transform(feats)

            prob = clf.predict_proba(feats)[0]  
            pred = np.argmax(prob)
            confidence = round(prob[pred] * 100, 2)
            result = 'Genuine' if pred == 1 else 'Forged'


    return render_template('index.html', result=result, filename=filename, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
