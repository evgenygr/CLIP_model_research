from io import BytesIO
import json
import numpy as np
import scipy as sp
from flask import Flask, render_template, request
import logging

from ml_tools import get_embedding


with open("master_vectors/metadata.json") as f:
    mv_metadata = json.load(f)
mv_descrs = [el['description'] for el in mv_metadata]
mv_dict = {
    el['description']: np.loadtxt(f'master_vectors/{el["filename"]}', delimiter=',')
    for el in mv_metadata
}

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

ALLOWED_EXTENSIONS = {'jpg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html', string_values=mv_descrs)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file1' not in request.files:
        return 'No file1 part'
    if 'file2' not in request.files:
        return 'No file2 part'

    file1 = request.files['file1']
    file2 = request.files['file2']
    master_vector = mv_dict[request.form['selected_string']]
    logging.debug(f'Master-vector dimensions: {master_vector.shape}')

    if file1.filename == '' or file2.filename == '':
        return 'Both files are required'

    logging.debug(f'Filename 1: {file1.filename}')
    logging.debug(f'Filename 2: {file2.filename}')

    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        emb_image1 = get_embedding(BytesIO(file1.read()))
        emb_image2 = get_embedding(BytesIO(file2.read()))

        logging.debug(f'image 1 embedding dimensions: {emb_image1.shape}')
        logging.debug(f'image 2 embedding dimensions: {emb_image2.shape}')

        # Process the file
        probs = sp.special.softmax(
            [emb_image1 @ master_vector.T, emb_image2 @ master_vector.T]
        )

        logging.debug(f'Probabilities: {probs}')

        # Pass the results to the template
        return render_template(
            'result.html',
            filename1=file1.filename, probability1=probs[0],
            filename2=file2.filename, probability2=probs[1],
            method=request.form['selected_string']
        )

    return 'Invalid file type'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
