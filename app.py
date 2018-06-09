import os
from collections import Counter
from flask import Flask, request, redirect, url_for, send_from_directory,render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from pylab import rcParams
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder


UPLOAD_FOLDER = 'flask-upload-test'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


data_file = 'flask-upload-test/data.csv'
df = pd.read_csv(data_file)
enc_df = pd.read_csv(data_file)
label_encoder = LabelEncoder()
cat_columns = df.dtypes.pipe(lambda x: x[x == 'object']).index
for col in cat_columns:
    df[col] = label_encoder.fit_transform(df[col])

loaded_model = joblib.load('pickle_model.pkl')
result = loaded_model.predict(df)
pred_series = pd.Series(result.tolist())
pred_series = pred_series.rename('Predicated Value')

final_df = enc_df.merge(pred_series.to_frame(), left_index=True, right_index=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print ('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print ('no filename')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = "data.csv" # secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

    return render_template("index.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/show', methods=['GET'])
def show_df():
    data_file = 'flask-upload-test/data.csv'
    df = pd.read_csv(data_file)
    to_html = "<!doctype html><title>Show DF</title>"+df.to_html()
    return to_html


@app.route('/html', methods=['GET'])
def html():
    global my_list
    return '''
    <!doctype html>
    
    <title>HTML</title>
    
    <h1>CSS</h1>
    <p>
    dfsifgdsifds
    </p>
    '''


@app.route('/plot', methods=['GET'])
def plot():
    data_file = 'flask-upload-test/data.csv'
    df = pd.read_csv(data_file)
    img = io.BytesIO()
    a, x = plt.subplots(figsize=(16, 10))
    sns.countplot(x="Topic", data=df, palette="muted")
    plt.title('Correlation between different features', y=1.03, size=18)
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return '<img src="data:image/png;base64,{}">'.format(plot_url) + '''<a href="/abs> </a>'''


@app.route('/abs', methods=['Get'])
def plot_count():
    img = io.BytesIO()
    sns.countplot(x='StudentAbsenceDays', data=final_df, hue='Predicated Value', palette='dark')
    plt.savefig(img, format='png')
    img.seek(0) # rewind to beginning of file
    plot_url = base64.b64encode(img.getvalue()).decode()

    return '<img src="data:image/png;base64,{}">'.format(plot_url) + ''' <!doctype html>
    
    <title>HTML</title>
    
    <p> We See that students which Absence days more than 7 has low grades </p>
    
    '''


@app.route('/encode', methods=['GET'])
def encode():
    data_file = 'flask-upload-test/data.csv'
    df = pd.read_csv(data_file)
    label_encoder = LabelEncoder()
    cat_columns = df.dtypes.pipe(lambda x: x[x == 'object']).index
    for col in cat_columns:
        df[col] = label_encoder.fit_transform(df[col])

    to_html_predict = "<!doctype html><title>Show Encoded DataFrame Values </title>" + df.to_html()
    return to_html_predict


@app.route('/predict', methods=['GET'])
def predict():
    data_file = 'flask-upload-test/data.csv'
    df = pd.read_csv(data_file)
    enc_df = pd.read_csv(data_file)
    label_encoder = LabelEncoder()
    cat_columns = df.dtypes.pipe(lambda x: x[x == 'object']).index
    for col in cat_columns:
        df[col] = label_encoder.fit_transform(df[col])

    loaded_model = joblib.load('pickle_model.pkl')
    result = loaded_model.predict(df)
    pred_series = pd.Series(result.tolist())
    pred_series = pred_series.rename('Predicated Value')

    final_df = enc_df.merge(pred_series.to_frame(), left_index=True, right_index=True)

    letter_counts = Counter(result)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    ax = df.plot(kind='bar', color=tuple(["g", "b", "r"]), legend=False,figsize=(15,8))
    patches, labels = ax.get_legend_handles_labels()
    ax.legend(patches, labels, loc='best')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    to_html_predict = "<!doctype html><title> Show Predicated Values </title>" + final_df.to_html()

    return '<h1> if you to see all the dataset with the predicted values press here  </h1> <a href="/predictDF"> <input type="button" value="Predict"></a>  <p> </p><img src="data:image/png;base64,{}">'.format(plot_url)


    #print(to_html_predict)

  #  return to_html_predict



@app.route('/predictDF', methods=['GET'])
def predict_df():
    data_file = 'flask-upload-test/data.csv'
    df = pd.read_csv(data_file)
    enc_df = pd.read_csv(data_file)
    label_encoder = LabelEncoder()
    cat_columns = df.dtypes.pipe(lambda x: x[x == 'object']).index
    for col in cat_columns:
        df[col] = label_encoder.fit_transform(df[col])

    loaded_model = joblib.load('pickle_model.pkl')
    result = loaded_model.predict(df)
    pred_series = pd.Series(result.tolist())
    pred_series = pred_series.rename('Predicated Value')
    final_df = enc_df.merge(pred_series.to_frame(), left_index=True, right_index=True)
    first_half_df = final_df[:20]
    to_html_predict = "<!doctype html><title> Show Predicated Values </title>" + first_half_df.to_html()
    print(to_html_predict)
    return to_html_predict


if __name__ == "__main__":
    app.debug = True
    app.run()
