# Import dependencies
import os
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Configure the app
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("UploadFile")

# Define routes
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(
            os.path.join(
                os.path.abspath(
                    os.path.dirname(__file__)
                ),
                app.config['UPLOAD_FOLDER'],
                secure_filename(file.filename)
            )
        )
        return "File uploaded successfully"
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8000)