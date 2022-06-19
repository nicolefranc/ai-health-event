# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, jsonify, render_template, request
from distilbert import predict, convert_label

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.


@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def index():
    return render_template('index.html')


@app.route('/output', methods=['POST'])
def output():
    print(request.form)
    input_model = request.form['input_model']
    input_text = request.form['input_text']
    # iterate through series of actions
    predicted_label = predict(input_model, input_text)
    predicted_label = convert_label(predicted_label)
    # out = f'The tweet is categorised as {predicted_label.upper()}.'
    out = jsonify({'text': input_text, 'prediction': predicted_label})
    print('out', out)
    return out


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True)
