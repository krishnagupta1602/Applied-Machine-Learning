from flask import Flask, request, render_template, url_for, redirect
# import joblib
# from sklearn.externals import joblib
import pickle
import score
# import pickle

app = Flask(__name__)

filename = open("model_3",'rb')
mlp =pickle.load(filename)

threshold=0.5


@app.route('/') 
def home():
    return render_template('spam.html')


@app.route('/spam', methods=['POST'])
def spam():
    sentence = request.form['sent']
    label, prop_score =score.score(sentence,mlp,threshold)
    label="Spam" if label == 1 else "not a spam"
    ans = f"""The sentence "{sentence}" is {label} with propensity{prop_score}."""
    return render_template('res.html', ans=ans)


if __name__ == '__main__': 
    app.run(debug=True)