from flask import Flask, request, jsonify
from flask_cors import CORS
from Wordanalogy import analoging
app = Flask(__name__)
CORS(app)

@app.route("/api/analogy", methods=['POST'])
def analogy():
    req = request.get_json()
    sentence = req["sentence"]
    word = req["word"]
    res = analoging(sentence, word)
    return jsonify({"res" : res.to_dict()})

if __name__ == '__main__':
    app.run(debug=True)
