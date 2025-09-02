from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/home", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Nisaa Chatbot."}), 200

@app.route("/get-suggestion", methods=["POST"])
def suggest():
    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
