from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load a pre-trained model for conversational AI
chatbot = pipeline('conversational', model='facebook/blenderbot-400M-distill')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Get a response from the chatbot
    response = chatbot(user_message)
    chatbot_reply = response[0]['generated_text']

    return jsonify({'reply': chatbot_reply})

if __name__ == '__main__':
    app.run(port=5000)
