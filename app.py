import os
from flask import Flask, render_template, request

# --- Paste the SelfLearningAI class and DUMMY_INTERNET dictionary here ---
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

DUMMY_INTERNET = {
    "Machine Learning Applications": "<html><body><p>Machine learning (ML) is key in recommendation systems and natural language processing (NLP).</p></body></html>",
    "Quantum Computing Basics": "<html><body><p>Quantum computing uses qubits, which can be in a superposition of states.</p></body></html>",
    "The History of Python": "<html><body><p>Python was conceived in the late 1980s by Guido van Rossum.</p></body></html>",
    "Neural Networks Explained": "<html><body><p>Artificial Neural Networks are inspired by the biological networks in animal brains.</p></body></html>"
}

class SelfLearningAI:
    def __init__(self):
        print("Initializing AI agent...")
        model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.knowledge_chunks = []
        self.vectorizer = TfidfVectorizer()
        self.knowledge_vectors = None
        print("AI Agent is ready.")

    def learn_from_content(self, html_content: str):
        soup = BeautifulSoup(html_content, 'html.parser')
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text()]
        for p_text in paragraphs:
            if p_text not in self.knowledge_chunks:
                self.knowledge_chunks.append(p_text)
        if self.knowledge_chunks:
            self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_chunks)

    def answer_question(self, question: str) -> str:
        if not self.knowledge_chunks: return "I have not learned anything yet."
        query_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vector, self.knowledge_vectors).flatten()
        context = self.knowledge_chunks[similarities.argsort()[-1]]
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        output_sequences = self.model.generate(
            **inputs, max_length=len(inputs['input_ids'][0]) + 50, num_return_sequences=1,
            no_repeat_ngram_size=2, pad_token_id=self.tokenizer.eos_token_id
        )
        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        answer_marker = "Answer:"
        answer_position = generated_text.find(answer_marker)
        if answer_position != -1:
            return generated_text[answer_position + len(answer_marker):].strip()
        return "Could not generate a formatted answer."

# --- Flask Web App Setup ---
app = Flask(__name__)
print("Creating AI instance...")
ai_system = SelfLearningAI()

# --- AI LEARNING PHASE (runs once on startup) ---
print("Starting initial learning phase...")
for topic, content in DUMMY_INTERNET.items():
    print(f"Learning about: {topic}")
    ai_system.learn_from_content(content)
print("Initial learning complete. AI is online.")
# --- END LEARNING PHASE ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    if not question:
        return render_template('index.html', error="Please enter a question.")
    answer = ai_system.answer_question(question)
    return render_template('result.html', question=question, answer=answer)

if __name__ == '__main__':
    # This part is for local testing only. Render and other services will use Gunicorn.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

