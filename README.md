# CrossBar-AI-Profile-Search

Using (Facebook AI Similarity Search) FAISS, RAG, and LangChain to create an interactive social media profile website.

1. Audrey
2. Fake (Middle-age white man, career-oriented, moderately liberal)
3. Company

Final product should be in similarity to : https://aws.amazon.com/q/?nc2=h_ql_prod_fs_q

# ---
Tips of improving prediction accuracy: 

# Mistral 7B (larger, more powerful)
(https://huggingface.co/mistralai/Mistral-7B-v0.1), 

# Microsoft’s Phi-2 (smaller, more efficient)
(from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
inputs = tokenizer("Your prompt", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))), 

# LocalAI 
(# Install LocalAI (https://github.com/go-skynet/LocalAI)
docker run -p 8080:8080 localai/localai
curl http://localhost:8080/v1/completions -H "Content-Type: application/json" -d '{
  "model": "mistral-7b",
  "prompt": "Hello world",
  "max_tokens": 128
}'), 

# Ollama 
(ollama pull mistral  # Downloads model
ollama run mistral "Tell me a joke"  # Runs locally)

# ---
# Develop in Colab with fake/sample data (Language Modeling to predict the next word alike GPT, Masked Language Modeling to predict missing words alike BERT, other self-supervised tasks)

# train and fine-tune locally (VS Code) on the real corporate/personal dataset. Use Colab only for loading pre-trained models.

* Style reference: https://business.x.com/en

* Question Bank (Reading / Writing) Training: https://satsuitequestionbank.collegeboard.org/digital/results

* Celebrity Facts Sources: https://people.com/celebrity/

* My Design Page: https://www.figma.com/design/mrH47mccymIcm1sl0xXzYp/CrossBar-AI-Profile-Searech?node-id=1-2896&t=IxHPbtuy9dw7gxfp-0

GitHub Page:
1. Intro Page: [audreyllin.github.io/CrossBarAPS/homepage.html](https://audreyllin.github.io/CrossBarAPS/homepage.html)

2. Audrey's Personal Page:

3. Fake's Personal Page: [audreyllin.github.io/CrossBarAPS/fake.html](https://audreyllin.github.io/CrossBarAPS/fake.html)

4. Corporal Page:

* Simple RAG Sample Page: [audreyllin.github.io/CrossBarAPS/rag-sample.html](https://audreyllin.github.io/CrossBarAPS/rag-sample.html)

Log: combine the css for .btn, .card, :hover, :focus, :nth-child(2n), .text-center, .mt-4, .hidden, .btn.primary, .card.featured .title, #header, #main-content, div, .error, #sidebar #menu-item, .active, .current, .title, .special
(and search with " {", "," "." )

# Stages of Model Training (Till Which Stage is it "Pre-Trained"?)
