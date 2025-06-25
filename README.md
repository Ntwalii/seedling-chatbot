# Seedling AI

Seedling AI is a domain-specific chatbot built using Hugging Face Transformers, TensorFlow, FastAPI, and React. It’s designed to respond to questions about seedlings and plant care in a conversational format.

### [Model deployed on hugging face](https://huggingface.co/aub1n/seedling-ai/tree/main)

## Features

- Chatbot interface built with React
- FastAPI backend serving a fine-tuned Hugging Face model
- TensorFlow-based inference pipeline
- Trained on domain-specific seedling Q&A data
- Supports interactive messaging between user and bot

## Model Training

Fine-tuned for 3 epochs using a small seedling-related dataset  
Learning Rate: `2e-5`  
Batch Size: `16`  
Manual evaluation showed significantly improved domain-relevant answers.

Notebook includes preprocessing, tokenizer setup, training loop, and testing outputs.


## Installation & Running Locally

### Backend (FastAPI)

1. Navigate to the server directory:
    ```bash
    cd server
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```

The server runs at `http://127.0.0.1:8000`.


### Frontend (React)

1. Navigate to the client directory:
    ```bash
    cd client
    ```

2. Install dependencies:
    ```bash
    npm install
    ```

3. Start the development server:
    ```bash
    npm run dev
    ```

Frontend will run at `http://localhost:5173`.


## Notebook

The notebook inside `/notebook` includes:

- Data cleaning & formatting
- Tokenization using Hugging Face
- Fine-tuning code (using TensorFlow/Keras)
- Output samples before and after training

Useful for understanding how the model was prepared and improved.


## 📹 Demo

[Video demo](https://www.youtube.com/watch?reload=9&v=p60B8vc_hVQ)
