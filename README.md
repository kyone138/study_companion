# Document-Grounded Conversational Study Assistant (RAG-Based Chatbot)

## Overview

The rise of Large Language Models (LLMs) has transformed how students interact with technology in their learning process. While tools like ChatGPT are powerful, they suffer from key limitations in educational settings:
- Hallucinated or outdated information
- Lack of grounding in course-specific materials
- Overly broad scope that conflicts with academic integrity policies

This project explores a **document-grounded conversational study assistant** that allows students to interact *only* with materials they provide. By combining **Retrieval-Augmented Generation (RAG)** with a lightweight yet powerful language model, we build a chatbot that acts as a **focused, reliable, and interactive study companion**.

---

## Key Idea

Instead of relying on the model’s parametric knowledge, the system:
1. Accepts user-provided documents (lecture slides, research papers, resumes, notes)
2. Retrieves relevant passages using vector search
3. Generates answers grounded strictly in the retrieved content
4. Maintains conversational context across multiple turns

This design reduces hallucinations and ensures responses remain relevant to the student’s actual coursework.

---

## System Architecture

### Core Components

- **Language Model:** Mistral-7B  
- **Retrieval Framework:** LangChain Retrieval-Augmented Generation (RAG)
- **Vector Store:** FAISS (Maximum Inner Product Search)
- **Conversation Memory:** Chat history integration
- **Prompt Engineering:** Instruction-tuned conversational prompt

---

## Why Mistral-7B?

We selected **Mistral-7B** due to its strong performance-to-efficiency ratio:

- Only **7B parameters**, making it suitable for limited-resource environments
- Outperforms LLaMA-2 (13B) and is comparable to LLaMA-1 (34B)
- Generates more coherent and consistent responses in our experiments

### Architectural Optimizations

#### Sliding Window Attention
- Reduces attention complexity from **O(n²)** to **O(n·w)**
- Preserves long-range dependencies while lowering memory usage

#### Grouped Query Attention
- Groups similar queries to reduce redundant computation
- Achieves significant speedups with minimal quality degradation

These efficiencies make Mistral-7B ideal for interactive applications like conversational agents.

---

## Retrieval-Augmented Generation (RAG)

To ground responses in user-provided content:

1. Documents are split into passages
2. Passages are embedded and stored in a FAISS index
3. Relevant passages are retrieved at query time
4. Retrieved text is concatenated with the user query
5. The model generates a response using only this context

This enables:
- Accurate answers to document-specific questions
- Reduced hallucinations
- Low-latency responses suitable for real-time interaction

---

## Prompt Engineering

We used iterative prompt engineering to ensure responses were:
- Conversational
- Helpful and explanatory
- Grounded in retrieved content

### Final Prompt Template

```text
You are a friendly study tutor chatbot that has access to a database of documents
provided by the students. Use the chat history and your existing knowledge to
answer the follow up question in a helpful and friendly way.
Make sure your tone is that of a friendly study buddy.
