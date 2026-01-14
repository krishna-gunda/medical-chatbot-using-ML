# 🩺 Medical Assistance Chatbot using Machine Learning

## Overview

The **Medical Assistance Chatbot** is an AI-powered web application designed to provide quick, reliable, and accessible health-related guidance. It uses **Machine Learning** and **Natural Language Processing (NLP)** to understand user symptoms, predict possible diseases, and suggest basic precautions. The goal is to improve healthcare accessibility, especially for users who need instant preliminary medical support.

> ⚠️ This system is intended for educational and preliminary guidance purposes only and does not replace professional medical consultation.

---

## Key Features

* 💬 Interactive chatbot interface (text + voice input)
* 🧠 Machine Learning–based disease prediction
* 🔍 Symptom extraction using NLP
* 📊 TF-IDF + SVM model for classification
* 🌐 Web-based interface using Flask
* 🎙 Voice input support using Speech Recognition
* 🔐 Focus on data privacy and security
* 🌍 Scalable design for future multilingual support

---

## Tech Stack

**Frontend**

* HTML5
* CSS3
* JavaScript

**Backend**

* Python
* Flask

**Machine Learning & NLP**

* Scikit-learn
* TF-IDF Vectorizer
* Support Vector Machine (SVM)
* Pandas, NumPy

**Other Tools**

* Pickle (model serialization)
* Logging module

---

## System Architecture

1. User enters symptoms via chat or voice input
2. NLP module extracts relevant medical entities
3. ML model predicts the most likely disease
4. System returns predicted disease and precautions
5. User receives guidance in real time

---

## Project Structure

```
Medical-Chatbot/
│── models/
│   └── disease_predictor.pkl
│── templates/
│   └── index.html
│── static/
│── disease_symptoms.csv
│── app.py
│── README.md
```

---

## Installation & Setup

### Prerequisites

* Python 3.8+
* pip

### Steps

```bash
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot
pip install -r requirements.txt
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## Sample Output

* Predicted disease based on symptoms
* Suggested precautions or next steps
* Friendly conversational responses

---

## Testing

The system was tested using:

* Basic medical queries
* Symptom-based predictions
* Edge cases and invalid inputs
* Performance and response time checks

---

## Limitations

* Not a replacement for professional doctors
* Accuracy depends on training data
* Cannot handle emergency medical cases
* Requires continuous updates with medical knowledge

---

## Future Enhancements

* Integration with Electronic Health Records (EHR)
* Deep learning models (BERT / GPT)
* Full multilingual support
* Mobile application version
* Advanced mental health assistance

---

## Contributors

* **G. Krishna**
* B. Nilesh
* D. Madhu

---

## Academic Context

This project was developed as an **Industrial Oriented Mini Project** for the B.Tech program in **Artificial Intelligence & Machine Learning**, affiliated with **JNTU Hyderabad**.

---

## License

This project is for academic and educational use only.

---

⭐ *If you are a recruiter, this project demonstrates practical skills in Python, Machine Learning, NLP, Flask, and full-stack integration.*
