# Ethical & Privacy Considerations – AI Bodybuilding Coach

| Concern | Mitigation |
|---------|------------|
| **Health risk** – Incorrect form or diet advice can injure users. | • Medical disclaimer in sidebar.<br>• Age gate blocks under-13 users.<br>• Answers cite authoritative sources so users can verify. |
| **Data privacy** – Users may upload proprietary documents. | • Uploads stored only on the host file-system (`data/uploads/`) and ignored by Git (`.gitignore`).<br>• No documents are sent to third-party storage; only embeddings (irreversible numerical vectors) go to ChromaDB. |
| **Bias & fairness** – Source material may reflect gender or cultural bias. | • Corpus mixes WHO, ACSM, NSCA and peer-reviewed research.<br>• Citations shown; users can assess credibility. |
| **LLM hallucination** | • Retrieved context displayed in an expander.<br>• If no relevant context exists, the answer explicitly labels itself "General knowledge". |
| **Cost / rate limits** | • Session-level rate limiter (helpers/monitoring.py).<br>• Token usage logged for future cost analysis. |

This project follows OpenAI policy and GDPR principles (no personal data retained beyond session scope). 