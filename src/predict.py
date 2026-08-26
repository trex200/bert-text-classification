"""Greedy CLS sentiment for one English sentence."""

import torch
import torch.nn.functional as F

from .data import PAD_token, create_tensors, normalizeString


def predict_sentence_sentiment(model, sentence, lang, device="cuda", max_len=64):
    model.eval()
    with torch.no_grad():
        normalized_sentence = normalizeString(sentence)
        input_tensor = create_tensors([normalized_sentence], lang, max_len).to(device)
        mask = (input_tensor != PAD_token).unsqueeze(1).unsqueeze(2).float().to(device)
        logits = model(input_tensor, mask)
        probs = F.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
        sentiment = "positive" if prediction == 1 else "negative"
        return sentiment, confidence
