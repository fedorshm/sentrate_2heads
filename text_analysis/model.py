import torch
from transformers import RobertaTokenizer, RobertaModel
from torch import nn

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super(MultiTaskModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier1 = nn.Linear(self.roberta.config.hidden_size, num_labels_task1)
        self.classifier2 = nn.Linear(self.roberta.config.hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        return logits1, logits2

def load_model():
    model_name = 'Gnider/distillroberta_2heads_sentimrate'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = MultiTaskModel(model_name, num_labels_task1=2, num_labels_task2=8)
    return model, tokenizer

def predict_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
    model.eval()
    with torch.no_grad():
        logits1, logits2 = model(**inputs)

    sentiment_probs = torch.softmax(logits1, dim=1).squeeze().tolist()
    sentiment_labels = ['negative', 'positive']
    sentiment_scores = {label: score for label, score in zip(sentiment_labels, sentiment_probs)}

    rating_probs = torch.softmax(logits2, dim=1).squeeze().tolist()
    top_3_ratings = sorted(range(len(rating_probs)), key=lambda i: rating_probs[i], reverse=True)[:3]
    top_3_scores = {str(rating): rating_probs[rating] for rating in top_3_ratings}

    return sentiment_scores, top_3_scores
  
