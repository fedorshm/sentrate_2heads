from django.shortcuts import render
from .model import load_model, predict_text

model, tokenizer = load_model()

def index(request):
    sentiment_scores = None
    top_3_scores = None
    if request.method == "POST":
        text = request.POST.get('text')
        sentiment_scores, top_3_scores = predict_text(text, model, tokenizer)

    return render(request, 'text_analysis/index.html', {
        'sentiment_scores': sentiment_scores,
        'top_3_scores': top_3_scores
    })
  
# Create your views here.
