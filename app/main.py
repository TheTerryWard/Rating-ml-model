import streamlit as st

import torch
import torch.nn.functional as F
from torch import nn
import time


import sentencepiece as spm

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        
        # padding_idx=pad_idx говорит модели игнорировать нули (ускорение!)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # Параллельные свертки (для 2, 3 и 4 слов)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                      out_channels=n_filters, 
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(n_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch, len] -> embedded: [batch, len, dim] -> [batch, dim, len]
        embedded = self.embedding(text).permute(0, 2, 1)
        
        # Свертка + ReLU + MaxPool для каждого размера фильтра
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Объединяем результаты
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)
    


@st.cache_resource
def load_model():
    sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
    VOCAB_SIZE = sp.GetPieceSize()
    model = TextCNN(
    vocab_size=VOCAB_SIZE,
    embed_dim=300,
    n_filters=100,
    filter_sizes= [3,4,5],
    output_dim=3,
    dropout= 0.5,
    pad_idx=0)
    model.load_state_dict(torch.load('textcnn.pth'))
    model.eval()
    
    
    return model, sp

model, sp = load_model()

st.title("TextCNN для класификации коротких отзывов")


text = st.text_area("Введите текст для классификации:")
if st.button("Предсказать") and text:
    # Токенизация
    start_time = time.time()
    tokens = sp.encode(text, out_type=int)
    if len(tokens) == 0:
        st.error("Текст пустой")
    else:
        while len(tokens) < 5:
            tokens.append(0)  # pad_idx=0
        input_tensor = torch.tensor([tokens[:256]]).long()  # [1, seq_len]
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            confidence = probs[0][pred].item()
        
        processing_time = time.time() - start_time
        preds_dict = {0:"Отрицательно", 1:"Нейтрально", 2:"Положительно"}
        the_pred = preds_dict[pred]
        st.success(f"Класс: {the_pred}, уверенность: {confidence:.2%}. Обработано за: {processing_time*1000:.1f} мс")
