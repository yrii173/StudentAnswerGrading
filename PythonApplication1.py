import os
import pandas as pd
import networkx as nx
import seaborn as sns 
import numpy as np
import docx2txt
import pdfplumber
import re
import matplotlib.pyplot as plt
import string
import fitz  # з PyMuPDF
import textract
from sklearn.preprocessing import normalize
from pyvis.network import Network
from docx import Document
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


#==================== Налаштування ======================
answers_path = Path("E:/Python/Ozinka/AI_Grading_Example/")
sample_path = Path("E:/Python/Ozinka/Zrazok.docx")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#==================== Функції ======================

def extract_text(file_path):
    ext = file_path.suffix.lower()
    try:
        if ext == ".txt":
            return Path(file_path).read_text(encoding="utf-8")
        elif ext == ".docx":
            return docx2txt.process(file_path)
        elif ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        else:
            return textract.process(str(file_path)).decode("utf-8")
    except Exception as e:
        return ""

def jaccard_similarity(text1, text2):
    translator = str.maketrans('', '', string.punctuation)
    words1 = set(text1.lower().translate(translator).split())
    words2 = set(text2.lower().translate(translator).split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)



# Отримання усередненого ембедінга для чанків
def get_chunked_embedding(text, model, chunk_size=500, method="mean"):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len((current_chunk + sentence).split()) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    embeddings = model.encode(chunks, convert_to_tensor=True)
    
    if method == "mean":
        return torch.mean(embeddings, dim=0)
    elif method == "max":
        return torch.max(embeddings, dim=0).values
    else:
        return embeddings[0]

#==================== Основна логіка ======================

sample_text = extract_text(sample_path)

sample_embedding = get_chunked_embedding(sample_text, model)
#print("Shunk========", sample_embedding)
results = []
files = list(answers_path.glob("*.*"))
embeddings = {}
texts = {}

for file in files:
    student_name = file.stem
    answer_text = extract_text(file).strip()
    texts[student_name] = answer_text

    if not answer_text or len(answer_text.split()) < 3:
        results.append({
            "Name": student_name,
            "Similarity": 0.0,
            "Completeness": 0.0,
            "LexicalSimilarity": 0.0,
            "FinalScore": 0.0,
            "Grade_national": 2,
            "ECTS": "F",
            "PlagiarismWith": "-",
            "Note": "Порожня або дуже коротка відповідь"
        })
        continue

    answer_embedding = get_chunked_embedding(answer_text, model)
    embeddings[student_name] = answer_embedding

    similarity_score = util.pytorch_cos_sim(answer_embedding, sample_embedding).item()
    completeness_score = util.pytorch_cos_sim(sample_embedding, answer_embedding).item()
    lexical_score = jaccard_similarity(answer_text, sample_text)

    final_score = round((0.6 * similarity_score + 0.3 * completeness_score + 0.1 * lexical_score), 2)
    #round(similarity_score, 3),
    if final_score >= 0.90:
        grade_national, grade_ects = 5, "A"
    elif final_score >= 0.82:
        grade_national, grade_ects = 4, "B"
    elif final_score >= 0.75:
        grade_national, grade_ects = 4, "C"
    elif final_score >= 0.64:
        grade_national, grade_ects = 3, "D"
    elif final_score >= 0.60:
        grade_national, grade_ects = 3, "E"
    else:
        grade_national, grade_ects = 2, "F"

    results.append({
        "Name": student_name,
        "Similarity": similarity_score,
        "Completeness": completeness_score,
        "LexicalSimilarity": lexical_score,
        "FinalScore": final_score,
        "Grade_national": grade_national,
        "ECTS": grade_ects,
        "PlagiarismWith": "-",  # додамо далі
        "Note": ""
    })

#==================== Перевірка на плагіат ======================

plagiarism_threshold = 0.98
plagiarism_graph = nx.Graph()


for i, name_i in enumerate(embeddings):
     for j, name_j in enumerate(embeddings):
          if i >= j:
              continue
          score = util.pytorch_cos_sim(embeddings[name_i], embeddings[name_j]).item()
          if score >= plagiarism_threshold:
               plagiarism_graph.add_edge(name_i, name_j, weight=score)

# Заповнення колонки PlagiarismWith без дублікатів
for entry in results:
    name = entry["Name"]
    if name in plagiarism_graph:
        neighbors = list(plagiarism_graph.neighbors(name))
        # Видалення тих, для кого вже цей зв'язок буде відображено в іншому рядку
        filtered = [n for n in neighbors if name < n]  # ← Алфавітний фільтр
        entry["PlagiarismWith"] = ", ".join(sorted(filtered)) if filtered else "-"
    else:
        entry["PlagiarismWith"] = "-"


#==================== Збереження та граф ======================




df = pd.DataFrame(results)
df = df[["Name", "Similarity", "Completeness", "LexicalSimilarity", "FinalScore", "Grade_national", "ECTS", "PlagiarismWith", "Note"]]
df.sort_values(by="Name", ascending=True, inplace=True)
df.to_excel("grading_results22.xlsx", index=False)


# 📈 Розподіл оцінок
plt.figure(figsize=(6, 5)) 
ax = sns.countplot(x="Grade_national", data=df, palette="Set2")

# Додаємо кількість над кожним стовпцем
for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(str(count), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Assessment Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of students")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("assessment_distribution.png")
plt.show()

# 📊 Секторна діаграма підозр на плагіат
plagiarism_counts = df["PlagiarismWith"].apply(lambda x: 0 if x == "-" else len(x.split(",")))
plagiarism_summary = plagiarism_counts.value_counts().sort_index()

labels = [f"{k} similarity(s)" for k in plagiarism_summary.index]
sizes = plagiarism_summary.values

plt.figure(figsize=(5, 5))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("Distribution of responses with multiple similarities")
plt.tight_layout()
plt.savefig("plagiarism_piechart.png")
plt.show()




if plagiarism_graph.number_of_nodes() == 0:
    print("📡 Граф плагіату не побудовано: немає збігів.")
else:
    # Розфарбування кластерів
    components = list(nx.connected_components(plagiarism_graph))
    palette = ['#FF9999', '#99CCFF', '#99FF99', '#FFCC99', '#CC99FF', '#FFFF99']
    node_colors = {}
    for idx, comp in enumerate(components):
        for node in comp:
            node_colors[node] = palette[idx % len(palette)]
    node_colors_list = [node_colors[n] for n in plagiarism_graph.nodes()]

    # Колір і товщина ребер
    edge_colors = []
    edge_widths = []
    for u, v, data in plagiarism_graph.edges(data=True):
        w = data['weight']
        edge_widths.append(2 + 6*(w - plagiarism_threshold)/(1-plagiarism_threshold))
        edge_colors.append('darkred' if w > 0.999 else 'gray')

    # Підписи ребер
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in plagiarism_graph.edges(data=True)}

    

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(plagiarism_graph, k=2.7, scale=10, iterations=100)
    weights = nx.get_edge_attributes(plagiarism_graph, 'weight')
    nx.draw(plagiarism_graph, pos, with_labels=True, node_color=node_colors_list, edge_color='gray', node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(plagiarism_graph, pos, edge_labels={k: f"{v:.2f}" for k, v in weights.items()}, font_color='red', font_size=8)
    plt.title("Plagiarism graph between student responses", fontsize=14, fontweight='bold')
    plt.suptitle("Plagiarism graph between student responses", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

   
print("Файл grading_results22.xlsx збережено")

#==========================================================



