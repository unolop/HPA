import pandas as pd 
import seaborn as sns
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 
from adjustText import adjust_text
import numpy as np 

def wrap_every_n_words(text, n=3):
    words = text.split()
    lines = [" ".join(words[i:i+n]) for i in range(0, len(words), n)]
    return "\n".join(lines) 
    
def get_labels(qtype, df=sdf, qs=qs): 
    q_to_w = (
        df[["question", qtype]]
        .drop_duplicates()
        .set_index("question")[qtype] 
        .to_dict()
    )

    labels = [q_to_w[q] for q in qs] 
    print(len(qs), len(labels) )
    return labels  

def show_qtype_proportion(dataset): 
    df = pd.DataFrame(list(dataset), columns=['answer_type', 'question_type', 'question_id']) 
    question_counts = df['question_type'].value_counts() 
    plt.figure(figsize=(10,15))
    question_counts.plot(kind='barh') # question_counts / question_counts.sum()  
    plt.xlabel('Counts')
    plt.title(f'Question Types in VQA Validation Set (# {len(dataset)})') 
    plt.tight_layout()
    plt.show()
    return df 

def visualize_by_semantics(X, qs, qtype, labels, unique_labels): 
    # colors = sns.color_palette("Set2", len(unique_labels)) 
    colors = sns.color_palette("husl", len(unique_labels))   
    
    Z = TSNE(
        n_components=2,
        perplexity=min(10, max(2, len(qs)//6)),
        init="pca",
        learning_rate="auto",
        random_state=42
    ).fit_transform(X)

    plt.clf() 
    plt.figure(figsize=(10, 7))
    for lab, color in zip(unique_labels, colors): 
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            label=lab,
            color=color, 
            s=20
        )
 
    texts = []
    for lab in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lab]
        center = Z[idx].mean(axis=0)
        d = np.linalg.norm(Z[idx] - center, axis=1)
        best_i = idx[np.argmin(d)]

        # create text with initial offset
        t = plt.text(
            Z[best_i, 0] + 0.5,  # initial x offset
            Z[best_i, 1] + 0.5,  # initial y offset
            f"{lab}: {wrap_every_n_words(qs[best_i], n=5)}",
            fontsize=6,
            alpha=0.9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        texts.append(t)

    # automatically reduce overlaps + draw arrows to points
    adjust_text(
        texts,
        x=Z[:, 0], y=Z[:, 1],
        arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.5),
        expand_text=(1.05, 1.2),
        expand_points=(1.2, 1.4),
        force_text=(0.2, 0.3),
        force_points=(0.1, 0.2)
    )

    plt.legend(title=qtype, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"Question-type stems (SentenceTransformer + t-SNE) #{len(labels)}") 
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show() 