import gradio as gr
from transformers import pipeline

# ---------------------------
# إعداد الموديلات الأساسية
# ---------------------------
text_gen = pipeline("text-generation", model="gpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment = pipeline("sentiment-analysis")
translator = pipeline("translation_en_to_ar", model="Helsinki-NLP/opus-mt-en-ar")
qa_model = pipeline("question-answering")

# ---------------------------
# دوال الأدوات
# ---------------------------
def generate_text(prompt):
    return text_gen(prompt, max_length=100)[0]['generated_text']

def summarize_text(text):
    return summarizer(text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']

def analyze_sentiment(text):
    return sentiment(text)[0]['label']

def translate_to_arabic(text):
    return translator(text)[0]['translation_text']

def answer_question(context, question):
    return qa_model(question=question, context=context)['answer']

# ---------------------------
# تصميم الواجهة Dashboard
# ---------------------------
with gr.Blocks(title="AI Study Dashboard") as demo:
    gr.Markdown("""
    # 🎓 My Hugging Face Study Dashboard  
    > تعلم أدوات الـNLP والذكاء الاصطناعي من خلال التجربة المباشرة 🔥  
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📘 قائمة الأدوات")
            gr.Markdown("""
            - 🗣️ توليد النصوص  
            - 📰 تلخيص النصوص  
            - 😊 تحليل المشاعر  
            - 🌍 الترجمة  
            - ❓ الإجابة على الأسئلة
            """)
            gr.Markdown("> اختر الأداة من التبويبات باليمين وجرب بنفسك 👇")

        with gr.Column(scale=3):
            with gr.Tab("🗣️ Text Generation"):
                inp = gr.Textbox(label="اكتب بداية الجملة", placeholder="Once upon a time...")
                out = gr.Textbox(label="الناتج")
                btn = gr.Button("توليد النص")
                btn.click(generate_text, inp, out)

            with gr.Tab("📰 Summarization"):
                inp2 = gr.Textbox(label="نص للتلخيص", lines=6)
                out2 = gr.Textbox(label="الملخّص")
                btn2 = gr.Button("لخّص")
                btn2.click(summarize_text, inp2, out2)

            with gr.Tab("😊 Sentiment Analysis"):
                inp3 = gr.Textbox(label="اكتب نص للتقييم")
                out3 = gr.Textbox(label="نتيجة الشعور")
                btn3 = gr.Button("تحليل")
                btn3.click(analyze_sentiment, inp3, out3)

            with gr.Tab("🌍 Translation (EN → AR)"):
                inp4 = gr.Textbox(label="اكتب نص بالإنجليزية")
                out4 = gr.Textbox(label="الترجمة بالعربية")
                btn4 = gr.Button("ترجمة")
                btn4.click(translate_to_arabic, inp4, out4)

            with gr.Tab("❓ Question Answering"):
                context = gr.Textbox(label="النص (المصدر)", lines=5)
                question = gr.Textbox(label="السؤال")
                answer = gr.Textbox(label="الإجابة")
                btn5 = gr.Button("أجب")
                btn5.click(answer_question, [context, question], answer)

demo.launch()
