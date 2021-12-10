1. Training data:
- Articles with summary: train_with_summ.txt
- Articles without summary: train_without_summ.txt
2. Please run wash_data.py as a data parser.
3. Data format is [‘summarization’: summarization, ‘article’: article], and every line is a data sample. The strings are encoded in UTF-8.
4. Note: There is a special tag <Paragraph> to denote separation of paragraphs. 