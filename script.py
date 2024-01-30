from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig, AutoModelForSeq2SeqLM
import torch

# emotional package
tokenizer = AutoTokenizer.from_pretrained("Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
config = LukeConfig.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime',
                                    output_hidden_states=True)
model = AutoModelForSequenceClassification.from_pretrained(
    'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=config)
# summary package
model2 = AutoModelForSeq2SeqLM.from_pretrained('Gou1839/Live-Door-3Line-Summary')
tokenizer2 = AutoTokenizer.from_pretrained('sonoisa/t5-base-japanese')

lyric = input("Enter Text:").replace('\\n', '\n')  # Get Lyric
text = lyric

# emotional
max_seq_length = 512
token = tokenizer(text,
                  truncation=True,
                  max_length=max_seq_length,
                  padding="max_length")
output = model(torch.tensor(token['input_ids']).unsqueeze(0), torch.tensor(token['attention_mask']).unsqueeze(0))
max_index = torch.argmax(torch.tensor(output.logits))

print("\n" + "=== Emotion tags included in Text ===" + "\n")
if max_index == 0:
    print('joy、うれしい')
elif max_index == 1:
    print('sadn'
          'ess、悲しい')
elif max_index == 2:
    print('anticipation、期待')
elif max_index == 3:
    print('surprise、驚き')
elif max_index == 4:
    print('anger、怒り')
elif max_index == 5:
    print('fear、恐れ')
elif max_index == 6:
    print('disgust、嫌悪')
elif max_index == 7:
    print('trust、信頼')

# summary

print("\n" + "=== Text summary results === " + "\n")
inputs = tokenizer2(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model2.generate(inputs["input_ids"],
                          max_length=140,
                          min_length=17,
                          num_beams=4,
                          no_repeat_ngram_size=2,
                          do_sample=True,
                          top_k=50,
                          early_stopping=True)

print(tokenizer2.decode(outputs[0], skip_special_tokens=True))
