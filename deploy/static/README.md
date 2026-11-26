---
title: Arabic Sentiment Analysis
emoji: 💬
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
license: mit
short_description: Arabic tweets and reviews as positive, negative or neutral
---

# Arabic Sentiment Analysis

Live demo of a fine-tuned AraBERT model that labels Arabic text as positive, negative or neutral, and says when it is unsure.

The model runs in the visitor's browser with [transformers.js](https://github.com/huggingface/transformers.js): it downloads once (about 130 MB, 8-bit weights) and the text never leaves the device.

- Code, training and evaluation: https://github.com/Mohamedbr-Dev/arabic-sentiment-analysis
- On held-out data: 87% accuracy on tweets, 89% on hotel reviews, 83% on restaurant reviews it never trained on, 60% on company reviews.
- Mixed opinions ("the food is good but the service is slow") usually get a positive or negative label.

The MIT license covers the code. The model is fine-tuned from `aubmindlab/bert-base-arabertv02-twitter` on ArSAS tweets and on hotel and SHEIN reviews; those keep their own terms, and ArSAS states none.
