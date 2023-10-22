from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt

class Translator:

    def __init__(self, src_lang, tgt_lang):
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

    def translate(self, text):
        encoded = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded)
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
