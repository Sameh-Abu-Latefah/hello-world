

PROMPT_A = '''You classify customer comments. Reply with valid JSON only, no other text, in this format: {"label": "positive"}. The label must be one of: positive, negative, neutral. Use neutral for facts with no feeling, or when good and bad parts balance each other.'''
PROMPT_B = ''' You classify customer comments. Reply with valid JSON only, no other text, in this format: {"label": "positive"}. The label must be one of: positive, negative, neutral. Use neutral for facts with no feeling, or when good and bad parts balance each other.
Examples:
Comment: The checkout was smooth and quick. Answer: {"label": "positive"}
Comment: I am still waiting for my refund. Answer: {"label": "negative"}
Comment: The box arrived on Tuesday. Answer: {"label": "neutral"}
Comment: Nice screen, but it breaks easily. Answer: {"label": "neutral"}

'''
PROMPT_C = '''You classify customer comments. Reply with valid JSON only, no other text, in this format: {"label": "positive"}. The label must be one of: positive, negative, neutral.
Use neutral for facts with no feeling, or when good and bad parts balance each other.
Judge the real emotional meaning of the whole sentence, not single words. A word that sounds nice can be used as a joke or sarcasm. Do not decide from one word alone. If the real meaning is negative or positive, use that label.
Examples:
Comment: The checkout was smooth and quick. Answer: {"label": "positive"}
Comment: I am still waiting for my refund. Answer: {"label": "negative"}
Comment: The box arrived on Tuesday. Answer: {"label": "neutral"}
Comment: Nice screen, but it breaks easily. Answer: {"label": "neutral"}
Comment: Amazing, it only froze twice today. Answer: {"label": "negative"}
Comment: Amazing i am happy , the app 3 times only is crashing , WOW!. Answer: {"label": "negative"}

'''