def get_event_tokens(tokenizer):
    tokens = ['Justice:Sentence', 'Business:End-Org', 'Personnel:Elect', 'Justice:Charge-Indict', 'Business:Declare-Bankruptcy', \
            'Justice:Extradite', 'Justice:Fine', 'Business:Start-Org', 'Justice:Pardon', 'Conflict:Demonstrate', 'Justice:Trial-Hearing',\
            'Justice:Arrest-Jail', 'Personnel:Start-Position', 'Transaction:Transfer-Money', 'Life:Marry', 'Life:Divorce', \
            'Movement:Transport', 'Justice:Convict', 'Justice:Acquit', 'Transaction:Transfer-Ownership', 'Life:Injure', 'Contact:Meet', \
            'Personnel:Nominate', 'Justice:Release-Parole', 'Contact:Phone-Write', 'Personnel:End-Position', 'Justice:Execute', 'Life:Die', \
            'Conflict:Attack', 'Justice:Sue', 'Justice:Appeal', 'Business:Merge-Org', 'Life:Be-Born', '|', '[', ']']
    tokens += tokenizer.all_special_tokens
    return tokenizer.encode(tokens, is_split_into_words=True, return_tensors="pt")[0]