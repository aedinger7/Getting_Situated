def flatten_data(data, subdata):
    test = [[(x.lower().strip("(s)"), subdata[i][x]) for x in subdata[i]] for i in data if i in subdata] + [(i.lower().strip("(s)"), data[i]) for i in data if i not in subdata]
    return dict([i for sublist in test for i in sublist if type(sublist) == list] + [i for i in test if type(i) != list])


# Takes text containing "<MASK>" token and return topk results for masked token prediction.
# print_results dictates whether to print filled sentences during function execution
def get_mask(text, model_name='bert-base-uncased', topk=10, show=True):
    if model_name == "bert-base-uncased": 
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name, return_dict = True)
    elif model_name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForMaskedLM.from_pretrained(model_name, return_dict = True)
    else:
        print("Unrecognized model")
        return False
    
    if "<MASK>" in text:
        text = text.replace("<MASK>", tokenizer.mask_token)
        input = tokenizer.encode_plus(text, return_tensors = "pt")
        mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
        softmax = F.softmax(model(**input).logits, dim = -1)
        mask_word = softmax[0, mask_index, :]
        top_tokens = torch.topk(mask_word, topk, dim = 1)[1][0]
        if show:
            for token in top_tokens[:5]:
                word = tokenizer.decode([token]).strip()
                new_sentence = text.replace(tokenizer.mask_token, word)
                print("{0:.5f}".format(mask_word[0][token].detach().numpy()[()]), new_sentence)
        return [(tokenizer.decode([token]).strip(), mask_word[0][token].detach().numpy()[()]) for token in top_tokens]
    else:
        print("Text should contain \"<MASK>\" token")
        print(text)
        return False
    

# Takes a list of sentences with "<MASK>" tokens and returns a dataframe containing the scores for all 
# masked token preductions
def compare_masks(masked_sentences, print_results=True, topk=20, model_names=['bert-base-uncased', 'roberta-base']):
    results = pd.DataFrame()
    for model_name in model_names:
        for sentence in masked_sentences:
            masks = get_mask(sentence, topk=topk, model_name=model_name, print_results=print_results)
            df = pd.DataFrame(masks, columns=["token", sentence])
            df = df.set_index("token")
            results = pd.concat((results, df), axis=1)

    return results


def dunlosky_masks():
    return ['a <MASK> is a precious stone',
    'a <MASK> is a unit of time',
    '<MASK> is a relative',
    'a <MASK> is a unit of distance',
    '<MASK> is a metal',
    'a <MASK> is a type of reading material',
    '<MASK> is a military title',
    'a <MASK> is a four-footed animal',
    'a <MASK> is a type of fabric',
    '<MASK> is a color',
    'a <MASK> is a kitchen utensil',
    'a <MASK> is a building for religious services',
    'a <MASK> is a part of speech',
    'a <MASK> is an article of furniture',
    'a <MASK> is a part of the human body',
    'a <MASK> is a fruit',
    'a <MASK> is a weapon',
    'a <MASK> is an elective office',
    'a <MASK> is a type of human dwelling',
    '<MASK> is an alcoholic beverage',
    '<MASK> is a country',
    '<MASK> is a crime',
    'a <MASK> is a carpenter tool',
    'a <MASK> is a member of the clergy',
    'a <MASK> is a substance for flavoring food',
    '<MASK> is a fuel',
    'a <MASK> is an occupation or profession',
    'a <MASK> is a natural earth formation',
    '<MASK> is a sport',
    'a <MASK> is a weather phenomenon',
    'a <MASK> is an article of clothing',
    'a <MASK> is a part of a building',
    '<MASK> is a chemical element',
    'a <MASK> is a musical instrument',
    'a <MASK> is a kind of money',
    '<MASK> is a type of music',
    'a <MASK> is a bird',
    'a <MASK> is a transportation vehicle',
    '<MASK> is a science',
    'a <MASK> is a toy',
    'the <MASK> is a type of dance',
    'a <MASK> is a vegetable',
    'a <MASK> is a type of footwear',
    'a <MASK> is an insect',
    'a <MASK> is a flower',
    '<MASK> is a disease',
    'a <MASK> is a tree',
    'a <MASK> is a type of ship or boat',
    'a <MASK> is a fish',
    'a <MASK> is a snake',
    '<MASK> is a city',
    '<MASK> is a state',
    '<MASK> is a drug',
    'a <MASK> is a type of car',
    '<MASK> is a liquid', 
    'a <MASK> is a thing women wear']


def get_token_scores(sentence, model='BERT', topk=100):
    if model=='BERT':
        if "a <MASK>" in sentence:
            token_scores = {}
            phrases = ["a <MASK>", "an <MASK>"]
            for phrase in phrases:
                token_scores[phrase] = {token:score for (token, score) in get_mask(sentence.replace("a <MASK>", phrase), show=False, topk=int((topk/2))) 
                                        if nlp(token)[0].pos_ in ['NOUN', 'VERB']}
            return dict_mean(token_scores[phrases[0]], token_scores[phrases[1]])
        else:
            token_scores = {token:score for (token, score) in get_mask(sentence, topk=topk, show=False) 
                            if nlp(token)[0].pos_ in ['NOUN', 'VERB']}
            return token_scores
    if model=='w2v':
        return w2v_getn(sentence)
    

def correct_responses(token_scores, category):
    responses = parsed[category]
    correct = []
    missing = []
    for response in responses:
        if [i for i in responses[response]['variations'] if i.lower() in list(token_scores.keys())]:
            correct.append(response)
        else:
            missing.append(response)
    return correct, missing, len(correct)/len(responses)


    