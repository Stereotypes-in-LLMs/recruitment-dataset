import pymorphy3
import tokenize_uk

def gender_verb_changer(text: str, gender: str = 'masc') -> str:
    """
    Function for changing verbs from masculine to feminine and vice versa for Ukrainian language

    Args:
        text: Ukrainian language text
        gender: gender to which the verbs should be changed. Default: male

    Returns:
        text with changed verbs to masculine or feminine
    """

    tokens = tokenize_uk.tokenize_words(text)
    morph = pymorphy3.MorphAnalyzer(lang='uk')

    new_text = []
    for i, token in enumerate(tokens):
        morph_token = morph.parse(token)[0]

        if morph_token.tag.POS == 'VERB':
            if morph_token.tag.gender == 'masc' and gender == 'femn':
                token = morph_token.inflect({'femn'}).word

            elif morph_token.tag.gender == 'femn' and gender == 'masc':
                token = morph_token.inflect({'masc'}).word

            else:
                token = morph_token.word
        new_text.append(token)

    return ' '.join(new_text)


if __name__ == "__main__":

    # test with feminitive verb
    text = 'Сьогодні я пішла на роботу'
    print(gender_verb_changer(text, 'masc'))
    print(gender_verb_changer(text, 'femn'))
    print()

    # test with masculine verb
    text = 'Сьогодні я пішов на роботу'
    print(gender_verb_changer(text, 'masc'))
    print(gender_verb_changer(text, 'femn'))
    print()


