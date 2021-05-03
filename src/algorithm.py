import nltk
from nltk import Tree 
gram_string = ' S -> NNP \n' #Every sent starts with this
grammar1 = nltk.CFG.fromstring(""" %s """ % (gram_string))
print_tree = False

""" Source -> http://www.nltk.org/book/ch05.html"""

def update_gram(sentences):
    pos = nltk.pos_tag(nltk.word_tokenize(sentences[0]))
    grammars = [ tuples[1] for tuples in pos]

    for i in range(len(grammars)-1):
        gram_string += ' '  + grammars[i] + " -> " + grammars[i+1] + '''| "''' + grammars[i+1] + '''"'''+ "\n"
    return gram_string

def get_grammar():
    return grammar1

def tree_from_str(str):
    return nltk.Tree.fromstring(str)

def str_from_tree_pretty(tree):
    return nltk.Tree.fromstring(str(tree)).pretty_print()

def print_tree(sent):

    sentence = nltk.pos_tag(nltk.word_tokenize(sent))
    regex_pattern = """NP: {<DT>?<JJ>*<NN>}
    VBD: {<VBD>}
    IN: {<IN>}"""
    chunker = nltk.RegexpParser(regex_pattern)
    print (chunker.parse(sentence)) 

def parser():
    rd_parser = nltk.RecursiveDescentParser(grammar1)
    while True:
        inp = input("Enter a sentence: " )
        sent = nltk.word_tokenize(inp)
        #print(sent)
        try:
            for y in rd_parser.parse(sent):
                if print_tree:
                    print (y)
            print("Valid sentence!")
        except ValueError as e:
            print(e)
        