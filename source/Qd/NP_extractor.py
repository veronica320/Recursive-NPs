'''
NP extractor: extract noun phrases from a sentence.
This is from an external open-source repository, but we can no longer find the link. Please contact us if you are the author and we'll add proper acknowledgement, thanks!
'''

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

lemmatizer = nltk.WordNetLemmatizer()

def leaves(tree):
	"""Finds NP (nounphrase) leaf nodes of a chunk tree."""
	for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
		yield subtree.leaves()

def get_word_postag(word):
	"""Get the part of speech of a word."""
	if pos_tag([word])[0][1].startswith('J'):
		return wordnet.ADJ
	if pos_tag([word])[0][1].startswith('V'):
		return wordnet.VERB
	if pos_tag([word])[0][1].startswith('N'):
		return wordnet.NOUN
	else:
		return wordnet.NOUN

def normalise(word):
	"""Normalises words to lowercase and stems and lemmatizes it."""
	word = word.lower()
	postag = get_word_postag(word)
	word = lemmatizer.lemmatize(word, postag)
	return word

def get_terms(tree):
	for leaf in leaves(tree):
		terms = [normalise(w) for w, t in leaf]
		yield terms

def extract_NPs(sentence):
	tokens = nltk.word_tokenize(sentence)
	postag = nltk.pos_tag(tokens)

	# Rule for NP chunk and VB Chunk
	grammar = r"""
		        NP: {<DT>?<JJ.*>*<NN.*>*}
		        """
	cp = nltk.RegexpParser(grammar)
	tree = cp.parse(postag)

	terms = get_terms(tree)

	features = []
	for term in terms:
		_term = ''
		for word in term:
			_term += ' ' + word
		features.append(_term.strip())
	return features