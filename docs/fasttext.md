# Fasttext
### Modo Supervisado:

Entrenar: 
$ fasttext supervised -input cooking.train -output model

Correr para un input cualquiera:
$ fasttext predict model.bin test.txt 1
(Aquí, 1 es la cantidad de labels que queremos que devuelva.)

Correr contra validation set (calcula P@k y R@k):
$ fasttext test model.bin cooking.valid 5

(Aquí, k = 5)

Formato:
Cada documento tiene que ser una sola linea. Los documentos se separan con \n. Cada documento debe iniciar con "__label__nombre [espacio en blanco] ".


### Modo no Supervisado:

Entrenar: 
$ fasttext skipgram -input subreddits.txt -output subreddits -epoch 1 -dim 300 -thread 8


Se puede modificar la cantidad de threads que usa fasttext con -threads

Formato:
Cada documento tiene que ser una sola linea. Los documentos se separan con \n.


### Making the model better

Looking at the data, we observe that some words contain uppercase letter or punctuation. One of the first step to improve the performance of our model is to apply some simple pre-processing. A crude normalization can be obtained using command line tools such as sed and tr:

cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt

This is a shell command that pipes the output of the sed command into the tr command.

The sed command replaces every occurrence of a punctuation character with a space followed by the same punctuation character. It does this by using a regular expression search for a character class containing any of the following punctuation characters: ., !, ?, ,, ', /, (, and ). Each of these characters is enclosed in parentheses to capture it as a group, and the entire group is then prefixed with a backslash to escape it. The replacement text is a space followed by the group match, which is represented by \1. The g at the end of the sed command means that the substitution should be applied globally to all occurrences in the input string.

The tr command then takes the output of sed and translates all uppercase characters to lowercase characters. This is done by using the [:upper:] and [:lower:] character classes, which are predefined character classes in the shell. [:upper:] matches all uppercase characters and [:lower:] matches all lowercase characters. Any uppercase characters in the input string are replaced with their corresponding lowercase characters by tr.

