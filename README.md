# emojis

In this project, I chose to analyse the use of emojis on Twitter.

I gathered more than 3 millions tweets over a week, excluding retweets, that I stored in a MongoDB database.

After cleaning them by removing punctuation, converingt uppercases to lowercases and lemmatizing, I used gensim implementation of Word2Vec to analyze the hidden semantic of emojis... Word2Vec is a very powerful tool that allow us to find the most similar words to a given word or sentence. This is how we can create a dictionnary of emojis by finding their "word synonyms!

I show in the [presentation](prez.pdf) that emojis can be considered as a new language and are a powerful tool to do sentiment analysis and topic modeling on Twitter.

The map of emojis is computed with t-SNE, a projection of the 100 dimensions of W2V on a two dimensions plane.
