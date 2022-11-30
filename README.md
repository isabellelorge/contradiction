# CONTRADICTION 

This repo contains a model built to detect whether the authors of a social media post and its response agree or disagree (or neither). 

For each user in the dataset, it extracts the named entities mentioned in all their posts, then creates 'pro' and 'con' sentence embeddings using SBERT and takes the difference between 'pro' and 'con' cosine similarity as the stance of this user towards that particular entity. 

At training time, it uses a Weighted Signed Convolutional graph to produce embeddings for the users from their entities graph which are used as features along with BERT embeddings of the text for classifying pairs of posts.
