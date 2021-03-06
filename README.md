# nlp
Boilerplate natural language processing


# Concepts
* *Word embedding skip gram*- It takes a word and  a window of k words arond it to build 2k contexts. Usually k=5 is selected. 
* *Word embedding negative sampling* - In order to improve the speed on execution instead of calculating similarity of word from all the contexts, randomly few context will be selected.
* *[Word Mover’s Distance (WMD)](http://proceedings.mlr.press/v37/kusnerb15.pdf)*- It utilizes word2vec embeddings property that distances between embedded word vectors are to some degree semantically meaningful. A text documents is represented as a weighted point cloud of embedded words. The distance between two text documents A and B is the minimum cumulative distance that words from document A need to travel to match exactly the point cloud of document B. The optimization problem underlying WMD reduces to a special case of the well-studied Earth Mover’s Distance.  **remove stop words** when using WMD, because it focuses on set of few important words in both the documents. When clustering, **WMD has higher accuracy on problems of short length and performs poorly on lengthy text** as per [1](https://www.kernix.com/blog/similarity-measure-of-textual-documents_p12) and [2](https://www.ibm.com/blogs/research/2018/11/word-movers-embedding/)
