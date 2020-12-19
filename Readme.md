This java implementation of Hidden Markov Models is based on Mark Stamp's paper *A Revealing Introduction to Hidden Markov Models* which can be found publicly at https://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf. 

The *test* class follows the example in the paper in which English text is fed into a model with two hidden states. Running the test class with name of a text file, the max number of iterations per run, and the number of random resets, will train the model and then print the Emission/Observation Matrix transposed. To verify that the model works correctly, the letters corresponding with vowels will clearly be associated in one column over the other. This shows that the model correctly linked vowels and consonants as the two *hidden states*. 