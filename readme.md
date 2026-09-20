Og Paper - https://arxiv.org/pdf/2205.14217
Og gaussian embedding paper - https://arxiv.org/pdf/1412.6623 (I haven't read yet, so fact check)

Basic idea instead of word2vec 

Use 

word or in this case letter 2 vec of gaussians learned mu and sigma 
So 
emb(letter) = [N(mu,sig)] x n

Then VAE style KL down to standard normal???
But then how do we presreve enough info ... 

Hmmmmmm

We could also just have learned matrices 
mu = [emb_length, vocab_size]
sigma = [emb_length, vocab_size]

then a sample is N(mu[token],sigma[token]), but that kind of defeats the purpose. We need to draw it towards something to have efficient sampling.

Unless our sample is some composed multimodal of all the gaussians together ... That could work.

Regardless, the idea is to then utilize that as that learned noise with in the model instead of the regular gaussian method.

Then constraint the composed gaussian 

Steps to implement
Reproduce Diffusion LM 1-1 on shakespeare
Do our idea 
Create a grid to test it on hyper parameter wise
Off shore to compute and run
Evaluate

Another idea I want to implement
