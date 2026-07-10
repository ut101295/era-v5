---
title: "Axiom — Learning OS"
source: "https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cmq9ghp7y03yc08o8ajxhdsu2/lesson?split=1"
author:
published:
created: 2026-07-07
description: "A premium learning operating system for ambitious technical education."
tags:
  - "clippings"
---

## Session 1: From Neural Networks to the Transformer

## 1\. What this session is

In this session you will travel the full path from the very first idea of a neural network to a working mental picture of how a transformer takes that idea and stretches it into something that can write code and answer questions in many languages. You will not need any prior background in machine learning to follow along. Every step of the journey is built directly on the step that comes before it, so by the time the word "attention" appears on the page, you will already understand why we needed something like it in the first place.

The session sits at the foundation of the entire course. Every later class, whether it is about tokenization, optimizers, distributed training, or alignment, assumes that the picture you build today is solid. There is no other moment in the calendar where we will go through the basics this slowly and this carefully. We will be moving at a much faster pace from Session 2 onwards, so it is worth taking the time today to make sure the conceptual ground is firmly in place.

## 2\. The job a neural network is trying to do

A neural network is a system that learns to find patterns by looking at examples. You collect many examples of the problem you want the system to solve, along with the correct answer for each example, and you show those pairs to the network many times over. After enough examples, the network adjusts itself into a state where it can give correct answers on examples it has never seen before. A simple way to picture this is to imagine a system that learns to predict the price of a house from its size, or one that learns to decide whether an email is spam from the text it contains.

The mental picture to plant for the rest of the session is straightforward. A neural network takes numbers as input, performs some math in the middle, and produces numbers as output. When the number it produces is wrong, the network adjusts itself by a small amount and tries again on the next example. Everything that follows in this session, including the parts that look complicated, is a refinement of this same idea applied at a much larger scale.

## 3\. The single neuron

The smallest piece of a neural network is called a neuron. A neuron takes a few numbers as its input, multiplies each input by its own weight, adds up the results, adds one more number called the bias to that running total, and then passes the final value through a small function that gently bends the output. The bending function at the end is called an activation. The whole operation inside a neuron fits into one short description, which is that a neuron performs a weighted sum on its inputs and then runs the result through an activation.

If you remember the equation of a straight line from school, written as `y = w*x + b`, you already understand most of what a neuron is doing. The weight `w` controls how strongly the input affects the output. The bias `b` shifts the entire result up or down. The activation function placed at the end is what gives the neuron the ability to learn patterns that are more complicated than straight lines, which covers almost every interesting pattern in the real world.

> A live diagram of one neuron with sliders on the weights and bias. As you move the sliders, the output value updates immediately, and a small graph next to the diagram shows how the activation curve changes shape under different settings. Goal: you should leave with a felt sense, through the sliders, of what the weights and the bias and the activation are actually doing to the output value, before you ever encounter the words "training" or "loss."

## 4\. From one neuron to a network

A single neuron on its own is limited in what it can learn. The real power of neural networks comes from placing many neurons next to each other to form a layer, and then stacking many layers on top of one another. The output of one layer becomes the input of the next layer, which means each layer gets to work on top of the patterns that the layer below it has already discovered. The deeper the stack of layers, the richer the patterns that the network can learn to recognise.

The intuition that helps most students at this point is to picture the earliest layers as picking up very simple pieces of information, such as basic edges in an image or basic letter patterns in text. The middle layers combine those simple pieces into slightly larger ideas, such as shapes or word fragments. The later layers combine those again into complete concepts, such as a face or a meaningful sentence. The word "deep" in the phrase "deep learning" refers exactly to this stacking of many layers on top of one another.

> A stack of layers shown side by side, with arrows flowing from one layer into the next. You can switch between a shallow network with two layers and a deeper network with many layers, and a small example pattern is shown being learned by each version. Goal: you should leave with the picture that depth gives a network the room to gradually compose simple patterns into harder and more meaningful ones.

## 5\. How the network learns

The learning happens through three ideas that work together in a single cycle. The first idea is that the network needs a way to measure how wrong its answer is, and this measurement is computed by a function called the loss. The second idea is that once we know how wrong the answer is, we can adjust every weight inside the network by a small amount in the direction that would have made the answer less wrong, and this adjustment process is called gradient descent. The third idea is that we repeat this measurement and adjustment over a very large number of examples, sometimes running into the billions, and over time the weights settle into values that produce correct answers across the entire range of examples the network has seen.

The single phrase that holds all of this together is that learning is the repeated adjustment of weights to make errors smaller. There is nothing more mysterious going on inside a neural network than this single sentence, and the same sentence remains true at the scale of the very largest models that exist in the world today.

> A training loop visualised one step at a time. A small network attempts to predict a value, the loss is displayed as a coloured bar, and as you click "step" the loss decreases while the weights inside the network change colour to show which of them moved and in which direction. Goal: you should leave with a working sense of the cycle that runs through every training step, which is a forward pass that produces a prediction, a comparison against the correct answer, and a backward pass that updates every weight in the network.

The specific job that a language model is trained on is to predict the next word, or more precisely the next token, given everything it has already read up to that point. The same training cycle that we have just described is run on this prediction task, over enormous amounts of text, until the model becomes good at it.

> A sentence is revealed one word at a time, with the model attempting to guess what the next word will be at each step. The widget shows the model's top guesses, reveals the actual next word, scores how confident the model was in its prediction, and updates a small loss bar accordingly. Goal: you should leave knowing clearly that the entire training task of a language model is the prediction of the next token, repeated across many billions of examples drawn from real text.

## 6\. Why a plain network struggles with language

The kind of network described in the previous sections works very well for problems where the input is a fixed list of numbers, such as the size of a house combined with the age of the property. Language is harder than this kind of problem for two reasons that both need to be addressed before any real progress can be made.

The first reason is that language is sequential in nature. A sentence is a list of words placed in a particular order, and the meaning of the sentence depends heavily on that order. A network that treats its input as a fixed unordered list has no way to express the fact that the order carries information. The second reason is that the meaning of any individual word depends on the context around it. The word "bank" carries a different meaning in the phrase "river bank" than it does in the phrase "bank account," and the only way to know which meaning is intended is to read the surrounding words. A plain network has no built-in mechanism for looking at the surrounding words when it processes any single word.

> Two short sentences are displayed side by side, both containing the word "bank" used in different senses. The widget shows how a plain feedforward network produces the same internal representation of the word in both sentences, while a human reader immediately distinguishes between the two meanings. Goal: you should leave convinced that an additional piece of machinery is required in the architecture before language can be handled properly.

## 7\. Words as numbers

Neural networks only operate on numbers, so the first step in handling language is to convert each word into a list of numbers. That list of numbers is called an embedding. The idea behind embeddings is that words with similar meanings should end up with similar lists of numbers, so that the network can use closeness in the embedding space as a proxy for closeness in meaning. The embedding for the word "king" sits close to the embedding for the word "queen" in this space. The embedding for the word "king" sits far away from the embedding for the word "banana."

The embedding is the bridge between human language and the math that runs inside the network. Once every word in the input has been turned into a vector of numbers, the rest of the network can perform its usual operations of weighted sums and activations on those numbers without needing to know that the original input was a word. The embedding layer is one of the largest pieces of any modern language model, and the design choices around it have a real effect on the quality and cost of the model.

> A scatter plot in two dimensions shows a small vocabulary of words placed in the embedding space. As you hover over a word, the words closest to it light up, and you can see the famous arithmetic example where the vector for "king" minus the vector for "man" plus the vector for "woman" lands near the vector for "queen." Goal: you should leave with a clear picture of how embeddings turn words into numbers in a way that preserves something meaningful about the relationship between the words.

## 8\. The older approach and where it ran out of room

Before transformers became the standard architecture for language, the most common way of handling language with neural networks was a family of designs called recurrent neural networks, often shortened to RNNs. An RNN reads a sentence one word at a time and tries to carry a running summary of what it has read so far, updating that summary at each new word. The approach works reasonably well for short sentences, but two problems become serious as the sentence grows in length.

The first problem is that the running summary tends to forget information from the beginning of the sentence by the time it reaches the end of the sentence, which makes it hard for the network to use any long-range context in its predictions. The second problem is that the network has to read the words one after another in strict sequence, which means the work of processing the sentence cannot be spread across many processors at the same time, and training becomes much slower than it would otherwise be.

These two limitations together set the stage for the architectural shift that follows in the next section. The field needed a way to look at all of the words in a sentence at the same time and to decide, for each word, which of the other words mattered most for its meaning.

> A recurrent network is shown reading a long sentence one word at a time, with a small memory bar that shrinks as the sentence gets longer to represent the gradual forgetting of earlier words. A parallel panel shows the same sentence being read all at once by a hypothetical architecture that has access to every word in a single pass. Goal: you should leave understanding why the combination of long-range memory and parallel processing became the design pressure that produced the transformer.

## 9\. Attention

Attention is the central idea inside the transformer, and it addresses both of the limitations of the older approach in a single mechanism. The idea behind attention is that for every word in a sentence, the network gets to look at every other word in the same sentence and decide how much each of those other words matters for understanding the current word. In the sentence "The animal did not cross the street because it was tired," attention is the mechanism that allows the model to figure out that the word "it" is referring back to the animal rather than to the street.

The way attention is usually explained is through three roles that every word plays at the same time. Each word produces a small vector called a query, which represents the kind of information the word is looking for. Each word also produces a small vector called a key, which represents the kind of information the word can offer to other words. The match between a query coming from one word and a key coming from another word determines how much attention the first word pays to the second. Finally, each word produces a small vector called a value, which carries the actual information that gets pulled in when attention has decided that this word matters.

The attention mechanism runs at the same time for every word in the sentence, which solves the speed problem of the older approach, and it can connect any word directly to any other word in a single step, which solves the long-range memory problem of the older approach.

> A sentence is displayed with attention links drawn between the words. As you click on any word in the sentence, the widget highlights which of the other words it is paying attention to, and the thickness of each link reflects how strongly the attention is flowing in that direction. You can switch between several sentences to see how the pattern of attention changes from one sentence to the next. Goal: you should leave with a working mental picture of attention as a process where every word forms a query and a key and a value at the same time, and the queries and keys decide whose values flow where.

## 10\. The transformer block

A transformer is built by taking the attention mechanism and wrapping it inside a small repeating unit called a transformer block. A single block contains an attention layer that performs the query, key, and value mixing described in the previous section, followed by a small feedforward network that processes each position independently of the others, together with a few pieces of supporting machinery that keep training stable. Those supporting pieces include residual connections, which are shortcuts that allow information to skip past a layer, and normalisation layers, which keep the values flowing through the network from growing too large or shrinking too small.

Because attention by itself does not know which word came first and which word came later in the sentence, the transformer adds positional information into each word's vector before the attention layer ever sees it. This addition is called positional encoding, and the specific design of positional encoding has been refined many times in the years since the transformer was first introduced. We will return to the modern variants in Class 8.

The full transformer is constructed by stacking many of these blocks one on top of the other. The output of one block becomes the input of the next block, and after passing through many blocks the model has had the opportunity to mix information across the entire sentence in progressively richer ways. The number of blocks in a modern language model is typically somewhere between a few dozen and a few hundred, depending on the size of the model.

> A single transformer block is displayed with each of its parts labelled, and you can click through each part in turn to see what happens to the data as it flows through. A separate panel shows several blocks stacked together to form the full transformer, with the data being passed from one block into the next. Goal: you should leave able to point at each part of a transformer block and explain in plain language what job that part is doing.

## 11\. From the architecture to ChatGPT

The final step is to connect the architecture you have just understood to the products you have probably used. A transformer trained on a very large amount of text, with the single task of predicting the next token at every position in that text, gradually develops a wide range of abilities that look like reasoning, writing, and answering questions. The model is never given any of these abilities directly. They emerge from the same next-token prediction task that we described in Section 5, repeated across hundreds of billions of words, on a model with hundreds of billions of parameters inside it.

The journey you have just walked through is the journey that ChatGPT and every other modern large language model is built on. The exact details of the architecture vary from one model to the next, and the training data differs between them, but the underlying structure that you have built up in this session is shared by all of them in essentially the same form.

> A short timeline shows the same transformer architecture being scaled up across several generations of models, with samples of the kind of text that each generation is capable of producing. Goal: you should leave with the picture that the difference between a small research transformer and a product like ChatGPT comes mostly from the amount of training data, the size of the model, and the engineering effort poured into the training run, rather than from a fundamentally different architecture.

## 12\. Minor topic: how this course actually runs

The Major topic above is the technical anchor for the rest of the course. The Minor topic for this session covers the operational picture, which is how the next six months of your life will be structured, and what is expected of you across that period. There are three things to internalise before next Saturday.

The first is the shape of the calendar. The course is twenty classes long, with each class held on a Saturday morning at seven o'clock India time. Across those twenty classes you will travel from this conceptual foundation to a working understanding of every layer in a modern training stack. On the twentieth class you will help launch an actual training run of a one hundred and twenty billion parameter mixture of experts model. The training run continues past the formal calendar, with students staffed into ongoing roles, until the trained model is released openly on HuggingFace and on ollama, accompanied by a research paper on arXiv.

The second is the way the course operates internally. The course is run as a research lab, and the open source training framework that the lab builds together is the main public artifact that comes out of the course. You will all work on the same framework over the six months, with the instructor leading the live build during class hours, and groups of students contributing candidate techniques such as new optimizers, new attention variants, and new memory tricks through a structured review process. The techniques used in the final training run will be the ones that come out of the field as state of the art at that moment in time, and the techniques you contribute will be the ones that survive a clear and well-defined testing process.

The third is a rule that will come up several times over the course, but which you should hear once today so that nothing about it takes you by surprise later on. The rule is that no contribution to model architecture or to training code will be accepted from a student who has not also contributed clean training data. The threshold for this data contribution is at least one billion clean tokens, with documented provenance for each shard, before any other technical contribution from that student is reviewed. Data is the raw material that every other piece of the course rests on, and the rule exists because building that raw material is the only way the course actually produces a working model at the end.

## 13\. What's next

Session 2 picks up the thread exactly where this session leaves it. You have seen in this session that words have to be converted into numbers before any neural network can begin to process them, and that the conversion process begins with a step called tokenization. Session 2 will go into the design of tokenizers in detail, including how the choice of vocabulary size is made, how subword units are constructed, and why the choice of tokenizer is one of the most important decisions in the entire pipeline of building a language model.

---

**The one thing to carry from this session.** A transformer is a stack of repeating blocks, each block is organised around an attention mechanism, attention is a process where every word forms a query and a key and a value and uses them to decide which other words matter for its meaning, and the entire stack is trained on the simple task of predicting the next token over an enormous amount of text. Every later session in the course will add detail to one piece of this picture without changing the shape of the picture itself.

## 14\. References:

- [LigntningLM - ERA V4 Training](https://lightninglm.theschoolofai.in/)
- [Arcturus V1 from EAG V2 Capstone](https://www.youtube.com/watch?v=fSmfjSdout4)

[Session Transcript](https://docs.google.com/document/d/1j9smTqjs4IybP_a50tp7id3uRnlZwqFgxgDnlxrYGh0/edit?usp=sharing)

## Video

**Studio**

**GMeet**
