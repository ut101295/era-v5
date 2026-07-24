---
title: "Axiom — Learning OS"
source: "https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cmq9gi24c03yf08o80th20zlh/lesson"
author:
published:
created: 2026-07-10
description: "A premium learning operating system for ambitious technical education."
tags:
  - "clippings"
---
[Axiom](https://axiom.theschoolofai.in/dashboard)

[Settings](https://axiom.theschoolofai.in/settings)

## Session 2: Tokenization and Vocabulary Design

## 1\. What this session is

Session 1 closed on a single promise, which is that **words have to become numbers** before any neural network can begin to work on them, and that the first step in turning words into numbers is a step called **tokenization**. This session opens up that one step and shows that it is far more consequential than it first appears. Tokenization is the place where raw human text is chopped into the units that the model will actually see, and the way you perform that chopping quietly determineds the **cost**, the **fairness**, and even the **basic spelling ability** of every model trained on top of it.

The plan for the session is to begin with the problem rather than the solution, so that every technique we introduce arrives as the answer to a difficulty we'll face. We will build up from the two obvious and broken ways of splitting text, discover subword tokenization as the middle path, learn the Byte-Pair Encoding algorithm as a loop you could run by hand, and then widen out to the family of related algorithms you will choose between when you build your own. From there we look at vocabulary size as a real engineering dial, at a frequency-sorted identifier scheme that came directly out of the V4 BrahmicTokenizer work, at the precise seam where tokenization ends and the embedding table begins, and finally at the Indic-script challenges that make this work a question of sovereign capability rather than a thin wrapper around someone else's tokenizer.

The real objective of today is not to have you reproduce an existing tokenizer. It is to equip you to design your own and to defend the choices behind it, which is exactly what the closing hands-on exercise asks you to do.

## 2\. The dilemma before the algorithm

Before we reach for any algorithm, it is worth feeling the problem that the algorithm exists to solve. Take a single sentence and try to split it in the two most obvious ways. The first way is to split it into individual characters, which gives you a tiny set of symbols to learn, since a couple of hundred characters cover most of English. The cost of that choice is that sequences become enormous, and the model spends a large part of its capacity rediscovering, over and over again, that the three characters t, h, and e tend to travel together as the word "the." The second way is to split the sentence into whole words, which gives you short sequences that are pleasant to work with. The cost of that choice is that the vocabulary explodes without limit, every typo or rare word falls off the edge as an unknown token, and morphologically rich languages, where a single root spawns dozens of surface forms, become hopeless.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_1_tokenization_dilemma_high_impact.html" width="100%" height="840" title="widget_1_tokenization_dilemma_high_impact"></iframe>

The question that this tension plants is the one that the rest of the session answers. Is there a middle path that keeps the common things whole and breaks only the rare things into pieces? That middle path exists, it is called subword tokenization, and **Byte-Pair Encoding** is one well-known way to find it. Everything that follows is built on top of this felt problem, so it is worth spending real time here until the trade-off feels obvious rather than abstract.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_1_tokenization_dilemma_0_0.html" width="1280" height="1150" title="widget_1_tokenization_dilemma_0_0"></iframe>

> Goal: you should leave feeling, rather than just being told, that neither extreme is workable and that something in between is needed.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_1_tokenization_dilemma_1_1.html" width="100%" height="700" title="widget_1_tokenization_dilemma_1_1"></iframe>

## 3\. Why tokenization is load-bearing

Now that the trade-off is in your hands, it is worth making the stakes explicit, because tokenization is unusual among the decisions in a training pipeline. It is decided exactly once, before training begins, and everything downstream inherits it permanently. The length of every sequence, and therefore the compute spent on every sentence, is fixed by it. The set of things the model is even capable of spelling is fixed by it, because a string that never became a token has no symbol the model can ever produce. The fairness of the model across different languages is fixed by it. And the size of the embedding table, which is one of the largest single pieces of the whole model, is fixed by it. You cannot repair a bad tokenizer by training for longer, in the same way that you cannot fix a badly drawn map by walking faster.

The clearest way to feel the cost is through the bill. Suppose your tokenizer splits Telugu into roughly three times as many tokens as it uses for the same meaning in English. Every Telugu user then pays close to three times the inference cost for the same request, and the model sees only about a third as much Telugu context inside the same fixed window. This is not a rounding error and it does not heal with scale, because a larger context window lifts both languages together and leaves the ratio between them untouched. This is the natural place for the Indic-script motivation to enter the session, not as a footnote at the end but as a direct consequence of a single early decision.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_2_why_tokenization_is_load_bearing_0.html" width="100%" height="1440" title="widget_2_why_tokenization_is_load_bearing_0"></iframe>

> A view of the tokenizer as a frozen decision, with a set of presets that lean the tokenizer toward characters, toward English words, or toward a balanced subword scheme, and a panel that traces how each downstream property changes as a result.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_2_why_tokenization_is_load_bearing_1.html" width="100%" height="1240" title="widget_2_why_tokenization_is_load_bearing_1"></iframe>

> An interactive cost calculator where you pick a language and watch the same sentence cost more tokens, more money on every request, and less usable context inside a fixed window, with a lineage panel that names the four things this one choice silently locks in.

## 4\. Byte-Pair Encoding, step by step

With the stakes clear, the algorithm itself turns out to be almost disappointingly simple, which is a good sign. Byte-Pair Encoding is a loop you could run on a whiteboard. You begin by treating every word as a sequence of single characters. You count every adjacent pair of symbols across the whole corpus, weighted by how often each word appears. You find the single most frequent pair and merge it into one new token, and you write that merge down in an ordered list. Then you repeat the count and the merge and the record, again and again, until the vocabulary reaches the size you were aiming for. That is the entire training procedure.

Two artifacts come out of this loop, and it is worth naming them clearly because everything later depends on them. The first artifact is the vocabulary, which is simply the set of tokens that now exist. The second artifact is the ordered list of merges, which is the set of rules. The rules are ordered for a reason, because to tokenize a brand new piece of text you replay exactly these merges in exactly this order, and the order is what makes the result deterministic. If you watch a tiny corpus go through three or four merges, you can see related words such as "low" and "lower" and "lowest" come to share the same early merges, which is the precise moment the middle path stops being an idea and becomes a mechanism.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_3_bpe_step_by_step_0.html" width="100%" height="1700" title="widget_3_bpe_step_by_step_0"></iframe>

> A clean step-through of the core loop, advancing one merge at a time so you can watch characters fuse into larger and larger tokens while the merge list grows beside them.

## 5\. The family of algorithms

Byte-Pair Encoding is the easiest member of a small family to understand, and it is worth meeting its relatives as a menu of choices rather than as a history lesson, because you will pick from this menu when you build your own tokenizer. WordPiece keeps almost the whole structure of Byte-Pair Encoding but changes the rule for which pair to merge. Instead of merging the pair that is simply most frequent, it merges the pair that most increases the likelihood of the corpus, which is a way of preferring pairs whose two halves travel together far more often than chance would predict. The practical effect is a gentle bias toward subwords that feel meaningful, so that a tightly bound pair can win a merge even when a more common but less cohesive pair exists.

SentencePiece is the member of the family that deserves the most of your attention, because it changes something more fundamental than the merge rule. It treats the raw input as a single stream of Unicode, spaces included, and it represents each space explicitly with a marker rather than throwing whitespace away. This makes the tokenizer language-agnostic and fully reversible, since you can always reconstruct the original text exactly, and it removes the hidden assumption that whitespace cleanly separates words. That assumption is an English habit, and it is exactly the assumption that breaks for many Indic scripts, where word boundaries are not marked the way they are in English. When you are deciding between these options, the right question is never which one came first. It is which choice fits the languages, the scripts, and the reversibility your system actually needs.

## Example

Suppose the current corpus tokenization gives:

| Pair | Pair count | Left-token count | Right-token count | WordPiece score |
| --- | --- | --- | --- | --- |
| `One + Plus` | 100 | 1,000 | 1,000 | `100 / (1,000 × 1,000) = 0.0001` |
| `i + Pod` | 40 | 45 | 50 | `40 / (45 × 50) = 0.0178` |

### BPE chooses

`One + Plus`

because 100 is greater than 40.

### WordPiece chooses

`i + Pod`

because `i` and `Pod` occur almost exclusively together. Meanwhile, `One` and `Plus` are common tokens that frequently occur in many other contexts.

Therefore:

- **BPE:** “Does this pair occur often?”
- **WordPiece score:** “Does this pair occur together unusually often relative to its parts?”

It is still frequency-based, but it measures **association**, rather than raw frequency.

## SentencePiece

**SentencePiece is not a third merge rule like BPE or WordPiece.** Its major distinction is **how it treats the input text**.

### The key idea

Traditional tokenizers often do this first:

```
"the cat"
→ ["the", "cat"]
```

They split on spaces, then learn subwords inside each word.

SentencePiece instead treats the sentence as one raw character stream and converts spaces into a normal visible symbol, usually `▁`:

```
"the cat"
→ "▁the▁cat"
```

Possible tokens might be:

```
["▁the", "▁cat"]
```

or:

```
["▁", "the", "▁ca", "t"]
```

The `▁` means:

> A space occurred before this token.

Because whitespace is encoded explicitly, detokenization is straightforward:

```
["▁the", "▁cat"]
→ "▁the▁cat"
→ " the cat"
```

The initial space is typically removed.

## Comparison

| Method | Main selection rule | Vocabulary construction | Whitespace handling |
| --- | --- | --- | --- |
| BPE | Merge most frequent pair | Starts small and grows | Usually pre-tokenized |
| WordPiece | Merge pair with strongest normalized association or likelihood gain | Starts small and grows | Usually pre-tokenized |
| SentencePiece | Retain tokens that best preserve corpus likelihood | Starts large and prunes | Space becomes `▁` |

The cleanest statement is:

> **SentencePiece defines how raw text is represented and tokenized without requiring language-specific word splitting. BPE or Unigram defines how its vocabulary is learned.**

## 6\. Vocabulary size as a real dial

Vocabulary size is often presented as a single bullet point, but it is better understood as a dial that pulls in two directions at once, and the whole point is that there is no neutral setting. When you choose a small vocabulary, your sequences become longer, because each piece of text is now made of more, smaller tokens, and longer sequences mean more compute spent on every single sentence. The compensation is that the embedding table stays tiny. When you choose a large vocabulary, your sequences become short and cheap to process, but the embedding table and the matching output projection grow in a straight line with the vocabulary size, and at smaller model scales these two tables can come to dominate the entire parameter count of the model.

This is where the dial stops being a tokenizer question and becomes an architecture question. At the 131K vocabulary used in the V4 BrahmicTokenizer, paired with a model width of 8096, the embedding table alone holds on the order of a billion parameters, which is already larger than many complete models and occupies a couple of gigabytes before a single transformer layer has been added. At that size, the question is no longer only which tokens to include. It becomes the question of how you are going to store this matrix at all, and that question is the deliberate hand-off into the seam we look at next.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_5_vocab_size_dial.html" width="100%" height="1180" title="widget_5_vocab_size_dial"></iframe>

> An interactive dial where you slide the vocabulary size and watch the two ends of the trade-off move in opposite directions, the sequence length and compute on one side and the embedding table on the other, while a parameter breakdown shows the tables quietly taking over the model at smaller scales. A hand-off panel reports the size of the table at the 131K setting and points directly at the storage problem.

## 7\. The seam, from token IDs to embeddings, and where Kronecker lives

It helps to be very precise about where one layer ends and the next begins, because a lot of confusion in this area comes from blurring the boundary. The tokenizer's entire job is finished the moment text has become a sequence of integer identifiers. Nothing about the tokenizer says what those integers mean as vectors. The very next question, which is what vector each identifier turns into, belongs to a different component entirely, which is the embedding table. Keeping these two layers separate in your mind is important, because it tells you exactly which kinds of fixes belong on which side of the seam.

The storage problem we set up in the previous section lives entirely on the embedding side of this seam. A dense embedding table at 131K by 8096 is enormous, and one way to tame it is to stop storing it densely. **Kronecker factorization** represents that single large table as the Kronecker product of two much smaller matrices, which reconstructs a table of the same shape while storing only a tiny fraction of the parameters. This is the technique that the V4 work released openly under an Apache 2.0 license, on the strength of how unusual it is. For today the important thing is not the mechanics but the placement. Kronecker factorization is an embedding-side trick, not a tokenization trick, and it lives on the far side of the seam from everything else in this session. The full treatment belongs to its own class on embeddings and model internals, and we will give it that treatment in Class 7.

## 8\. The Indic-script deep dive

Now that you have the full vocabulary of ideas from the rest of the session, the Indic-script problems can be pulled into one focused block, because you finally have the language to understand why they are hard. Scripts such as Devanagari, Telugu, Bengali, and Odia are not simply English written with different letters. They carry conjunct consonants, where several consonants fuse into a single written cluster. They use joiner characters, the zero-width joiner and the zero-width non-joiner, which are invisible on the page yet carry real meaning and change what the correct token boundaries are. They attach vowel signs, the matras, onto consonants, and a naive tokenizer will happily slice through the middle of one of these grapheme clusters. On top of all of this sit Unicode normalization issues, where the very same visible character can be encoded as more than one underlying byte sequence.

In Hindi, several consonants can combine into one visual unit called a **संयुक्ताक्षर** or conjunct consonant.

For example:

```
क् + ष = क्ष
त् + र = त्र
ज् + ञ = ज्ञ
श् + र = श्र
```

The symbol `्` is the **halant/virama**. It removes the inherent vowel from a consonant.

Normally:

```
क = ka
क् = k
ष = ṣa
क् + ष = क्ष
```

Although `क्ष` may look like one character, Unicode stores it as multiple code points:

```
क + ् + ष
```

This matters to tokenizers. A tokenizer that splits by visible glyphs may treat `क्ष` as one unit, while a tokenizer operating on Unicode code points sees three components.

## Hindi example: शक्ति

The word:

```
शक्ति
```

contains the conjunct `क्त`:

```
श + क् + त + ि
```

Visually, `क्` and `त` combine into `क्त`. A tokenizer should ideally avoid breaking the word at an arbitrary point inside this sequence, such as:

```
शक | ् | ति
```

A better segmentation might preserve the conjunct:

```
श | क्ति
```

or preserve useful linguistic pieces:

```
शक्ति
```

## Zero-width joiner and non-joiner

Hindi text can also contain invisible Unicode characters:

- **ZWJ:** zero-width joiner, `U+200D`
- **ZWNJ:** zero-width non-joiner, `U+200C`

They are placed around the halant to influence how consonants are rendered.

### Normal conjunct

```
क + ् + ष
```

Usually renders as:

```
क्ष
```

### With ZWNJ

```
क + ् + ZWNJ + ष
```

This requests that the consonants remain visually separate:

```
क्‌ष
```

The halant may remain visible rather than forming the `क्ष` ligature.

### With ZWJ

```
क + ् + ZWJ + ष
```

This requests a joined or half-form rendering:

```
क्‍ष
```

The exact visual result depends on the font and rendering engine.

## Why this matters for tokenization

These strings may look identical or nearly identical:

```
क्ष
क्‍ष
क्‌ष
```

But internally, their Unicode sequences differ:

```
क ् ष
क ् ZWJ ष
क ् ZWNJ ष
```

A naïve tokenizer may therefore assign different tokens to text that a reader considers equivalent, or split inside a conjunct incorrectly.

In Hindi and other Indic languages, ZWJ and ZWNJ usually carry **rendering and orthographic intent**, rather than changing the dictionary meaning of the word. Their importance is that they change the underlying character sequence and sometimes the visible conjunct form, which directly affects normalization and token boundaries.

The failure to watch for is concrete and damaging. A naive byte-level Byte-Pair Encoder, run without any awareness of grapheme boundaries, will sometimes place a token boundary in the middle of a Telugu syllable, so that the model is handed a half-formed grapheme that corresponds to nothing a human reader would recognise. This is the failure that motivates the surgical retrofit at the heart of the V4 BrahmicTokenizer, where Brahmic slots are allocated deliberately and inserted into a base vocabulary in a controlled way rather than left to chance. This is the emotional core of why the work matters. Building a tokenizer that respects these scripts is what makes the difference between sovereign capability and a thin wrapper around a tokenizer that was never designed with these languages in mind.

<iframe src="https://axiom.theschoolofai.in/widgets/widget_8_indic_script_deepdive.html" width="1280" height="960" title="widget_8_indic_script_deepdive"></iframe>

> A walkthrough across several Indic scripts that shows each specific hazard in turn, the conjuncts, the invisible joiners, the combining vowel signs, and the normalization traps, and then demonstrates the concrete failure of a naive tokenizer breaking a syllable across a token boundary.

## 9\. Build it, then design it

Your assignment would be to pick India's page on Wikipedia in English, Hindi, Telugu, and one more language of your choice. Ask your AI Agent to design a BPE tokenizer in such a way that:

- you have 10000 tokens (your vocab) overall for all languages,
- (Total English tokens)/(Total English Vocab, say 5000 words) must be around 1.2 or less, let's call this X1
- Similarly ratios for your Hindi, Telugu and another language is X2, X3, X4
- Sort X1, X2, X3, X4.. say its X4 (largest), X2, X3, X1 (least).
- Your assignment score is going to be 1000/(X4 - X1).

## 10\. Minor topic: opening up the transformer block

Session 1 ended with the whole transformer block drawn as a single picture, with attention at its centre and a few supporting pieces around it. This minor topic opens up four of those pieces and looks at each one on its own. The thread that runs through all four is the same one that made depth convincing in Session 1, which is that we do not just describe a part, we let it prove itself by training a real model in the browser and watching it either solve a task or fail to. Every loss curve below comes from actual forward passes, real cross-entropy, and hand-written gradient descent running on your machine, not from a scripted animation. We deliberately stay with the parts you can watch win on toy data, and leave inference-time machinery such as caching for a later class.

## A transformer reads a set, so order needs its own signal

The first thing to understand about the input is that a transformer does not naturally know which token came first. It receives the token embeddings as a set, and on its own it has no way to tell one ordering of the same tokens apart from another. The fix is to add a second learned vector to each token that depends only on its slot in the sequence, so the same word in position one and position three becomes two different inputs. This is the positional embedding, and it is the difference between a model that can reason about order and one that cannot.

<iframe src="https://axiom.theschoolofai.in/widgets/m1_widget_1_token_plus_position_.html" width="1280" height="1040" title="m1_widget_1_token_plus_position_"></iframe>

> Two tiny networks train side by side on the same order questions, where the dataset is rigged so every sequence appears once with each label under a swap of two tokens. The token-only model pools its inputs into a bag and therefore sees the swapped pair as identical, so its loss is pinned at chance. The token-plus-position model gives the same token a different vector in each slot and learns every rule.

## Attention, the actual computation

Session 1 described attention through the roles of query, key, and value. Here we make the computation concrete. Each token projects itself into a query, a key, and a value. The query of the current token is compared against the key of every visible token by a scaled dot product, which produces a score for each. A softmax turns those scores into weights that sum to one, and the output of the token is the weighted sum of the value vectors. In a decoder the comparison is restricted so that a token can only look at itself and earlier tokens, which is enforced by masking the future positions before the softmax.

<iframe src="https://axiom.theschoolofai.in/widgets/m3_widget_3_attention_what_it_actually_is.html" width="1280" height="950" title="m3_widget_3_attention_what_it_actually_is"></iframe>

> A single-head causal attention walkthrough on one sentence. Step through each word and watch its query score the visible keys, the softmax turn those scores into a distribution that sums to one, the future cells stay masked, and the output assemble as a blend of values.

## Why query, key, and value are three projections, not one

The most common question once the computation is clear is why we bother with three separate projections when all three start from the same token. The answer is that the projection is what defines the role. The query and the key together create the matching relationship, which decides who attends to whom, while the value carries the content that gets transported once a match is made. Collapsing these destroys the model's ability to attend asymmetrically, where one token needs to point at a specific other token rather than simply at whatever looks most like itself.

<iframe src="https://axiom.theschoolofai.in/widgets/m4_widget_4_why_qkv_must_be_separate.html" width="1280" height="1080" title="m4_widget_4_why_qkv_must_be_separate"></iframe>

> The same small attention model is trained four ways on a task where each token must retrieve its clockwise neighbour, which is a relationship that is not symmetric. With separate query, key, and value the model solves it. With query and key tied the score matrix is forced to be symmetric, and the cycle it would need to represent is mathematically impossible, so it stalls. The partial ties land in between.

## Normalization, what keeps a deep stack trainable

The last piece is the one that makes it possible to stack many blocks without the numbers running away. A normalization step rescales the activations of each token before they enter the next sub-layer. LayerNorm, used in the original Transformer and in GPT-2 and BERT, takes the feature vector of a single token and subtracts its mean and divides by its standard deviation, so each token leaves with mean zero and unit scale. RMSNorm, used in LLaMA, Mistral, Gemma and most recent large language models, does the cheaper thing of dividing by the root mean square only, without removing the mean, and works just as well. In both cases the statistics for one token come only from that token and never from its neighbours.

<iframe src="https://axiom.theschoolofai.in/widgets/m2_widget_2_normalization.html" width="1280" height="950" title="m2_widget_2_normalization"></iframe>

> The matrix view shows exactly what LayerNorm and RMSNorm do to a set of token activation vectors, and names which models use which. Alongside it, a sixteen-block residual network trains on a task where scale is deliberate noise, once with normalization and once without. Without it the activations grow until the loss becomes meaningless, and with it the curve stays smooth. Goal: you should leave knowing both what these norms do to the numbers and why a deep model will not train without scale control.

## 11\. What's next

Session 3 turns from the shape of the tokens to the raw material they are made of, which is data. The class is titled Data Collection and Sourcing, and it marks the start of the four-terabyte cleaning effort that the entire training run depends on. You have seen this session that the tokenizer is decided once and inherited forever, and the same is true, only more so, of the data. From Session 3 onward the course becomes a working lab, and the first and most important flywheel, the gathering and cleaning of real training data, starts to turn. There is a gating rule attached to that work which you will hear stated in writing during the class, and it exists because no architecture or optimizer contribution means anything without the cleaned, documented data underneath it.

---

**The one thing to carry from this session.** Tokenization is a single decision made before training that the whole model inherits and cannot later undo, the Byte-Pair Encoding algorithm that produces it is just count the pairs, merge the most frequent, record the rule, and repeat, the vocabulary size you pick trades sequence length against the size of the embedding table, and the deliberate choices on top, frequency-sorted identifiers and a script-aware design for Indic languages, are what turn a generic tokenizer into one worth building yourself. Tokens become integer identifiers, and only then, on the far side of a clean seam, do those identifiers become embedding vectors.

[Transcript](https://docs.google.com/document/d/1KzmEEoziEeuZVUQa33pf5FEdcdmkFH_HvWDZk8odtes/edit?usp=sharing)

## Video

**Studio**

![](https://www.youtube.com/watch?v=oPSXIGzlX40)

**GMeet**

![](https://www.youtube.com/watch?v=CRj9F9mJcRM)