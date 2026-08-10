# RLHF for Small Language Models — Dense Reward Shaping

A reinforcement learning project that teaches a small language model (**GPT-2, 125M parameters**) to write more coherent short stories — by giving it feedback at **every step** of writing, instead of a single score at the end. The project shows that for a model this small, that steady stream of feedback is the difference between learning and not learning at all. Every part — the reward model, the training loop, and a controlled experiment comparing the two kinds of feedback — is included and explained step by step.

**Result: with dense step-by-step feedback, GPT-2 beats the original model on 64% of unseen stories — while the usual sparse approach fails to learn anything at all.**

---

## What is RLHF (Reinforcement Learning from Human Feedback)?

RLHF is the family of techniques used to train modern chatbots to be more helpful and coherent. The idea is simple: the model writes something, a **reward** tells it how good that attempt was, and the model gradually adjusts itself to earn more reward next time — much like a student improving through a teacher's feedback. The tricky part, and the whole focus of this project, is *how often* that feedback should arrive.

## What is the Credit Assignment Problem?

Imagine grading a student's entire essay with a single number at the bottom of the last page. The student knows the essay was weak — but not **which sentence** to fix. That is the **credit assignment problem**: from one score at the very end, the model cannot tell which of its many word choices were good or bad.

It gets worse when good scores are rare. If a positive signal almost never appears, the model just drifts, guessing at random — searching for a needle in a haystack. And **small models suffer from this the most**, because they lack the spare capacity that large models use to slowly average out a weak, unreliable signal.

## Why a Small Model?

The biggest AI breakthroughs use enormous models and enormous computing power that most students, researchers, and small teams simply cannot access. Small models are the practical, affordable alternative — and learning to train them *well* is a valuable challenge in its own right. This entire project runs on a **single free GPU** (Google Colab's NVIDIA T4), which is exactly the point: good technique over raw resources.

## The Core Idea — Sparse vs. Dense Rewards

The project compares two ways of rewarding the model, changing **only this one thing**:

| | **Sparse reward** | **Dense reward** |
| --- | --- | --- |
| When feedback arrives | once, at the end of the story | at every step of writing |
| What the model receives | a single yes/no score | a continuous quality score |
| The problem it creates | fires on only **0.04%** of tries — almost no signal | a steady signal the model can follow |

Because everything else — the model, the settings, the random seeds — is held identical, any difference in the results comes from the **density of the feedback alone**.

## How the Code Works — Step by Step

### Step 1: Train the Reward Model (the "judge")

Before the writer can improve, it needs something to score its stories. We train a **RoBERTa-base** model to judge how *coherent* a piece of text is. It learns from real human stories (good examples) versus scrambled, corrupted, or machine-generated text (bad examples) — like learning to grade essays by studying many good and bad ones side by side.
**Output:** a trained reward model that gives any story a coherence score.

### Step 2: Make the Score Usable for Learning

Two design choices turn the judge into a good teacher:

- It reports a **smooth, continuous score** (raw logits) instead of a blunt "good / bad" — so the writer can tell *small* improvements apart.
- It is trained to **rank two stories against each other** (a Bradley–Terry ranking) rather than label them in isolation — giving it a real sense of *relative* quality.

**Output:** a reward signal fine-grained enough for reinforcement learning to follow.

### Step 3: Train GPT-2 with PPO

The writer (GPT-2) reads a story opening, writes a continuation, and is rewarded by the judge. It learns using **PPO** (Proximal Policy Optimization), a standard reinforcement-learning method, via Hugging Face's TRL library. We run this twice — once with the **dense** reward, once with the **sparse** one — across three random seeds (42, 99, 7) so the results aren't a fluke.
**Output:** trained story-writing models for each condition.

### Step 4: Evaluate on Unseen Stories

We test the trained model on **50 story prompts it never saw during training** — the real exam — scoring each with the judge and comparing against the original GPT-2.
**Output:** a win rate and average reward on held-out prompts.

### Step 5: Check for Reward Hacking

A higher score can sometimes mean the model found a loophole (repeating words the judge happens to like) rather than genuinely writing better. So we measure repetition and word variety, and read the outputs by hand.
**Output:** confirmation that the gains are real improvements, not gamed ones.

## The Hardest Part — Getting RLHF to Actually Train

The first versions of this did not just underperform — they refused to learn at all. Run after run, the reward curve sat completely flat, as though the model was not updating.

Getting it to work meant tracking down a chain of subtle failures, each of which quietly killed learning on its own:

1. **The judge was too good — in the wrong way.** My first reward model scored human text as 1 and machine text as 0, with nothing in between. That sounds ideal, but it is useless for learning: if the writer *slightly* improves a weak story, the judge still says "0," so the writer gets no hint that it moved in the right direction. **Fix:** switch to raw, continuous scores and train the judge to *rank* two texts against each other (a Bradley–Terry objective), so it rewards *relative* improvement instead of only perfect-vs-terrible.

2. **Every reward was negative, so the model learned to do nothing.** When all the feedback is negative, the safest move for the model is to change as little as possible and simply avoid the penalty — so it froze in place. **Fix:** add a moving baseline so an *average* story scores exactly 0 and a *better-than-average* one scores positive — finally giving the model a reason to improve rather than hide.

3. **Training kept destabilizing.** The policy would lurch too far in one update and start producing gibberish, its divergence from the original model exploding. **Fix:** normalize the advantages properly and tune the KL penalty that keeps the writer anchored to its original self.

Once these were in place, the reward curve finally began to climb — and the real question, sparse vs. dense, could be answered cleanly.

*(This debugging journey is the part I'm proudest of — diagnosing why an RL loop silently fails is most of the real work in reinforcement learning.)*

## Results

The dense reward clearly wins:

| Measure | Result | 
| --- | --- |
| How often the sparse reward fired| **0.04%** of generations (almost never)|
| Did the sparse model improve in training?| No — stayed flat | 
| Did the dense model improve in training?| Yes — it improved steadily | 
| Held-out reward — original GPT-2| −3.603 |
| Held-out reward — dense model| −3.457 (+0.147) |
| Dense model beat original GPT-2 on| **32 of 50 stories (64%)** |
| Repetitive or gibberish text?| No — dense stays on topic |

The improvement is **modest but consistent** — exactly what you would expect from a model this small — and the reward-hacking checks confirm the model earns its higher scores honestly.

The figure below shows six measurements of the training process over time. The clearest one is the **top-left panel** — the dense model (blue) climbs steadily, while the sparse model (red) stays flat and never gets off the ground.

The exact per-seed training metrics are logged in Weights & Biases: https://wandb.ai/sp2984-northern-arizona-university/trl (dense runs: wandering-wood-271, deft-dew-272, clean-puddle-273; sparse runs: amber-dew-274, bumbling-vortex-275, faithful-pine-276; seeds 42/99/7, 90 steps each.)


[Training results: sparse vs. dense reward]<img width="1289" height="740" alt="figure1_ppo_metrics" src="https://github.com/user-attachments/assets/ad792c08-995f-4016-8c76-a9288cc4da2b" />

## Key Components, Explained Simply

| Component | What it does |
| --- | --- |
| **GPT-2 (125M)** | the "writer" — reads a story opening and writes what comes next |
| **RoBERTa reward model** | the "judge" — scores how coherent each story is |
| **PPO (via TRL)** | the learning method that updates the writer based on its rewards |
| **Bradley–Terry loss** | trains the judge by comparing two stories, producing smooth scores |
| **KL penalty** | keeps the writer from drifting into gibberish while chasing reward |
| **Reward normalizer** | keeps scores in a stable range so training doesn't blow up |

## Training Setup

| Setting | Value |
| --- | --- |
| Policy (writer) model | GPT-2 Base (125M) |
| Reward (judge) model | RoBERTa-base, Bradley–Terry ranking |
| RL method | PPO (Hugging Face TRL) |
| Learning rate | 1e-6 |
| Batch size | 64 |
| Training steps | 90 per seed |
| Seeds | 42, 99, 7 |
| Hardware | 1× NVIDIA T4 (Google Colab, free tier) |
| Total compute | ~3.2 GPU-hours |

## Tech Stack

- **Python** — core language for the whole project
- **PyTorch** — the deep-learning engine everything runs on
- **Hugging Face Transformers** — loads GPT-2 (writer) and RoBERTa (judge)
- **TRL** — provides the PPO reinforcement-learning trainer
- **Datasets** — loads the ROCStories data
- **Weights & Biases (wandb)** — tracks and plots training metrics
- **NumPy · pandas · scikit-learn · Matplotlib** — data handling, metrics, and figures

## How to Run

This project runs as a single **Google Colab notebook**.

1. Open **`dense_coherence_reward_ppo.ipynb`** in Google Colab.
2. Switch to a GPU runtime: **Runtime → Change runtime type → T4 GPU** (training is very slow on CPU).
3. Install the dependencies from the first cells (or run `pip install -r requirements.txt`), then **restart the runtime** before importing.
4. Run the cells from top to bottom. They walk through the stages described above — loading the RoBERTa coherence reward model from Hugging Face, running PPO for the **dense** and **sparse** conditions across seeds 42/99/7, and evaluating on held-out prompts.

The full run is light — about **3 GPU-hours** for all six training runs on a free T4.

## Dataset

**ROCStories** (Mostafazadeh et al., 2016) — about **50,000 short, five-sentence everyday stories** written by people, each capturing a small slice of commonsense cause-and-effect. The human stories teach the judge what good writing looks like, and their opening lines become the prompts the writer is asked to continue.

## Use the Trained Reward Model (no training needed)

The trained judge is available on Hugging Face, so anyone can load it in a few lines:

>  **[https://huggingface.co/Paimagham/roberta-story-coherence]**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("Paimagham/roberta-story-coherence")
model = AutoModelForSequenceClassification.from_pretrained("Paimagham/roberta-story-coherence")
```

## Honest Limitations

- The improvement is real but **small** — exactly what we would expect at 125M parameters.
- The reward is a single score per story, not per word — finer-grained feedback is a natural next step.
- The idea that "small models need dense feedback more" is demonstrated at 125M but **not yet tested on larger models**.
- One random starting seed (123) consistently failed to train and was replaced with seed 99 — noted here for full transparency.

## About the Paper

This code accompanies the paper *Mitigating the Credit Assignment Problem in Small Language Models via Dense Reward Shaping* (submitted to the NeurIPS 2026 LIGHT Workshop: Deployable Small Foundation Models).

```bibtex
@inproceedings{paimagham2026dense,
  title     = {Mitigating the Credit Assignment Problem in Small Language Models via Dense Reward Shaping},
  author    = {Paimagham, SaiBhavitha Reddy},
  booktitle = {NeurIPS 2026 Workshop on Deployable Small Foundation Models (LIGHT)},
  year      = {2026}
}
```

## Why I Built This

After building a neural network from scratch to understand *how* models learn, I wanted to understand how they learn from **feedback** — the mechanism at the heart of modern AI alignment. Reinforcement learning at a small scale turned out to be a demanding teacher: getting PPO to train stably on a 125M model meant diagnosing and fixing a long chain of subtle failures, from exploding KL divergence to a reward signal that gave the model nothing to learn from. Working through each one gave me a hands-on understanding of reward modeling and training stability — the very things that make today's large language models work — which is exactly why I wanted to build it.

## License & Contact

Released under the **MIT License**.

**SaiBhavitha Reddy Paimagham** · [GitHub](https://github.com/paimagham) · paimaghambhavitha@gmail.com

