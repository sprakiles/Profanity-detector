# Profanity Detector 🤬➡️🛡️

yo so i made this AI thing that catches bad words in text... even when ppl try to be sneaky about it lol

## why tho?

was honestly just tired of seeing toxic stuff online and thought "hey maybe i can do something about this" so here we are 

the cool part is it can catch stuff like:
- normal swear words (obviously)
- when ppl replace letters like `@ss` or `sh!t` 
- spaced out words like `f u c k` (nice try lol)
- leetspeak stuff like `h3ll` or `1d10t`

## how good is it?

trained it on like 8000+ tweets and got:
- **93.8% F1 score** - pretty solid tbh
- **94% accuracy** - catches most bad stuff
- **95% precision** - doesnt flag too many innocent msgs 

## installation

u need python 3.8+ and some libs:

```bash
pip install scikit-learn joblib numpy scipy
```

## how to use

### basic usage

```python
from profanity_detector import ProfanityDetector

detector = ProfanityDetector()

# check if something is bad
is_bad, confidence = detector.detect("wtf bro")
print(f"profane: {is_bad}, confidence: {confidence:.1%}")
# profane: True, confidence: 89.2%
```

### censoring bad text

this is the cool part - replaces bad text with #### (same length as original like roblox ac)

```python
from profanity_detector import censor

# clean text stays the same
print(censor("hello there!"))  # "hello there!"

# bad text gets censored
print(censor("go to hell"))    # "##########" 

# even sneaky stuff
print(censor("ur an @sshole")) # "##############"
```

### quick functions

for when ur lazy (like me lmao)

```python
from profanity_detector import is_profane, censor, get_profanity_score

# just need true/false?
if is_profane("some text"):
    print("thats not nice :(")

# want the raw score? (0 to 1)
score = get_profanity_score("some text")
print(f"badness level: {score:.1%}")

# censor with custom threshold (60% instead of default 50%)
clean = censor("some text", threshold=0.6)
```

## threshold??

so the `threshold` param controls how confident the model needs to be before censoring:
- `0.5` = 50% confident its bad (default, catches more stuff)
- `0.6` = 60% confident (less false positives)
- `0.7` = 70% confident (even more careful)

pick what works for ur use case tbh

## test results

ran it on some examples:

| input | detected as | confidence | censored |
|-------|-------------|------------|----------|
| "Hello there!" | ✅ clean | 95% | Hello there! |
| "This is bullshit" | 🚫 profane | 91% | ################ |
| "What the h3ll" | 🚫 profane | 62% | ############# |
| "f u c k off" | 🚫 profane | 100% | ########### |
| "ur an @sshole" | 🚫 profane | 78% | ############## |

## files

```
profanity-detector/
├── profanity_detector.py   # main code
├── model/
│   ├── preprocessor.joblib # text vectorizer thingy
│   └── svm_model.joblib    # the actual trained model
├── data/
│   └── labeled_data.csv    # dataset i trained on 
├── requirements.txt
└── README.md               # ur reading it rn
```

## dataset

trained on the davidson hate speech dataset - its got like 25k tweets labeled as:
- hate speech
- offensive language  
- neither

grabbed it from here: https://github.com/t-davidson/hate-speech-and-offensive-language

## known issues / todo

- [ ] sometimes misses really creative obfuscation... ppl are creative lol
- [ ] could add more languages (rn its english only)
- [ ] maybe make it detect which specific words are bad?
- [ ] add some slang dictionary or smth

## contributing

if u wanna help make it better feel free to open a PR or whatever. no rules really just dont be toxic LOL

## license

MIT - do whatever u want with it honestly

---

made with 💀 and lots of coffee

if this helped u out maybe leave a ⭐? ty!
