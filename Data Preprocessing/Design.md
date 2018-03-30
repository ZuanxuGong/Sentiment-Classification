# Data Preprocessing

## Approach 1: Separate lines and words
```python
for line in open(filename):
    for word in line.split():
        word = word.strip(strip)
```
Here stripe includes whitespace, punctuation, digits and “\ ” ’ ”

## Approach 2: Use unified lowercase
```python
word = word.lower()
```
This is because function WordNetLemmatizer().lemmatize needs a lowercase word as an input.

## Approach 3: Lemmatization
```python
word = WordNetLemmatizer().lemmatize(word, 'n')
word = WordNetLemmatizer().lemmatize(word, 'v')
word = WordNetLemmatizer().lemmatize(word, 'a')
```
For example, this will turn “liked” into “like”, “is” into “be”, “conversations” into “conversation”. However, this package is not 100% accurate.
 
## Approach 4: Delete some noisy words
Definition of the noisy words: <br>
The Length of the word is less than 2. For example, “I”, “a”. <br>
The frequency of the word in all 3000 words is less than five times, such as “zombie”, “lino”, “tank”, etc. <br>

## Approach 5: Normalization
The original frequency of a word is larger than 1, so I use frequency/MaxF to normalize the value into [0,1]. <br>
MaxF: max frequency of a word in a line.

## The approach or procedure to construct a feature vector
### Step1
Put all words into a dictionary to record each word’s frequency in all 3000 samples. (These words are unified lowercase and lemmatized. The lengths of these words are larger than 1)
### Step2 
Delete some words whose frequency is less than 5 times in all 3000 samples and give the position on the feature vector to other words.
### Step3
Find the MaxF to normalize the feature vector into [0, 1]

## Difficulties and words that are not filtered:
Failed to turn “you’re”, “you’d”, “you’ll” into “you are”, “you would”, “you will”
