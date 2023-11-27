# Introduction
This repo implements a solution of a semantic photo rating task.
# Overall problem statement
Classification of a photo as "worth to be printed for an album" and "not worth to be printed" is highly
subjective. The business-case of Stampix makes this classification even more challenging since the
classifier needs to be able to classify photos that underwent some prior selection by a user (since
those photos were already printed). So, we deal with a case where those photos that are obviously boring,
overlighted or just made by accident do not even reach the storage.

Another peculiarity of the problem is that our classification is dependent not only on the photo istelf,
but on other photos from a collection also. Since if we have just 10 photos and 15 places in an album
then there is no need to make any choice. Another edge case is when we have 15 photos from the best
photographers and only 10 places in an album. Then in the frameworks of ordinary classification we
should classify 5 of those photos as "bad". Long story short, classification doesn't work here. We
need to assign a numeric ratings to photos.

Photos can be rated taking into account its human-understandable content (semantics), jpeg compression
quality, color spectrum. Perhaps, there are characteristics of a photo, but most probably all of them may
be assessed with some traditional numeric methods (like, by means of Fourier transform or SVD), which is
less challenging. In contrast, the assessment of semantics is at the cutting edge of today's Computer
vision technology - it is the most challenging sub-task.

## Semantic rating of photos
At this study we will tinker with [CLIP-model](https://arxiv.org/abs/2103.00020) developed by OpenAI.
It has 2 major components:
1. text encoder
2. picture encoder

Each component is meant to build an embedding of its input modality. And the crucial thing here that those
embeddings should exist in the same space, so cosine distance would make sence. This unification of text
and images allowed to train CLIP in semi-supervised manner on a huge collection of data producing a
general-purpose pre-trained model.

The generality of the model lets us hope that in 768-dimensional embedding space we could find an axis that
has a semantics of "how visually appealing is this photo?" If this axis exists then we can use the
corresponding embedding as a master-vector in the following criterion:  
__The greater is the scalar product of photo embedding and master-vector, the more visually appealing is
the photo.__  
In line of a "contrastive learning" (which was used to pre-train CLIP) we could use the embedding of a
prompt like  
"_a professionally-looking photo with nice composition that is worth placing into a photobook_"  
but it is not fine-tunable for some particular user and it is not apparent in advance whether it will work
better then one or another fine-tuning method.

## Problems of CLIP model
Generally, CLIP model has a great potential because it is based on transformer architecture, and thus might
be able to process images of arbitrary size, but currently available version works only with 224x224 images,
so there is a default preprocessing in the model - the largest central square patch is taken out of a picture
and then resolution is reduced to 224x224, thus it is not able to reliably estimate composition and image
quality. But any more ambitious solution would be of production-level complexity.

## How to test a model
One needs to clone a repo, `cd` to it's root and run
```commandline
docker build -t clip_testbench:latest -f photo_classifier.Dockerfile .
```
After the build is complete, run
```commandline
docker run -it -p 5000:5000 clip_testbench:latest
```
then visit `localhost:5000` in a browser. Interface allows to load 2 jpeg photos and pick up a
master-vector that should be used to their mutual rating.
The outputs are:
1. a probability of the 1st photo being the best for an album
2. a probability of the 2nd photo being the best for an album

In a dropdown a data-centric master-vector is an embedding that was calculated with a formalism described
in 'prompt_engineering.ipynb' jupyter notebook. Technical and mathematical details are reported there.

## A room for improvement

1. As it is briefly discussed in 'prompt_engineering.ipynb' we calculate embedding by the use of a big matrix
with around 1000 x 1000 size, but its rank is not greater then the amount of photos used for training.
A method for composing a much smaller matrix with maximal rank and maximal informativeness can be developed,
but it requires some scrutiny.
2. The dataset gathering is also a tedious task and it definitely makes sense to look for readymade large
datasets. Something like a collection of Instagram photos together with amount of likes divided by the
amount of subscribers might be very informative in terms of defining a tier of any photo.
3. Photo pairs with big tier differences should be penalized stronger for incorrect classification, but
this would elevate complexity of the math involved even higher. But for production works it is definitely
doable
4. Assessment of discerning power of a rating model is not done since it is as non-trivial as picking up
best photo pairs for learning (see p.1 of this list). But one can do sanity check with the web-server.