# Volleylitics 
This write-up sums up the thoughts, process, issues, and solutions.

## Table of Contents

- [How did I even get the idea?](#how-did-i-even-get-the-idea)
- [The Problem](#the-problem)
- [The Beginning](#the-beginning)
- [Initial Attempt](#initial-attempt)
  - [Whistles??? DSP??? Train models???](#whistles-dsp-train-models)
    - [Where do I even start?](#where-do-i-even-start)
    - [Classifiers](#classifiers)
    - [A little lesson I learned](#a-little-lesson-i-learned)
    - [My solution](#my-solution)
  - [Rally Segmentation](#rally-segmentation)
    - [How?](#how)
    - [CNN Again](#cnn-again)
    - [A problem appeared](#a-problem-appeared)
  - [The Biggest Weakness](#the-biggest-weakness)
- [Ball Detection](#ball-detection)
- [Ground Contact Detection](#ground-contact-detection)
  - [Homography](#homography)
- [Ball Trajectory](#ball-trajectory)
  - [Testing](#testing)
- [The Site](#the-site)
- [Whats next?](#whats-next)
<a id="how-did-i-even-get-the-idea"></a>

## How did I even get the idea?
Aside from the fact that sports analytics is probably pretty popular and pretty straightforward as an idea, during the 23/24 season in my league I started paying attention to some trends we had on our team. I'm a firm believer that our games are decided by the smallest of margins, like weak serves, inaccurate freeball reception or god forbid a freeball that we let hit the ground lol. The next element is our defense. During matches against different teams we started to notice how we block in a disorganized way and take defensive positions in bad ways.
So I thought to myself, I'm a CS student, I record the games, I have the data. Why not do something with it?

<a id="the-problem"></a>

## The Problem
During that season it was my 2nd year of my CS degree so besides fighting for my life against Calculus 2 and Linear Algebra 2 (Cool courses btw). I also didn't have the Computer Vision knowledge for such a project, nor had I ever made any project close to that scale.


<a id="the-beginning"></a>

## The Beginning 
It's 2025, and I'm about to take the Computer Vision course in my uni. My curiosity and lack of patience didn't let me wait, so I began running small tests and trying out POCs for different elements of the game. 
I tested some YOLO model for ball detection and it looked very cool and I thought to myself, "huh this is gonna be easy." Needless to say, I had no idea about model training, evaluation, metrics and such.

<a id="initial-attempt"></a>

## Initial Attempt
After having an encouraging experience with the small POCs I made, it was time to begin the real thing. I crafted a hasty plan in my head and it was along the lines of:
1. Load match
2. Detect ball
3. Find ground contact 
4. Report it
5. Done

One small issue I noticed was that running inference with the model on a 25-second rally took about 3 minutes. A full match video is 1 hour at minimum. That means that one match would take 7 hours to run ball detection on it. Big nono.

So back to the planning board:
1. Load match
2. Detect all whistles
3. Take the last 2 seconds before a whistle
4. Detect ground contact
5. Report

Nice, all I need to do is detect the whistles.
Another thing I noticed was that the current model wasn't strong enough, it missed too many obvious balls so I couldn't rely on that.

<a id="whistles-dsp-train-models"></a>

### Whistles??? DSP??? Train models???


At this point I have felt like my plan is pretty solid and that I should tackle the whistle detection part head on. 
<a id="where-do-i-even-start"></a>

###### Where do I even start?
During the POCs I just eyeballed and worked very inefficiently, so this time I've decided to annotate as many matches as I could, which turned out to be 13 matches. 

A massive mistake I made is to annotate matches that I didn't have saved raw and downloaded from YouTube. Disregarding the compression YouTube does. (Major mistake).

So I tried applying the knowledge about annotation which was basically me going over the video and tapping when a whistle was happening and logging the time.


  ```json
      {
        "match_id": "match1",
        "whistle_id": 1,
        "time": 396.473,
        "type": "other",
        "t_raw": 396.473,
        "global_id": 1,
        "t_anchor": 396.303
    },
  ```
  
A quick run-through about this structure:
Match and whistle id are obvious, the duplicate with time and t_raw is something I was too lazy to fix but basically they mean the time of the keystroke when I was annotating the whistles. "type" is something I've added later which represents the type of whistle annotated. Volleyball contains several types of referee whistles and they all sound the same. Serve whistle which marks the beginning of play, Point end whistle, which is self explanatory, and administrative whistles like substitution, timeout, set and match end whistles.
At some point I realized that It's not reliable because there's a degree of human error such as anticipation and attention so I decided to anchor it to some threshold I set up which would "center" the whistle around it.

```python
# around each manual whistle click:  
# search ±0.60s in the 3700–4300 Hz whistle band  
  
if energy crosses 0.25 and flux is above 0.35:  
	if the rise holds for 3 of the next 5 frames:  
		anchor = onset  
elif peak_energy >= 0.20:  
	anchor = local_peak  
else:  
	anchor = raw_click
```

<a id="so-at-this-point-i-have-a-reliable-annotation-but-how-do-i-actually-detect-the-whistles"></a>

##### So at this point I have a reliable annotation but how do i actually detect the whistles?
Let's not forget that I'm a CS student and unfortunately my uni doesn't have EE courses and let alone DSP courses so I was winging it with the numbers and techniques.
This led me to a trial and error stage that was never ending. What features are good to compare? 

```python
# Convert audio into overlapping short-time spectral frames  
S = stft(audio, sr=22050, n_fft=2048, hop=128)  
mag = abs(S)  
  
# Keep only the whistle frequency region  
band = mag[(freqs >= 3700) & (freqs <= 4300)]  
  
# Represent each frame by whistle-band properties  
band_energy = mean(band, axis=0)  
band_peak = max(band, axis=0)  
band_mean = mean(band, axis=0) + 1e-8  
sharpness = band_peak / band_mean  
flatness = spectral_flatness(mag)  
  
# Compare frames using a whistle-likeness score  
score = band_energy + sharpness - 1.2 * flatness
```

Early on, I discovered that whistles have a frequency of 3700-4300 Hz. giving me this result:
Raw:
<audio controls src="../data/blog/images/match15_179.wav"></audio>
Band filtered:
<audio controls src="../data/blog/images/input_3700_4300.wav"></audio>
I thought I've discovered something and that from here it's gonna be easy. Just take a full match, extract audio, run it through this band-pass filter, take the peaks, and that's it. Well, almost but not at all lol.
This approach would've worked perfectly if a match recording was human cheering, ball hit sounds and ref whistles. A massive obstacle was this:

<audio controls src="../data/blog/images/Shoe_squeak.wav"></audio>
You might think this is a whistle, and a very clear one. 
This is actually a shoe squeak coming from a player dragging his foot on the court.
How do I counter it? 
<audio controls src="../data/blog/images/shoe_squeak_band.wav"></audio>
Obviously the bandpass filtering doesn't help here, as shoe squeaks happen to be at the exact same frequency range as the whistles.

Shoe Squeak:

![Shoe Squeak Spectrogram](../data/blog/images/squeak.png)
Whistle:

![Whistle Spectrogram](../data/blog/images/whistle.png)

To my non-DSP eyes and ears it sounded impossibly similar. 
<a id="classifiers"></a>

#### Classifiers
Looking at the spectrogram and reading a little bit about DSP, I decided to use my fresh knowledge from my CV course and train a small classifier with some handcrafted features and the result wasn't satisfying, it was too strict for a candidate detector but too loose for a final classifier. 
Another issue I suspected I would have and eventually did have was that every ref has his own whistling style, every court has its own acoustics, and the camera distance from the ref also changes.
The next step was making a Reddit post in r/DSP where i explained my methods, surprisingly they approved, and suggested I try tsfresh to find even more features.
Some of the features:
![Handcrafted Features](../data/blog/images/features.png)

<a id="what-is-tsfresh"></a>

##### What is tsfresh?
From the official tsfresh docs:
``tsfresh is a python package. It automatically calculates a large number of time series characteristics, the so called features. Further the package contains methods to evaluate the explaining power and importance of such characteristics for regression or classification tasks
``

So I let tsfresh run on my dataset and sadly, I've already used most of the important features. It did help me improve by 1-2 features but it wasn't really enough. Another suggestion on Reddit was to use a CNN on the spectrograms.

Luckily my CV course was DL/CNN focused so I was already mostly familiar with this approach. One important caveat I had in mind was that I dont want the part of whistle detection to be very compute heavy because that's not even the main point of the project and if it takes 1 hour to find the whistle and not be perfect. So, my smarty pants decided that I should create and train my own CNN with PyTorch and at first it was very promising.
I dont have the exact numbers but it was trained on approx. 6k samples of negatives and positives extracted from my hand labeling and anchoring mentioned above. Val F1 was around 99% and I thought that with more data I could squeeze another 0.5%.

<a id="a-little-lesson-i-learned"></a>

#### A little lesson I learned
At this point I wanted to test the pipeline of whistle detection.
Needless to say, the result was very bad. The precision and recall weren't above 80%.
What was the issue?
No idea. Maybe the hyper params aren't good? Maybe the match I tried on was an outlier?
Well it was none of those. What actually happened was that It's very easy for the model to differ between a perfectly centered whistle and a perfectly centered shoe squeak or other false positive candidate. The issue was that my candidate detector didn't promise a perfectly centered whistle like what the model has trained on. 
<a id="my-solution"></a>

#### My solution
I realized that my naive approach wasn't gonna make it. My CV course has taught me how to annotate images and not sounds, So I didn't have a parallel to bounding boxes for sound snippets.
The solution was to extract training data for the model with the candidate detector.
It worked! The F1 shot up to 92%. Not good.
At this point i decided to take the drastic approach and pull out the big guns.
I used a custom PyTorch ResNet18 whistle classifier that took a 6-channel spectrogram representation as input: full-band log-mel, delta, and delta-delta, together with the same three channels computed over a whistle-focused band. 
Honestly, the log-mel part was a suggestion i also got on the Reddit, and in my trial and error this gave the best result. at that point I already decided that Its time to proceed to the next stage of the project.


So, to sum this up: 
Whistle detection was not a single model pass, but a sequence of filtering steps. I started by converting the match audio into an STFT and scoring each frame with DSP features that tend to highlight whistles, like band energy, sharpness, and flatness. From there I grouped promising regions into rough candidates, refined them around a better center, and filtered them again using rule-based statistics computed from the match itself. Only then did I send the surviving candidates to a learned whistle classifier, which scored 1-second log-mel snippets around each one. The last step was temporal NMS, which merged nearby duplicates and produced the final whistle times used later for rally segmentation.

These are the numbers it gave me:
Precision 98.35%
Recall 99.05%

<a id="rally-segmentation"></a>

## Rally Segmentation
Unlike football (soccer), which is largely continuous, volleyball is naturally structured as a sequence of discrete events.
In other words, in volleyball, the current point has no relation to the previous or to the next, The game "resets" after each point. Similar to tennis actually.
Since my goal was to take the last 2 seconds before a whistle, I actually needed to classify the whistles. Because for every ground contact of the ball there is a whistle representing the point ending but, each point ending whistle has a point start whistle. There are also unrelated whistles between the points for timeouts, subs etc. 
That means I can't blindly take any whistle and roll it back 2 seconds and find the ground contact. In addition, having a good way to segment the video into relevant clips of just gameplay would be a compute saver and very useful for future features.

<a id="how"></a>

#### How?
It is one thing to take an image and detect the ball, another thing to take a whistle-like event and determine whether it's a whistle. But how do I make the computer look at a clip between 2 whistles and determine: "Yes, this is ongoing play"?

My first approach was to just compare motion changes between frames. While it worked when a point was chaotic in the middle, the beginning of a point is always calm because the players await the serve. So, motion difference approach didn't work. 
The second idea I had was to use the audio, when a point ends there's usually an increase in players/crowd cheering. That failed quickly too because at boring games nobody is excited and in super important games where there's hundreds of people in the crowd the audio isnt helpful at all because there's constant noise. This technique is also unreliable.

<a id="cnn-again"></a>

#### CNN Again
During my CV course, our final task was to make a Font recognizer.
I liked the task and I also liked the fact that the grade was given by novelty and efficiency too. So I thought that I can reach a good result with relying heavily on classic CV techniques(Harris Corner detection, Hough transform etc). Spoiler, it didn't work well at all.
but I was actually rewarded by the professor for the attempt and the architecture i made there. In the end i made a version that used some CNN model that I transfer-learned to the dataset given in the course and got a better result. This rant is to say that at that point I really understood how powerful CNNs are.
So after exhausting different ideas like the ones mentioned above. I've decided to try and just dump a bunch of frames like:
In play:
![In play frame](../data/blog/images/inplay.png)
Not In play:
![Not in play frame](../data/blog/images/notinplay.png)
Which worked surprisingly well. about 92% F1, which to me sounded promising because if a point is 15 seconds recorded in 60fps, it gives us 900 frames and with 92% accuracy it's 828 frames accurately labelled by the CNN. Hell yeah I'm taking that.
So with some hysteresis I've gotten a very good result.
<a id="a-problem-appeared"></a>

#### A problem appeared
Since taking all frames out of a point is wasteful, I decided that skipping every 8 frames would be the best balance between efficiency and accuracy. 
When testing I've noticed that even though the model accuracy is pretty high, quite a few rallies are missed. Most of the rallies were short ones without too much play (missed serve, ace serve). I realized that Its the weak spot of my approach.

The way I solved it is quite funny, When in doubt, CNN. With a big enough dataset, I had 11 matches where every match has about 130 rallies. I ran the model to get a probabilities array which looks something like this:

<details>
<summary>Probabilities Array</summary>
 "seq": [
      0.8278279900550842,
      0.6557295641513786,
      0.4901670425227194,
      0.3383643193678423,
      0.3359259634909003,
      0.4160310737412385,
      0.6885888522321526,
      0.8408216003215675,
      0.6790975404508187,
      0.592930322343653,
      0.5550355309187764,
      0.5578257772657607,
      0.5705974914810874,
      0.6039687033855553,
      0.5841894216007657,
      0.5253288944562277,
      0.5903998228034587,
      0.6929735806253217,
      0.8786051273345945,
      0.9573880440056927,
      0.9460174290820805,
      0.9663256912520437,
      0.9882508714993795,
      0.9955477570042466,
      0.9975716905160383,
      0.9944984371011908,
      0.8778138028250805,
      0.7626575394110245,
      0.7848033146424727,
      0.8285958881330006,
      0.8963545596960819,
      0.9398110123595805,
      0.96453598713634,
      0.896205802758535,
      0.8429949133083076,
      0.8089738981892365,
      0.8541021888906304,
      0.8886231295388166,
      0.5426430220555795,
      0.47828843105923036,
      0.824637718272931,
      0.9449348696554549,
      0.975478367371993,
      0.9007188858407921,
      0.8653396235571967,
      0.8961907300082119,
      0.7555388206183318,
      0.5415511399206494,
      0.44114112673383843,
      0.47837905781437584,
      0.7840304802162468,
      0.7299817926955958,
      0.49147151034287884,
      0.6861455723492782,
      0.8424468961628999,
      0.8943316207991707,
      0.886060350471073,
      0.8420206507047018,
      0.8253387941254511,
      0.7999477850066293,
      0.7472121318181355,
      0.7921867069571902,
      0.9041625815208509,
      0.7353247945958936,
      0.6023488496289106,
      0.668218605446093,
      0.5450628151496273,
      0.2729778067030097,
      0.517037771115396,
      0.7783122478109414,
      0.834876102630538,
      0.8905910777323169,
      0.9455393227663905,
      0.8665602959767749,
      0.7395860769531961,
      0.4955592697316956,
      0.32135731568842324,
      0.21938796838124647,
      0.14552350793824056,
      0.12605461591120692,
      0.4122979893828873,
      0.48402535102584343,
      0.3015115134643797,
      0.5781373324418302,
      0.94556019522927,
      0.9688380198045209,
      0.9329512053065844,
      0.8166004816691089,
      0.8407978283034429,
      0.910760078767333,
      0.9511706774885003,
      0.9585648269364334,
      0.9142810557827808,
      0.8643311659495038,
      0.8024783086295116,
      0.4667536041351301,
      0.24674550827705513,
      0.23612937090372807,
      0.3817826923396842,
      0.6016640067100525
    ],
    "label": 1
  },

</details>
And it worked quite well. But not well enough.
My final touch was the cascading:

```python
inplay_ratio = rally_cnn_vote(interval)  
yellow_score = visual_motion_vote(interval)  
score = inplay_ratio + 0.02 * yellow_score  
  
if inplay_ratio > 0.35:  
	rally = True  
elif yellow_score > 12:  
	rally = True  
elif yellow_score > 10 and duration < 6:  
	rally = True  
elif inplay_ratio > 0.25 and yellow_score > 4:  
	rally = True  
elif inplay_ratio > 0.20 and yellow_score > 5:  
	rally = True  
elif inplay_ratio > 0.15 and yellow_score > 2:  
	rally = True  
elif 0.1 < score < 0.3:  
	send_to_HITL()  
else:  
	reject()

```
What's yellow score?
The ball is yellow, and it compares how much the ball has moved in the top side of the image.
Finally, the most ambiguous cases are sent for human evaluation, which takes me approximately 2-3 seconds per clip and the average amount of clips is 30-ish.
>Yellow score is a possible issue because 26/27 season might introduce a new ball which is not yellow. The alternative is discussed below


<details> 
  <summary>Results</summary>
  ```

	======== Rally Detection Evaluation ========
	 GT rallies: 116
	
	===================================
	MODEL: RAW
	===================================
	Detected rallies: 113
	TP: 105 | FP: 8 | FN: 11
	Precision: 0.929
	Recall: 0.905
	Mean start error: 0.052
	Mean end error: 0.059
	
	---- FALSE POSITIVES ----
	FP: 00:10:57 → 00:11:04 (dur 6.12s)
	FP: 00:11:04 → 00:11:26 (dur 21.98s)
	FP: 00:21:15 → 00:21:25 (dur 10.28s)
	FP: 00:21:25 → 00:21:29 (dur 4.03s)
	FP: 00:33:54 → 00:33:58 (dur 4.16s)
	FP: 00:37:21 → 00:37:31 (dur 10.77s)
	FP: 00:42:21 → 00:42:27 (dur 5.46s)
	FP: 00:42:27 → 00:42:29 (dur 2.68s)
	
	---- FALSE NEGATIVES ----
	FN: 00:10:57 → 00:11:25 (dur 28.84s)
	FN: 00:11:44 → 00:11:53 (dur 8.34s)
	FN: 00:14:23 → 00:14:32 (dur 8.66s)
	FN: 00:21:15 → 00:21:29 (dur 14.24s)
	FN: 00:28:01 → 00:28:11 (dur 10.1s)
	FN: 00:37:19 → 00:37:31 (dur 12.65s)
	FN: 00:42:21 → 00:42:29 (dur 8.04s)
	FN: 00:43:44 → 00:43:50 (dur 6.15s)
	FN: 01:05:26 → 01:05:34 (dur 8.13s)
	FN: 01:05:48 → 01:05:58 (dur 9.91s)
	FN: 01:07:45 → 01:07:51 (dur 5.96s)
	
	===================================
	MODEL: WITH_HITL
	===================================
	Detected rallies: 115
	TP: 113 | FP: 2 | FN: 3
	Precision: 0.983
	Recall: 0.974
	Mean start error: 0.059
	Mean end error: 0.058
	
	---- FALSE POSITIVES ----
	FP: 00:33:54 → 00:33:58 (dur 4.16s)
	FP: 00:37:21 → 00:37:31 (dur 10.77s)
	
	---- FALSE NEGATIVES ----
	FN: 00:14:23 → 00:14:32 (dur 8.66s)
	FN: 00:37:19 → 00:37:31 (dur 12.65s)
	FN: 01:05:26 → 01:05:34 (dur 8.13s)
	```
</details>

To me it seemed like I made it worked super quick. In this example we can see that we have:
FP: 00:37:21 → 00:37:31 (dur 10.77s) and FN: 00:37:19 → 00:37:31 (dur 12.65s)
which refer to the same point, so the false positive is actually a real point and since I don't mind the beginning and i just need the last 2 seconds then cases like these aren't an issue for me. Missing and falsely detecting 1-3 rallies isn't a real issue either. 
So I considered this a success.

<a id="the-biggest-weakness"></a>

### The Biggest Weakness
Since the whistle detection isn't perfect. At first I thought that missing and hallucinating a whistle wouldn't be a big problem because I added logic for the rally segmentation that sees if 2 rally intervals are adjacent we can just combine them.
The issue that I didn't take into account is what happens if I detect a real rally but a false whistle got detected near the end of the rally.
![Probability Graph](../data/blog/images/graph.png)
If we imagine a case like the one above, where the blue is the probability that the current frames are rally frames. We get this bug where the rally gets cut short and the ending portion is lost. I added logic that takes short rallies and throws them because a volleyball rally is at the very least 5 seconds.
The straightforward fix here is to improve the whistle detection even more.
An additional idea that Is planned to be implemented is to actually use the ball detection model instead of the yellow motion cue and that should allow giving additional weight to rallies and then shorter portions can be evaluated before getting thrown.

Sadly, this issue affects the entire pipeline, and it is interesting to see how an error in whistle detection cascades into the downstream stages.

<a id="ball-detection"></a>

## Ball Detection
Finally I've reached the stage that I had in mind when I first imagined this project. 
This stage is pretty straightforward. I annotated about 2000 images, which I reused from rally detection. At first I used my own labeling app thing but then i transferred to Roboflow to have a more organized work environment. 
The model is YOLOv11 Object Detection (Nano) and the metrics are:
- **mAP@50:** 97.7%
- **Precision:** 99.5%
- **Recall:** 92.3%
- **F1:** 95.8%

The model does the job well enough. It's mostly far balls that get missed and it's not a real issue because the ground contact occurs on the near side of the field.
The model tends to miss very fast balls traveling almost orthogonally to the camera axis, but this is an expected and relatively uncommon failure case.

<a id="ground-contact-detection"></a>

## Ground Contact Detection
Now this is an interesting stage. I realized that Its a non trivial problem to solve because having a stationary camera behind the court makes it impossible to determine the depth of the ball when in the air. 
Adding to that, a ball's ground contact might take 1-2 frames.
Even reliably testing the solutions isn't easy.

<a id="homography"></a>

### Homography
As said above, while the ball is in the air, there is no way to determine its location in space.
**But**, there's a single moment when the ball is close enough to a reference point with known coordinates that I can use.
That moment is the ground contact.
![Court Coordinates](../data/blog/images/court.png)

The volleyball court is 18x9 meters, that means that one half is 9x9 meters.
Before entering this stage in the pipeline, I manually input the 4 court corners representing the close half. With these 4 points we can calculate the homography matrix H:

```python
# The 4 corners
img_pts = np.array([
    [1, 1040],
    [1919, 1034],
    [1426, 779],
    [516, 780]
], dtype=np.float32)

# net line coords
net_line = np.array([
    [471, 530],
    [1441, 530]
], dtype=np.float32)


court_pts = np.array([
    [0, 0],
    [9, 0],
    [9, 9],
    [0, 9]
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, court_pts)
```

A nice heuristic is that when the ball is closer to the ground its y-coordinate in the image space is greater.
So, we track the ball image space coordinates and we take the maximum out of the last 15 positions we tracked. The result in image space doesn't mean a lot to us but since we have the homography matrix we can transform these coordinates into the court plane like this:

```python
pt = np.array([[[985, 575]]], dtype=np.float32)
mapped = cv2.perspectiveTransform(pt, H)
```
and we get the approximate location of the ground contact.

Example:
![Match Frame](../data/blog/images/raw.png)

![Transformed Frame](../data/blog/images/image.png)

After that we just save all the coordinates to a file.

<details>
<summary> File</summary>

```json
[
  {
    "clip_name": "rally_006.mp4",
    "rally_id": 6,
    "start": 470.8136054421769,
    "end": 492.3675283446712,
    "set_id": 1,
    "positions": [
      [
        1240,
        725,
        374
      ],
      [
        1241,
        723,
        384
      ],
      [
        1242,
        722,
        396
      ],
      [
        1243,
        739,
        405
      ],
      [
        1244,
        772,
        411
      ],
      [
        1245,
        799,
        416
      ],
      [
        1246,
        813,
        419
      ],
      [
        1247,
        803,
        418
      ],
      [
        1248,
        792,
        416
      ],
      [
        1249,
        784,
        416
      ],
      [
        1250,
        777,
        416
      ],
      [
        1251,
        767,
        416
      ],
      [
        1252,
        757,
        416
      ],
      [
        1253,
        749,
        417
      ],
      [
        1254,
        738,
        419
      ],
      [
        1255,
        729,
        419
      ],
      [
        1256,
        719,
        421
      ],
      [
        1257,
        710,
        423
      ],
      [
        1258,
        699,
        425
      ],
      [
        1259,
        689,
        428
      ],
      [
        1260,
        679,
        431
      ],
      [
        1261,
        669,
        435
      ],
      [
        1262,
        658,
        438
      ],
      [
        1263,
        649,
        443
      ],
      [
        1264,
        637,
        448
      ],
      [
        1265,
        626,
        453
      ],
      [
        1266,
        616,
        458
      ],
      [
        1267,
        604,
        464
      ],
      [
        1268,
        592,
        470
      ],
      [
        1269,
        582,
        478
      ],
      [
        1270,
        570,
        486
      ],
      [
        1271,
        557,
        494
      ],
      [
        1272,
        546,
        502
      ],
      [
        1273,
        535,
        511
      ],
      [
        1274,
        524,
        520
      ],
      [
        1275,
        510,
        529
      ],
      [
        1276,
        497,
        541
      ],
      [
        1277,
        485,
        552
      ],
      [
        1278,
        472,
        563
      ],
      [
        1279,
        458,
        576
      ],
      [
        1280,
        444,
        589
      ],
      [
        1281,
        431,
        601
      ],
      [
        1282,
        419,
        616
      ],
      [
        1283,
        405,
        631
      ],
      [
        1284,
        390,
        644
      ],
      [
        1285,
        378,
        661
      ],
      [
        1286,
        362,
        676
      ],
      [
        1287,
        348,
        695
      ],
      [
        1288,
        331,
        716
      ],
      [
        1289,
        316,
        735
      ],
      [
        1290,
        300,
        753
      ],
      [
        1291,
        286,
        771
      ],
      [
        1292,
        270,
        796
      ],
      [
        1293,
        257,
        800
      ],
      [
        1294,
        245,
        788
      ],
      [
        1295,
        232,
        778
      ],
      [
        1296,
        217,
        766
      ],
      [
        1297,
        205,
        756
      ],
      [
        1298,
        192,
        745
      ],
      [
        1299,
        180,
        736
      ],
      [
        1300,
        166,
        724
      ],
      [
        1301,
        151,
        717
      ],
      [
        1302,
        137,
        707
      ],
      [
        1303,
        123,
        700
      ],
      [
        1304,
        106,
        692
      ],
      [
        1305,
        91,
        682
      ],
      [
        1306,
        78,
        674
      ],
      [
        1307,
        62,
        667
      ],
      [
        1308,
        49,
        660
      ],
      [
        1309,
        33,
        654
      ],
      [
        1310,
        22,
        648
      ],
      [
        1311,
        13,
        644
      ],
      [
        1312,
        5,
        640
      ],
      [
        1318,
        808,
        541
      ]
    ],
    "attack_point": [
      1.3600405679513186,
      9.0
    ],
    "landing_point": [
      -0.2877609431743622,
      3.585444211959839
    ],
    "debug_output_path": "/workspace/heatmaps/match14/debug_clips/rally_006_debug.mp4",
    "set": 1
  },
```
</details>


<a id="ball-trajectory"></a>

## Ball Trajectory
A nice feature I've added is calculating the ball trajectory.
This feature is also kind of non trivial because its hard to determine at which point the ball went on the "attack" trajectory. 
The cool heuristic here is to use the net line calculation from before:
```python
# net line coords
net_line = np.array([
    [471, 530],
    [1441, 530]
], dtype=np.float32)

```
As a proxy for timing the net crossing point.
![Ball trajectory and net crossing point](../data/blog/images/track.png)
Because before the ball falls, it needs to first cross the net.
From here it's pretty simple, draw a trajectory from ground contact to net crossing point and transforming with the homography matrix and we get the simple ball trajectory!
<a id="testing"></a>

### Testing 
There is no fully reliable automatic metric for this stage yet, so evaluation is currently qualitative through visual inspection. It seems that It can get better with a direct relation to the ball detection model.


<a id="the-site"></a>

## The Site
The results are passed as JSON files to the site.
There isn't much to say about the site, its mostly vibe coded with Lovable. It did a very good job in my opinion. Letting us comfortably inspect videos, matches, and the heatmaps.


<a id="whats-next"></a>

## Whats next?
The whistle detection and rally segmentation can be seen as infrastructre for a bigger project.
With these the next possible steps are:
- Adding a camera on the other side of the court to map the other half too.
- Adding an orthogonal camera and obtain court depth
- Pose and Action detection
- Serve statistics
- Top down tactical view
The final goal would be having a full match statistics generator but the complexity and compute cost are obviously rough.
