# 1-3. Deployment

## [1] Key Challenges

2 main challenges

- 1) ML & statistical issues ( Concept & Data drift )
- 2) software engine issues

<br>

### a) Concept & Data drift

*ex) Speech Recognition*

Training data

- purchased data ( clean )

Test

- data from a few month

$\rightarrow$ the data have might changed!

<br>

2 kinds of drift

- Concept drift = relation of $X$ & $Y$ have changed

- Data drift = $X$ have changed

<br>

### b) software engine issues

checklist of questions

- 1) real-time vs batch
- 2) cloud vs edge/browser
- 3) computing resources
- 4) latency, throughput (QPS, query per second)
- 5) logging
- 6) security & privacy

<br>

### Summary

![figure2](/assets/img/mlops/img11.png)

Challenge 1) software engine issues

$\rightarrow$ in "deploying in production"

<br>

Challenge 2) concept & data drift

$\rightarrow$ in "monitoring & maintaining system"

<br>

## [2] Deployment patterns

### Common Deployment Cases

- 1) new product/capability
- 2) automation with AI ( or assistance )
- 3) replace previous ML systems

<br>

Key ideas

- 1) gradual ramp up 
- 2) roll back

<br>

Example : visual inspection )

### a) Shadow Mode deployment

- shadows humans' judgement & run parallel
- ML's output is NOT used

$\rightarrow$ purpose : **gather data of how the model is performing, by comparing with the human judgement**

![figure2](/assets/img/mlops/img12.png)

<br>

### b) Canary deployment

- roll out to **only small fraction** initially
- if OK ... ramp up to traffic "gradually"

<br>

### c) Blue green deployment

![figure2](/assets/img/mlops/img13.png)

Version

- BLUE = old version
- GREEN = new version

<br>

Router switches from BLUE $\rightarrow$ GREEN

- easy way to "roll back"
- do not have to be "at once"! ( gradually OK )

<br>

### "Degrees of automation"

( Left : manual ) ---------- ( Right : automatic ) 

![figure2](/assets/img/mlops/img14.png)

<br>

## [3] Monitoring

### a) Monitoring dashboard

![figure2](/assets/img/mlops/img15.png)

- set thresholds for alarms!

<br>

### b) metrics to check (ex)

1. Software metrics

   - memory, compute, latency, throughput, server load..

2. Input ($X$) metrics

   ( ex. speech recognition, defect image classification )

   - ex) avg input length, avg input volume
   - ex) \# of missing values, avg image brightness

3. Output ($Y$) metrics
   - ex) \# of null values, CTR ...

<br>

### c) Iterative Procedure

"ML modeling" & "deployment" is both an iterative procedure!

![figure2](/assets/img/mlops/img16.png)

<br>

### d) Model maintenance

may need to **retrain models!**

- manual retraining ( more common )
- automatic retraining

<br>

## [4] Pipeline monitoring

Many AI systems involves a **pipeline of multiple steps**.

<br>

ex) speech recognition

- Input : audio & Output : transcript. 
- implemented on mobile apps ( picture below )

![figure2](/assets/img/mlops/img17.png)

<br>

### Pipeline

Step 1) fed into VAD ( Voice Activity Detection )

- 1-1) check if anyone is speaking
- 1-2) looks at the long stream of audio &
  shortens the audio to just the part "where someone is talking"
- 1-3) send that to the (cloud) server

Step 2) perform speech recognition on the server

<br>

Thus, **change in step 1) may affect the final result!**

- ex ) change in way a new cell phone's microphone works
- ex ) change in user's profile & fed into rec sys

![figure2](/assets/img/mlops/img18.png)

$\rightarrow$  when working on ML pipelines, these effects can be **complex to keep track on**!

$\therefore$ **brainstorm metrics to monitor that can detect changes ( concept / data drift )**

- software metric / input metric / output metrics

<br>

### How quickly does the data change?

in general...

- User data = **slower drift**

- Enterprise data = **faster drift**