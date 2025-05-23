# CS221-Car-Tracking-solved

Download Here: [CS221 Car Tracking solved](https://jarviscodinghub.com/assignment/car-tracking-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

This assignment is a modified version of the Driverless Car assignment written by Chris Piech.

A study by the World Health Organization found that road accidents kill a shocking 1.24 million people a year worldwide. In response, there has been great interest in developing autonomous driving technology that can can drive with calculated precision and reduce this death toll. Building an autonomous driving system is an incredibly complex endeavor. In this assignment, you will focus on the sensing system, which allows us to track other cars based on noisy sensor readings.

Getting started. Let’s start by trying to drive manually:

python drive.py -l lombard -i none
You can steer by either using the arrow keys or ‘w’, ‘a’, and ‘d’. The up key and ‘w’ accelerates your car forward, the left key and ‘a’ turns the steering wheel to the left, and the right key and ‘d’ turns the steering wheel to the right. Note that you cannot reverse the car or turn in place. Quit by pressing ‘q’. Your goal is to drive from the start to finish (the green box) without getting in an accident. How well can you do on crooked Lombard street without knowing the location of other cars? Don’t worry if you aren’t very good; the teaching staff were only able to get to the finish line 4/10 times. An accident rate of 60% is pretty abysmal, which is why we’re going to build an AI to do this.

Flags for python drive.py:

-a: Enable autonomous driving (as opposed to manual).
-i : Use none, exactInference, particleFilter to (approximately) compute the belief distributions.
-l

: Use this map (e.g. small or lombard). Defaults to small.
-d: Debug by showing all the cars on the map.
-p: All other cars remain parked (so that they don’t move).
Modeling car locations. We assume that the world is a two-dimensional rectangular grid on which your car and K other cars reside. At each time step t, your car gets a noisy estimate of the distance to each of the cars. As a simplifying assumption, we assume that each of the K other cars moves independently and that the noise in sensor readings for each car is also independent. Therefore, in the following, we will reason about each car independently (notationally, we will assume there is just one other car).
At each time step t, let Ct∈R2 be a pair of coordinates representing the actual location of the single other car (which is unobserved). We assume there is a local conditional distribution p(ct∣ct−1) which governs the car’s movement. Let at∈R2 be your car’s position, which you observe and also control. To minimize costs, we use a simple sensing system based on a microphone. The microphone provides us with Dt, which is a Gaussian random variable with mean equal to the true distance between your car and the other car and variance σ2 (in the code, σ is Const.SONAR_STD, which is about two-thirds the length of a car). In symbols,

Dt∼N(∥at−Ct∥,σ2).
For example, if your car is at at=(1,3) and the other car is at Ct=(4,7), then the actual distance is 5 and Dt might be 4.6 or 5.2, etc. Use util.pdf(mean, std, value) to compute the probability density function (PDF) of a Gaussian with given mean and standard deviation, evaluated at value. Note that the PDF does not return a probability — densities can exceed 1 — but for the purposes of this assignment, you can get away with treating it like a probability. The Gaussian probability density function for the noisy distance observation Dt, which is centered around your distance to the car μ=∥at−Ct∥, is shown in the following figure:

Your job is to implement a car tracker that (approximately) computes the posterior distribution P(Ct∣D1=d1,…,Dt=dt) (your beliefs of where the other car is) and update it for each t=1,2,…. We will take care of using this information to actual drive the car (i.e., set at to avoid a collision with ct), so you don’t have to worry about that part.

To simplify things, we will discretize the world into tiles represented by (row, col) pairs, where 0 <= row < numRows and 0 <= col < numCols. For each tile, we store a probability distribution whose values can be accessed by self.belief.getProb(row, col). To convert from a tile to a location, use util.rowToY(row) and util.colToX(col).

Here’s an overview of the assignment components:

Problem 1 (written) will give you some practice with probabilistic inference on a simple Bayesian network.
In Problems 2 and 3 (code), you will implement ExactInference, which computes a full probability distribution of another car’s location over tiles (row, col).
In Problem 4 (code), you will implement ParticleFilter, which works with particle-based representation of this same distribution.
Problem 5 (written) gives you a chance to extend your probability analyses to a slightly more realistic scenario where there are multiple other cars and we can’t automatically distinguish between them.
A few important notes before we get started:

Past experience suggests that this will be one of the most conceptually challenging assignments of the quarter for many students. Please start early, especially if you’re low on late days!
We strongly recommend that you attend/watch the lectures on Bayesian networks and HMMs before getting started, and keep the slides handy for reference while you’re working.
The code portions of this assignment are short and straightforward — no more than about 30 lines in total — but only if your understanding of the probability concepts is clear! (If not, see the previous point.)
As a notational reminder: we use the lowercase expressions p(x) or p(x|y) for local conditional probability distributions, which are defined by the Bayesian network. We use the uppercase expressions P(X=x) or P(X=x|Y=y) for joint and posterior probability distributions, which are not pre-defined in the Bayesian network but can be computed by probabilistic inference. Please review the lecture slides for more details.
Problem 1: Bayesian network basics
First, let us look at a simplified version of the car tracking problem. For this problem only, let Ct∈{0,1} be the actual location of the car we wish to observe at time step t∈{1,2,3}. Let Dt∈{0,1} be a sensor reading for the location of that car measured at time t. Here’s what the Bayesian network (it’s an HMM, in fact) looks like:

The distribution over the initial car distribution is uniform; that is, for each value c1∈{0,1}:
p(c1)=0.5.
The following local conditional distribution governs the movement of the car (with probability ϵ, the car moves). For each t∈{2,3}:
p(ct∣ct−1)={ϵ1−ϵif ct≠ct−1if ct=ct−1.
The following local conditional distribution governs the noise in the sensor reading (with probability η, the sensor reports the wrong position). For each t∈{1,2,3}:
p(dt∣ct)={η1−ηif dt≠ctif dt=ct.
Below, you will be asked to find the posterior distribution for the car’s position at the second time step (C2) given different sensor readings.

Important: For the following computations, try to follow the general strategy described in lecture (marginalize non-ancestral variables, condition, and perform variable elimination). Try to delay normalization until the very end. You’ll get more insight than trying to chug through lots of equations.

[2 points] Suppose we have a sensor reading for the second timestep, D2=0. Compute the posterior distribution P(C2=1∣D2=0). We encourage you to draw out the (factor) graph.
[2 points] Suppose a time step has elapsed and we got another sensor reading, D3=1, but we are still interested in C2. Compute the posterior distribution P(C2=1∣D2=0,D3=1). The resulting expression might be moderately complex. We encourage you to draw out the (factor) graph.
[3 points] Suppose ϵ=0.1 and η=0.2.
i. Compute and compare the probabilities P(C2=1∣D2=0) and P(C2=1∣D2=0,D3=1). Give numbers, round your answer to 4 significant digits.

ii. How did adding the second sensor reading D3=1 change the result? Explain your intuition for why this change makes sense in terms of the car positions and associated sensor observations.

iii. What would you have to set ϵ while keeping η=0.2 so that P(C2=1∣D2=0)=P(C2=1∣D2=0,D3=1)? Explain your intuition in terms of the car positions with respect to the observations.

Problem 2: Emission probabilities
In this problem, we assume that the other car is stationary (e.g., Ct=Ct−1 for all time steps t). You will implement a function observe that upon observing a new distance measurement Dt=dt updates the current posterior probability from
P(Ct∣D1=d1,…,Dt−1=dt−1)
to
P(Ct∣D1=d1,…,Dt=dt)∝P(Ct∣D1=d1,…,Dt−1=dt−1)p(dt∣ct),
where we have multiplied in the emission probabilities p(dt∣ct) described earlier. The current posterior probability is stored as self.belief in ExactInference.

[7 points] Fill in the observe method in the ExactInference class of submission.py. This method should modify self.belief in place to update the posterior probability of each tile given the observed noisy distance. After you’re done, you should be able to find the stationary car by driving around it (-p means cars don’t move):
Notes:

You can start driving with exact inference now.
python drive.py -a -p -d -k 1 -i exactInference
You can also turn off -a to drive manually.
It’s generally best to run drive.py on your local machine, but if you do decide to run it on cardinal/rice instead, please ssh into the farmshare machines with either the -X or -Y option in order to get the graphical interface; otherwise, you will get some display error message. Note: do expect this graphical interface to be a bit slow… drive.py is not used for grading, but is just there for you to visualize and enjoy the game!
Read through the code in utils.py for the Belief class before you get started… you’ll need to use this class for several of the code tasks in this assignment.
Remember to normalize the posterior probability after you update it. (There’s a useful function for this in utils.py).
On the small map, the autonomous driver will sometimes drive in circles around the middle block before heading for the target area. In general, don’t worry too much about the precise path the car takes. Instead, focus on whether your car tracker correctly infers the location of other cars.
Don’t worry if your car crashes once in a while! Accidents do happen, whether you are human or AI. However, even if there was an accident, your driver should have been aware that there was a high probability that another car was in the area.
Problem 3: Transition probabilities
Now, let’s consider the case where the other car is moving according to transition probabilities p(ct+1∣ct). We have provided the transition probabilities for you in self.transProb. Specifically, self.transProb[(oldTile, newTile)] is the probability of the other car being in newTile at time step t+1 given that it was in oldTile at time step t.

In this part, you will implement a function elapseTime that updates the posterior probability about the location of the car at a current time t
P(Ct=ct∣D1=d1,…,Dt=dt)
to the next time step t+1 conditioned on the same evidence, via the recurrence:
P(Ct+1=ct+1∣D1=d1,…,Dt=dt)∝∑ctP(Ct=ct∣D1=d1,…,Dt=dt)p(ct+1∣ct).
Again, the posterior probability is stored as self.belief in ExactInference.

[7 points] Finish ExactInference by implementing the elapseTime method. When you are all done, you should be able to track a moving car well enough to drive autonomously:
python drive.py -a -d -k 1 -i exactInference
Notes:

You can also drive autonomously in the presence of more than one car:
python drive.py -a -d -k 3 -i exactInference
You can also drive down Lombard:
python drive.py -a -d -k 3 -i exactInference -l lombard
On Lombard, the autonomous driver may attempt to drive up and down the street before heading towards the target area. Again, focus on the car tracking component, instead of the actual driving.
Problem 4: Particle filtering
Though exact inference works well for the small maps, it wastes a lot of effort computing probabilities for every available tile, even for tiles that are unlikely to have a car on them. We can solve this problem using a particle filter. Updates to the particle filter have complexity that’s linear in the number of particles, rather than linear in the number of tiles.

In this problem, you’ll implement two short but important methods for the ParticleFilter class in submission.py. When you’re finished, your code should be able to track cars nearly as effectively as it does with exact inference.

[18 points] Some of the code has been provided for you. For example, the particles have already been initialized randomly. You need to fill in the observe and elapseTime functions. These should modify self.particles, which is a map from tiles (row, col) to the number of particles existing at that tile, and self.belief, which needs to be updated each time you re-sample the particle locations.
You should use the same transition probabilities as in exact inference. The belief distribution generated by a particle filter is expected to look noisier compared to the one obtained by exact inference.

python drive.py -a -i particleFilter -l lombard
To debug, you might want to start with the parked car flag (-p) and the display car flag (-d).
Note: The random number generator inside util.weightedRandomChoice behaves differently on different systems’ versions of Python (e.g., Unix and Windows versions of Python). Please test this question (run grader.py) on rice. When copying files to rice, make sure you copy the entire folder using scp with the recursive option -r.

Problem 5: Which car is it?
So far, we have assumed that we have a distinct noisy distance reading for each car, but in reality, our microphone would just pick up an undistinguished set of these signals, and we wouldn’t know which distance reading corresponds to which car. First, let’s extend the notation from before: let Cti∈R2 be the location of the i-th car at the time step t, for i=1,…,K and t=1,…,T. Recall that all the cars move independently according to the transition dynamics as before.

Let Dti∈R be the noisy distance measurement of the i-th car, which is now not directly observed. Instead, we observe the set of distances Dt={Dt1,…,DtK}. (For simplicity, we’ll assume that all distances are distinct values.) Alternatively, you can think of Et=(Et1,…,EtK) as a list which is a uniformly random permutation of the noisy distances (Dt1,…,DtK). For example, suppose K=2 and T=2. Before, we might have gotten distance readings of 1 and 2 for the first car and 3 and 4 for the second car at time steps 1 and 2, respectively. Now, our sensor readings would be permutations of {1,3} (at time step 1) and {2,4} (at time step 2). Thus, even if we knew the second car was distance 3 away at time t=1, we wouldn’t know if it moved further away (to distance 4) or closer (to distance 2) at time t=2.

[5 points] Suppose we have K=2 cars and one time step T=1. Write an expression for the conditional distribution P(C11,C12∣E1=e1) as a function of the PDF of a Gaussian pN(v;μ,σ2) and the prior probability p(c11) and p(c12) over car locations. Your final answer should not contain variables d11, d12.
Remember that pN(v;μ,σ2) is the probability of a random variable, v, in a Gaussian distribution with mean μ and standard deviation σ.

Hint: for K=1, the answer would be
P(C11=c11∣E1=e1)∝p(c11)pN(e11;∥a1−c11∥,σ2).
where at is the position of the car at time t. You might find it useful to draw the Bayesian network and think about the distribution of Et given Dt1,…,DtK.

[4 points] Assuming the prior p(c1i) is the same for all i, show that the number of assignments for all K cars (c11,…,c1K) that obtain the maximum value of P(C11=c11,…,C1K=c1K∣E1=e1) is at least K!.
You can also assume that the car locations that maximize the probability above are unique (C1i≠c1j for all i≠j).

Note: you don’t need to produce a complicated proof for this question. It is acceptable to provide a clear explanation based on your intuitive understanding of the scenario.

[2 points] For general K, what is the treewidth corresponding to the posterior distribution over all K car locations at all T time steps conditioned on all the sensor readings:
P(C11=c11,…,C1K=c1K,…,CT1=cT1,…,CTK=cTK∣E1=e1,…,ET=eT)?
Briefly justify your answer.
[6 points] (extra credit) Now suppose you change your sensors so that at each time step t, they return the list of exact positions of the K cars, but shifted (with wrap around) by a random amount. For example, if the true car positions at time step 1 are c11=1,c12=3,c13=8,c14=5, then e1 would be [1,3,8,5], [3,8,5,1], [8,5,1,3], or [5,1,3,8], each with probability 1/4. Describe an efficient algorithm for computing p(cti∣e1,…,eT) for any time step t and car i. Your algorithm should not be exponential in K or T.
