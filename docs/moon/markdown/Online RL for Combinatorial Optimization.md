# **Real-Time Algorithmic Adaptation: Online Reinforcement Learning for Test-Time Augmentation of Combinatorial Optimization Meta-Heuristics**

The landscape of combinatorial optimization (CO) has historically been dominated by handcrafted meta-heuristics and hyper-heuristics designed through a combination of mathematical intuition and empirical trial-and-error.1 However, the inherent complexity and diversity of real-world problem instances often render static algorithmic configurations suboptimal.3 A significant paradigm shift is currently underway, characterized by the integration of machine learning—specifically reinforcement learning (RL) techniques—into the search process to enable dynamic, test-time adaptation.5 Unlike neural combinatorial optimization (NCO) approaches that rely on expensive offline training and often struggle with generalization, online reinforcement learning methods like Multi-Armed Bandits (MAB), Contextual Multi-Armed Bandits (CMAB), and Temporal Difference (TD) learning offer a mechanism for "mid-inference" adaptation.5 These methods allow an optimization algorithm to learn the most effective local search operators and tune continuous hyper-parameters on the fly for the specific instance at hand, effectively bridging the gap between general-purpose solvers and instance-specific expertise.8

## **The Mechanics of Online Adaptation in Heuristic Search**

The fundamental challenge in combinatorial optimization is navigating a vast and often rugged search space to find high-quality solutions within a limited computational budget.12 Traditional meta-heuristics like Large Neighborhood Search (LNS) or Differential Evolution (DE) employ fixed strategies for operator selection and parameter settings.4 The integration of online RL re-frames this navigation as a sequential decision-making process where an autonomous agent selects actions—such as which neighborhood operator to apply—based on observed feedback from the environment.2

### **Adaptive Operator Selection via Multi-Armed Bandits**

At any given iteration of a search process, multiple potential operators (arms) are available. Adaptive Operator Selection (AOS) utilizes the Multi-Armed Bandit framework to manage the tension between exploration—trying underused operators to discover their current utility—and exploitation—using operators that have historically yielded high improvements.18 This is essentially a "context-free" learning scenario where the agent maintains a running estimate of each operator's reward distribution based solely on its past performance.20

The standard Upper Confidence Bound (UCB1) algorithm is frequently employed in this context because it provides a rigorous mathematical solution to the exploration-exploitation dilemma.18 It assigns a value to each operator ![][image1] that balances its empirical average reward ![][image2] with an uncertainty term.18 The selection rule ensures that as the total number of iterations ![][image3] grows, the algorithm focuses on the most successful operators while periodically "checking in" on others to see if their utility has changed.18

| Selection Policy | Mathematical Foundation | Primary Advantage | Limitation |
| :---- | :---- | :---- | :---- |
| **![][image4]\-greedy** | Bernoulli Sampling | Consistent top performer across diverse problems; extremely low overhead.5 | Can be too explorative if ![][image4] is not properly decayed.20 |
| **UCB1** | Concentration Inequalities | Theoretically optimal regret bounds in static environments.18 | Can be slow to adapt in non-stationary optimization landscapes without resets.18 |
| **Softmax** | Boltzmann Distribution | Probabilistic selection based on relative performance values.23 | Highly sensitive to the temperature parameter ![][image5].23 |
| **Thompson Sampling** | Bayesian Posterior | Naturally handles uncertainty through probabilistic updates using Beta distributions.24 | Performance can lag if the reward frequency is extremely low.24 |
| **EXP3** | Weight-based Sampling | Robust in adversarial settings where reward distributions are not well-behaved.26 | Generally slower convergence than UCB-based methods in standard settings.26 |

### **Handling Non-Stationarity and the Dynamic Nature of Search**

A critical insight into the optimization process is that it is inherently non-stationary.18 An operator that is highly effective for "diversification" (exploring new areas of the search space) at the beginning of a run may be completely ineffective for "intensification" (fine-tuning a local optimum) later on.12 Standard MAB algorithms, which assume static reward distributions, often fail to adapt quickly enough to these shifts.18

To address this, the Dynamic Multi-Armed Bandit (D-MAB) framework hybridizes UCB1 with change detection mechanisms like the Page-Hinkley (PH) test.18 The PH test monitors the cumulative difference between observed rewards and their average, signaling a "restart" of the bandit when the difference ![][image6] exceeds a threshold ![][image7].18 This "cold restart" clears the accumulated statistics ![][image8] and ![][image2], allowing the algorithm to re-evaluate all operators under the new search conditions without the "inertia" of outdated historical data.18 This mechanism is vital for maintaining the agility of the search agent as it transitions through different phases of the optimization trajectory.18

## **Case Study: The Balans Meta-Solver for Mixed-Integer Programming**

A state-of-the-art implementation of these principles is found in the *Balans* meta-solver, designed to accelerate Mixed-Integer Programming (MIP) solving without offline training.8 *Balans* operates on top of existing MIP solvers like SCIP or Gurobi, utilizing an Adaptive Large Neighborhood Search (ALNS) framework.8 In this system, each "arm" of the multi-armed bandit is defined as a pair consisting of a "destroy" operator and a "repair" operator.8

The "destroy" operators in *Balans* include strategies such as:

* **Random Objective:** Introduces randomness in the objective function to force the solver out of local minima.9
* **RENS (Relaxation Enforced Neighborhood Search):** Focuses on exploring feasible roundings of an optimal LP relaxation.9
* **RINS (Relaxation Induced Neighborhood Search):** Fixes variables that have the same value in both the current incumbent and the LP relaxation.9

The "repair" operator is typically a call to the underlying MIP solver to re-optimize the sub-MIP created by the destroy operation.8 The learning agent's goal is to maximize the cumulative reward, which *Balans* computes based on the outcome of the acceptance criterion.8

| Outcome of Neighborhood Application | Reward Signal (Indicative Logic) | Impact on Bandit Selection |
| :---- | :---- | :---- |
| **New Best Solution Found** | Highest Reward (+1.0 or higher) | Increases probability of selecting this operator pair significantly.8 |
| **Improved Solution Found** | Moderate Reward | Reinforces the operator pair as a useful local improver.8 |
| **Worse but Accepted Solution** | Low Reward | Maintains some interest in the operator for diversification.8 |
| **Rejected Solution** | Zero or Negative Reward | Penalizes the operator, leading to less frequent selection in the near term.8 |

The performance of *Balans* underscores a fundamental truth in online learning for optimization: the solver rarely depends on a single "best" operator.8 Instead, it achieves superior performance by exploring and sequencing "weaker" neighborhoods that are contextually effective during specific segments of the search.8 This approach eliminates the reliance on expensive offline data collection and avoids the generalization gaps that plague supervised learning models in the MIP domain.9

## **Contextual Multi-Armed Bandits and State-Aware Selection**

While standard MAB methods treat each iteration as an independent trial, Contextual Multi-Armed Bandits (CMAB) acknowledge that the optimal action often depends on the specific state of the solution or the problem instance.21 In this framework, the agent observes a "context" vector ![][image9] before choosing an action.21 This context allows the model to tailor its strategy to specific scenarios, moving closer to "one-to-one personalization" of the search strategy.21

### **Feature Engineering for Optimization Contexts**

The efficacy of CMAB depends on the quality and relevance of the contextual features.21 In the context of AOS, these features must capture the dynamics of the search process while remaining domain-independent to ensure generality.13

Research into feature-based input spaces has identified several critical categories of optimization context 7:

* **Population Diversity Features:** Metrics like solution diversity (![][image10]) and population diameter (![][image11]) indicate whether the search is converging or still broadly exploring the solution space.32
* **Fitness Landscape Features:** Population fitness deviation (![][image12]) and the fitness gap between parents and offspring (![][image13]) provide insights into the ruggedness of the current region.32
* **Search Trajectory Features:** The proportion of new best children (![][image14]) and convergence velocity (![][image15]) help the agent determine if the current operator portfolio is yielding diminishing returns.32
* **Distance-based Features:** Euclidean or Manhattan distances between global best solutions and parent solutions (![][image16]) help characterize the spatial distribution of high-quality optima.32

### **Gaussian Processes for Contextual Smoothing**

When dealing with high-dimensional context spaces, learning an independent reward distribution for every possible context is impossible.36 CMAB algorithms like KernelUCB or GP-UCB address this by imposing "smoothness" on the reward function via Gaussian Processes (GPs).36 GPs allow the agent to generalize from one context to another; if an operator works well in a state ![][image9], it is likely to work well in a similar state ![][image17].31

The GP-UCB algorithm maintains a posterior mean ![][image18] and variance ![][image19] for each action ![][image20].36 The selection is governed by an acquisition function that incorporates the "maximum information gain" ![][image21], effectively prioritizing actions with high predicted rewards or high uncertainty.36 This is particularly powerful for "volatile" arms where the set of available neighborhoods might change depending on the current solution's constraints.36

## **Temporal Difference Learning and Heuristic Sequencing**

Unlike bandit methods, which are inherently "myopic" and evaluate operators based on immediate feedback, Temporal Difference (TD) learning methods like Q-learning and Sarsa focus on the long-term cumulative reward.38 This is vital for hyper-heuristics that need to learn *sequences* of heuristics.12 For instance, a "ruin" operator might initially degrade solution quality (negative immediate reward) but set the stage for a "local search" operator to find a much better optimum than was previously reachable.13

### **Q-Learning for Hyper-Heuristics (QL-HH)**

In a Q-learning based hyper-heuristic, the agent maintains a Q-table where each entry ![][image22] represents the estimated value of applying heuristic ![][image20] in state ![][image23].12 The state ![][image23] in these systems is often a quantized representation of the search progress, such as whether the last move improved the solution or not.12

The reward representation in these systems is critical.12 In the PSO-GWO-Q hyper-heuristic, for example, a binary reward table is used:

* **Reward \+1:** If the algorithm improved the solution in the current iteration.12
* **Reward \-1:** If no improvement was found.12

The Q-values are updated using the Bellman equation, which allows the agent to "bootstrap" its estimates from future states.38 To account for non-stationarity, a learning rate ![][image24] and a discount factor ![][image21] are tuned to balance immediate gains against long-term potential.12

### **True Online TD Learning and Efficiency**

Recent advancements in "True Online" TD(![][image7]) methods have improved the stability of these learning agents.38 Traditional TD methods only approximate the "forward view" of rewards (the actual future return) unless step sizes are infinitesimally small.38 True Online TD(![][image7]) maintains an exact equivalence with the forward view at all times by utilizing eligibility traces that are updated incrementally.38 This provides the search agent with more accurate credit assignment, ensuring that operators responsible for opening up new, promising regions of the search space are correctly rewarded even if their immediate impact was negative.38

## **Adaptive Hyper-Parameter Tuning Mid-Inference**

Beyond the selection of discrete operators, online reinforcement learning is increasingly used to control the continuous parameters that define meta-heuristic behavior.3 Parameters such as the step size in CMA-ES, or the mutation scaling factor ![][image25] and crossover rate ![][image26] in Differential Evolution, have a massive impact on search trajectory.3

### **Cluster-Based Parameter Adaptation (CPA)**

The CPA method is a representative online parameter-tuning approach for population-based meta-heuristics.42 It treats the parameter configuration space as a ![][image27]\-dimensional search space and uses unsupervised learning to identify "promising areas".42

The CPA process follows a cyclical three-stage mechanism:

1. **Stage 0 (Initialization):** Control parameters are randomly generated within predefined bounds.42
2. **Guided Clustering:** The system applies K-means clustering to classify the parameters associated with successful search agents.42
3. **Proportional Sampling:** New parameter sets are generated around the centroids of these clusters.42 Larger, more successful clusters receive more samples (exploitation), while smaller clusters promote continued exploration of the parameter space.42
4. **Directional Sampling and Evaporation:** New candidates are drawn using random unit vectors, while an "evaporation" model gradually decreases the probability of sampling around old, stagnant centroids.42

This online adaptive scheme allows the algorithm to discover the "sweet spot" for its parameters on a per-instance basis, outperforming static handcrafted rules that may be biased toward specific benchmark sets.3

### **RL as an Optimal Control Problem**

For some researchers, the task of adapting all aspects of an optimization algorithm—including momentum, preconditioner, and weight decay—is viewed as a problem of meta-optimization or online control.43 This framework uses "meta-regret" as a metric, comparing the performance of the adaptive algorithm against the best optimization method available in hindsight.43 By applying convex relaxation techniques and non-stochastic control theory, these methods provide sublinear regret guarantees, ensuring that the learning agent will converge to a strategy as effective as the best possible fixed parameter set.43

## **Analysis of State Representation and Information Density**

The "Domain Barrier" in frameworks like HyFlex prevents hyper-heuristics from seeing the internal problem representation, such as the specific locations in a Traveling Salesman Problem or the constraints in a Bin Packing instance.23 This forces the RL agent to rely on a limited, often "large-state" set of features derived from the search history.33

### **The LAST-RL Framework**

The LAST-RL (Large-State Reinforcement Learning) hyper-heuristic provides a template for highly effective state representation in domain-independent optimization.33 Rather than using a single state variable, LAST-RL employs 12,447 features derived from the trajectory of solution changes.44

Key trajectory features in LAST-RL include:

* **relImprH:** The relative number of improvements in the last ![][image28] heuristic applications, capturing recent intensification success.44
* **rel0H:** The relative number of moves that resulted in no change to the objective, indicating potential stagnation or a plateau in the landscape.44
* **Chain Success Rates:** The historical success of specific sequences (e.g., Crossover followed by Local Search).33

By utilizing such a rich feature set, the RL agent can differentiate between different stages of the search and select operators that are likely to break out of specific types of local optima.33

## **Benchmarking Generality: HyFlex and CHeSC 2011**

The Cross-Domain Heuristic Search Challenge (CHeSC 2011\) remains the most rigorous benchmark for online adaptive hyper-heuristics.34 The challenge requires a single high-level strategy to manage different sets of low-level heuristics across multiple domains.46

### **RL Performance Trends in Cross-Domain Search**

Initial RL-based submissions to CHeSC 2011, such as those by Di Gaspero and Urli, struggled to compete with complex handcrafted adaptive heuristics like GIHH.47 This was largely due to limited state representations and the computational overhead of the learning updates.47 However, the field has evolved, and contemporary RL-based methods that incorporate trajectory-based features (like LAST-RL) or sophisticated exploration distributions based on Iterated Local Search now consistently place in the top three of international benchmarks.33

| Benchmark Property | HyFlex/CHeSC Requirement | RL Adaptation Strategy |
| :---- | :---- | :---- |
| **Domain Barrier** | No problem-specific data.34 | Use of objective function trajectory and search dynamics as features.33 |
| **Generality** | Must perform well on Personnel Scheduling, SAT, Flow Shop, etc..35 | RL agents learn domain-specific "good moves" online without prior knowledge.7 |
| **Limited Time** | Fixed runtime budget for all algorithms.46 | Preference for low-overhead methods like MAB and ![][image4]\-greedy over Deep RL.5 |
| **Heuristic Diversity** | Must manage Mutation, Ruin/Recreate, and Local Search.34 | Adaptive operator selection via bandits to balance diversification and intensification.8 |

## **Comparative Analysis: Online Learning vs. Deep Reinforcement Learning**

A significant debate exists regarding the use of Deep Reinforcement Learning (DRL) versus simpler online methods like bandits for test-time adaptation.5 While DRL models (e.g., PPO, DDQN) can learn far more complex policies, their computational cost is often prohibitive in the context of heuristic search.5

Experimental evaluations show that:

* **Computational Overhead:** The forward and backward passes of a neural network in DRL consume significant time that could otherwise be used for additional solution evaluations.5
* **Stability:** DRL methods are highly sensitive to reward function design. Large variations in cost, often due to constraint violation penalties, can destabilize the learning signal.5
* **Runtime Advantage:** DRL-based hyper-heuristics only become competitive when the search runtime is substantially longer than what is typically allowed in standard competitions.5
* **Efficiency of Simple Policies:** Policies like ![][image4]\-greedy consistently rank among the best performers because their near-zero overhead allows for thousands more iterations, often finding better solutions through sheer volume of exploration.5

## **Second-Order Insights: The Impact of "Forgetful" Learning**

One of the most profound insights from the study of online RL in optimization is the value of forgetting.18 Because the "optimal" operator changes throughout the search, an agent that remembers too much of its early history becomes "heavy" and slow to react to new opportunities.18

This has led to the development of three "strategic transformations" for hyper-heuristics:

1. **Solution Acceptance Control:** Dynamically adjusting how frequently "worse" solutions are accepted to escape local optima.49
2. **Repeated Operator Application:** Learning when to apply the same successful operator multiple times in a row before re-evaluating.49
3. **Perturbation Intensity Tuning:** Automatically scaling the "size" of a mutation or ruin operator based on current search success.49

When these principles are accompanied by even a trivial selection mechanism (like random selection), they can outperform complex offline-trained models, highlighting that the *dynamics* of the search are often more important than the *sophistication* of the selector.49

## **Future Directions: Preference Optimization and LLM-Guided Solvers**

The field is currently moving toward more nuanced reward signals and high-level strategy generation.50

### **Preference Optimization**

Standard RL struggles with quantitative rewards in combinatorial spaces because the magnitudes of cost improvements vary wildly.5 Preference Optimization addresses this by transforming quantitative signals into qualitative preferences (e.g., Solution A is better than Solution B).50 By modeling the "superiority" among sampled solutions rather than the absolute reward, algorithms can avoid the intractable computations of traditional entropy-regularized RL while escaping local optima more effectively.50

### **LLMs and Meta-Optimization**

The use of Large Language Models (LLMs) to refine meta-optimizers represents the latest frontier.51 These systems, such as the "Meta-Optimization of Heuristics" (MoH) framework, use LLMs to iteratively discover and construction new heuristic optimizers.51 This moves beyond the selection of *existing* operators and into the *autonomous generation* of new search strategies, further reducing the reliance on human design.51

## **Synthesis and Conclusion**

The integration of online reinforcement learning methods into combinatorial optimization meta-heuristics has fundamental transformed the way solvers navigate complex search landscapes. By shifting from static, handcrafted designs to autonomous, adaptive agents, researchers have unlocked a level of generality and robustness that was previously unattainable.

Key takeaways from the current state of the art include:

* **The Power of Simplicity:** In many test-time scenarios, low-overhead methods like ![][image4]\-greedy and UCB bandits outperform Deep RL due to the high computational cost of neural network inference.5
* **Importance of Change Detection:** In the non-stationary environment of optimization, mechanisms like the Page-Hinkley test are essential for allowing the search agent to "re-learn" the best strategy as the search transitions from exploration to intensification.18
* **Trajectory-based Context:** Effective test-time adaptation requires a rich state representation that goes beyond the current cost to include the "Large State" trajectory of recent successes and failures.33
* **Solver Agnosticism:** Methods like *Balans* show that online learning can be integrated into heavy-duty solvers like MIP solvers, providing performance gains by simply intelligently scheduling existing heuristics.8

Ultimately, the goal of these online methods is to raise the level of generality at which optimization methodologies can operate. By removing the need for extensive offline training and human-expert tuning, these systems democratize high-performance optimization, allowing for the rapid and robust solution of complex, real-world problems as they arrive. The future of the field lies in the continued synthesis of reinforcement learning and traditional mathematical optimization, creating solvers that are not just "fast," but "intelligent" in their approach to the unknown landscapes of combinatorial search.

#### **Works cited**

1. Enabling Population-Based Architectures for Neural Combinatorial Optimization \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2601.08696v1](https://arxiv.org/html/2601.08696v1)
2. Automated Design of Metaheuristics Using Reinforcement Learning within a Novel General Search Framework \- University of Nottingham, accessed March 4, 2026, [https://people.cs.nott.ac.uk/pszrq/files/TEVC22-GSF.pdf](https://people.cs.nott.ac.uk/pszrq/files/TEVC22-GSF.pdf)
3. Hyper-Heuristic Approach for Tuning Parameter Adaptation in Differential Evolution \- MDPI, accessed March 4, 2026, [https://www.mdpi.com/2075-1680/13/1/59](https://www.mdpi.com/2075-1680/13/1/59)
4. Reinforcement learning based adaptive metaheuristics \- IRIS, accessed March 4, 2026, [https://iris.unitn.it/retrieve/f20b0245-edc0-41bd-95a6-68aa2420d0c8/3520304.3533983.pdf](https://iris.unitn.it/retrieve/f20b0245-edc0-41bd-95a6-68aa2420d0c8/3520304.3533983.pdf)
5. Reinforcement Learning Methods for Neighborhood ... \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/abs/2601.07948](https://arxiv.org/abs/2601.07948)
6. Machine Learning for Enhancing Metaheuristics in Global Optimization: A Comprehensive Review \- MDPI, accessed March 4, 2026, [https://www.mdpi.com/2227-7390/13/18/2909](https://www.mdpi.com/2227-7390/13/18/2909)
7. Adaptive Operator Selection with Reinforcement Learning | Request PDF \- ResearchGate, accessed March 4, 2026, [https://www.researchgate.net/publication/355323727\_Adaptive\_Operator\_Selection\_with\_Reinforcement\_Learning](https://www.researchgate.net/publication/355323727_Adaptive_Operator_Selection_with_Reinforcement_Learning)
8. Balans: Multi-Armed Bandits-based Adaptive Large Neighborhood Search for Mixed-Integer Programming Problems \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2412.14382v2](https://arxiv.org/html/2412.14382v2)
9. \[Literature Review\] Balans: Multi-Armed Bandits-based Adaptive Large Neighborhood Search for Mixed-Integer Programming Problem, accessed March 4, 2026, [https://www.themoonlight.io/en/review/balans-multi-armed-bandits-based-adaptive-large-neighborhood-search-for-mixed-integer-programming-problem](https://www.themoonlight.io/en/review/balans-multi-armed-bandits-based-adaptive-large-neighborhood-search-for-mixed-integer-programming-problem)
10. Balans: Multi-Armed Bandits-based Adaptive Large Neighborhood Search for Mixed-Integer Programming Problems \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2412.14382v3](https://arxiv.org/html/2412.14382v3)
11. Balans: Multi-Armed Bandits-based Adaptive Large Neighborhood Search for Mixed-Integer Programming Problem \- ResearchGate, accessed March 4, 2026, [https://www.researchgate.net/publication/387263624\_Balans\_Multi-Armed\_Bandits-based\_Adaptive\_Large\_Neighborhood\_Search\_for\_Mixed-Integer\_Programming\_Problem](https://www.researchgate.net/publication/387263624_Balans_Multi-Armed_Bandits-based_Adaptive_Large_Neighborhood_Search_for_Mixed-Integer_Programming_Problem)
12. Coordinating Some Heuristics Using Q-learning for the Class of ..., accessed March 4, 2026, [https://www.worldscientific.com/doi/full/10.1142/S2196888825500198](https://www.worldscientific.com/doi/full/10.1142/S2196888825500198)
13. A Deep Reinforcement Learning Based Hyper-heuristic for Combinatorial Optimisation with Uncertainties \- University of Nottingham, accessed March 4, 2026, [https://people.cs.nott.ac.uk/pszrq/files/EJOR21-drl-hh.pdf](https://people.cs.nott.ac.uk/pszrq/files/EJOR21-drl-hh.pdf)
14. A review of reinforcement learning based hyper-heuristics \- PMC \- NIH, accessed March 4, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11232579/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11232579/)
15. Balans: Multi-Armed Bandits-based Adaptive Large Neighborhood Search for Mixed-Integer Programming Problems \- IJCAI, accessed March 4, 2026, [https://www.ijcai.org/proceedings/2025/0286.pdf](https://www.ijcai.org/proceedings/2025/0286.pdf)
16. Balans: Multi-Armed Bandits-based Adaptive Large Neighborhood Search for Mixed-Integer Programming Problem \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/pdf/2412.14382?](https://arxiv.org/pdf/2412.14382)
17. arXiv:2104.01646v3 \[cs.LG\] 18 May 2021, accessed March 4, 2026, [https://arxiv.org/pdf/2104.01646](https://arxiv.org/pdf/2104.01646)
18. Adaptive Operator Selection with Dynamic Multi ... \- Computer Science, accessed March 4, 2026, [https://www.cs.ubc.ca/\~hutter/EARG.shtml/earg/papers08/pap333s1-dacosta.pdf](https://www.cs.ubc.ca/~hutter/EARG.shtml/earg/papers08/pap333s1-dacosta.pdf)
19. Operator Selection using Improved Dynamic Multi-Armed Bandit \- CMAP, accessed March 4, 2026, [http://www.cmap.polytechnique.fr/\~nikolaus.hansen/proceedings/2015/GECCO/proceedings/p1311.pdf](http://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2015/GECCO/proceedings/p1311.pdf)
20. Multi-armed bandit \- Wikipedia, accessed March 4, 2026, [https://en.wikipedia.org/wiki/Multi-armed\_bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit)
21. Reinforcement Learning Applications \- Contextual Bandits \- Meegle, accessed March 4, 2026, [https://www.meegle.com/en\_us/topics/contextual-bandits/reinforcement-learning-applications](https://www.meegle.com/en_us/topics/contextual-bandits/reinforcement-learning-applications)
22. (PDF) Adaptive Operator Selection with Dynamic Multi-Armed Bandits \- ResearchGate, accessed March 4, 2026, [https://www.researchgate.net/publication/220740267\_Adaptive\_Operator\_Selection\_with\_Dynamic\_Multi-Armed\_Bandits](https://www.researchgate.net/publication/220740267_Adaptive_Operator_Selection_with_Dynamic_Multi-Armed_Bandits)
23. A Reinforcement Learning approach for the Cross-Domain Heuristic ..., accessed March 4, 2026, [https://people.cs.nott.ac.uk/pszwj1/chesc2011/entries/urli-chesc.pdf](https://people.cs.nott.ac.uk/pszwj1/chesc2011/entries/urli-chesc.pdf)
24. Thompson sampling in practice: modifications and limitations \- Hongsup Shin, accessed March 4, 2026, [https://hongsupshin.github.io/posts/2025-01-18/](https://hongsupshin.github.io/posts/2025-01-18/)
25. Thompson Sampling | Towards Data Science, accessed March 4, 2026, [https://towardsdatascience.com/thompson-sampling-fc28817eacb8/](https://towardsdatascience.com/thompson-sampling-fc28817eacb8/)
26. Multi-armed bandit-based hyper-heuristics for combinatorial optimization problems, accessed March 4, 2026, [https://ideas.repec.org/a/eee/ejores/v312y2024i1p70-91.html](https://ideas.repec.org/a/eee/ejores/v312y2024i1p70-91.html)
27. Understanding contextual bandits: a guide to dynamic decision-making \- Kameleoon, accessed March 4, 2026, [https://www.kameleoon.com/blog/contextual-bandits](https://www.kameleoon.com/blog/contextual-bandits)
28. Scalable and Interpretable Contextual Bandits: A Literature Review and Retail Offer Prototype \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/html/2505.16918v1](https://arxiv.org/html/2505.16918v1)
29. Run Contextual Multi-Armed Bandit optimizations, accessed March 4, 2026, [https://docs.developers.optimizely.com/feature-experimentation/docs/run-contextual-multi-armed-bandits](https://docs.developers.optimizely.com/feature-experimentation/docs/run-contextual-multi-armed-bandits)
30. (PDF) Contextual Multi-Armed Bandits. \- ResearchGate, accessed March 4, 2026, [https://www.researchgate.net/publication/220320918\_Contextual\_Multi-Armed\_Bandits](https://www.researchgate.net/publication/220320918_Contextual_Multi-Armed_Bandits)
31. \[2409.13888\] Causal Feature Selection Method for Contextual Multi-Armed Bandits in Recommender System \- arXiv, accessed March 4, 2026, [https://arxiv.org/abs/2409.13888](https://arxiv.org/abs/2409.13888)
32. adaptive operator selection utilising generalised \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/pdf/2401.05350](https://arxiv.org/pdf/2401.05350)
33. Large-State Reinforcement Learning for Hyper-Heuristics ..., accessed March 4, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/26466](https://ojs.aaai.org/index.php/AAAI/article/view/26466)
34. HyFlex: a benchmark framework for cross-domain heuristic search \- SciSpace, accessed March 4, 2026, [https://scispace.com/pdf/hyflex-a-benchmark-framework-for-cross-domain-heuristic-2mgdshv3ei.pdf](https://scispace.com/pdf/hyflex-a-benchmark-framework-for-cross-domain-heuristic-2mgdshv3ei.pdf)
35. A new hyper-heuristic implementation in HyFlex: a study on generality \- Lirias, accessed March 4, 2026, [https://lirias.kuleuven.be/retrieve/a67fadae-7f16-4b50-bc6a-93346fb8f951/](https://lirias.kuleuven.be/retrieve/a67fadae-7f16-4b50-bc6a-93346fb8f951/)
36. Contextual Combinatorial Bandits With Changing Action Sets Via Gaussian Processes, accessed March 4, 2026, [https://arxiv.org/html/2110.02248v3](https://arxiv.org/html/2110.02248v3)
37. Contextual Combinatorial Volatile Multi-armed Bandit with Adaptive ..., accessed March 4, 2026, [https://proceedings.mlr.press/v108/nika20a.html](https://proceedings.mlr.press/v108/nika20a.html)
38. True Online Temporal-Difference Learning, accessed March 4, 2026, [https://www.jmlr.org/papers/volume17/15-599/15-599.pdf](https://www.jmlr.org/papers/volume17/15-599/15-599.pdf)
39. (PDF) True Online Temporal-Difference Learning \- ResearchGate, accessed March 4, 2026, [https://www.researchgate.net/publication/287250918\_True\_Online\_Temporal-Difference\_Learning](https://www.researchgate.net/publication/287250918_True_Online_Temporal-Difference_Learning)
40. Reinforcement Learning: Temporal Difference (TD) Learning – Jordan J Hood, accessed March 4, 2026, [https://www.lancaster.ac.uk/stor-i-student-sites/jordan-j-hood/2021/04/12/reinforcement-learning-temporal-difference-td-learning/](https://www.lancaster.ac.uk/stor-i-student-sites/jordan-j-hood/2021/04/12/reinforcement-learning-temporal-difference-td-learning/)
41. (PDF) Q-Learning based Metaheuristic Optimization Algorithms: A short review and perspectives \- ResearchGate, accessed March 4, 2026, [https://www.researchgate.net/publication/366905818\_Q-Learning\_based\_Metaheuristic\_Optimization\_Algorithms\_A\_short\_review\_and\_perspectives](https://www.researchgate.net/publication/366905818_Q-Learning_based_Metaheuristic_Optimization_Algorithms_A_short_review_and_perspectives)
42. Online Cluster-Based Parameter Control for Metaheuristics \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/abs/2504.05144](https://arxiv.org/abs/2504.05144)
43. Online Control for Meta-optimization \- NIPS, accessed March 4, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2023/file/745b7e084d5ca5afc07fb454ab2be522-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/745b7e084d5ca5afc07fb454ab2be522-Paper-Conference.pdf)
44. Large-State Reinforcement Learning for Hyper-Heuristics \- reposiTUm, accessed March 4, 2026, [https://repositum.tuwien.at/bitstream/20.500.12708/192686/1/Kletzander-2023-Large-State%20Reinforcement%20Learning%20for%20Hyper-Heuristics-vor.pdf](https://repositum.tuwien.at/bitstream/20.500.12708/192686/1/Kletzander-2023-Large-State%20Reinforcement%20Learning%20for%20Hyper-Heuristics-vor.pdf)
45. Cross-domain Selection Hyper-heuristics with Deep Reinforcement Learning \- reposiTUm, accessed March 4, 2026, [https://repositum.tuwien.at/bitstream/20.500.12708/209227/1/Mayrhofer%20Hannes%20-%202024%20-%20Cross-domain%20selection%20hyper-heuristics%20with%20deep...pdf](https://repositum.tuwien.at/bitstream/20.500.12708/209227/1/Mayrhofer%20Hannes%20-%202024%20-%20Cross-domain%20selection%20hyper-heuristics%20with%20deep...pdf)
46. CHeSC: Cross-Domain Heuristic Search Competition \- University of Nottingham, accessed March 4, 2026, [https://people.cs.nott.ac.uk/pszwj1/chesc2011/](https://people.cs.nott.ac.uk/pszwj1/chesc2011/)
47. Reinforcement Learning for Cross-Domain Hyper-Heuristics \- IJCAI, accessed March 4, 2026, [https://www.ijcai.org/proceedings/2022/0664.pdf](https://www.ijcai.org/proceedings/2022/0664.pdf)
48. (PDF) A Reinforcement Learning approach for the Cross-Domain Heuristic Search Challenge \- ResearchGate, accessed March 4, 2026, [https://www.researchgate.net/publication/215777402\_A\_Reinforcement\_Learning\_approach\_for\_the\_Cross-Domain\_Heuristic\_Search\_Challenge](https://www.researchgate.net/publication/215777402_A_Reinforcement_Learning_approach_for_the_Cross-Domain_Heuristic_Search_Challenge)
49. Key Principles in Cross-Domain Hyper-Heuristic Performance \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/pdf/2509.02782?](https://arxiv.org/pdf/2509.02782)
50. ICML Poster Preference Optimization for Combinatorial Optimization Problems \- ICML 2026, accessed March 4, 2026, [https://icml.cc/virtual/2025/poster/45664](https://icml.cc/virtual/2025/poster/45664)
51. \[2505.20881\] Generalizable Heuristic Generation Through Large Language Models with Meta-Optimization \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/abs/2505.20881](https://arxiv.org/abs/2505.20881)
52. Multi-Armed Bandits Meet Large Language Models \- arXiv, accessed March 4, 2026, [https://arxiv.org/html/2505.13355v1](https://arxiv.org/html/2505.13355v1)
53. \[2506.19977\] Context Attribution with Multi-Armed Bandit Optimization \- arXiv.org, accessed March 4, 2026, [https://arxiv.org/abs/2506.19977](https://arxiv.org/abs/2506.19977)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAXCAYAAADHhFVIAAAAu0lEQVR4XmNgwAEYESQjjIILQwFUAlUIRiDRSABJCMpEpdCYDAyBQN5zIB2AYSyQigUSF4EsDYgQhguRXY4CMEVAYilAch9Qah+QNkOWUgXidVCjTgCJZcj6q4EcayCtCJT4D2SDTIECiDIWIL4LZC9CdjqYBNJhQOI/kGMJxDJAdh1CASNDH5C8DtECZutCmBBZkKd3AhnLgGw/hDgMQI2AUsRIgAGeyMYAMEUIKWy6oTSyIjQBBEDWDQDprhFg1U4tAQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAXCAYAAAARIY8tAAACMUlEQVR4XpWUv2sVQRDH50IkghYq6pOoARttFYmFqAgWCYQUGq1FCytBUPEHggiCoEXAzn9ABSUpjAmBWBmJhRYqggiChX+AYiBt/Ozu3O3s3d57zy/vuzP7ndmd2b27J1KhiG4NMaKeNanUgrak3kJaxUo5pa3TNqmBfnI02oEzcKsTYpOZiv8D3WMTwwreC/yP2O1pVkS+hDZU08phEL6Bx1WaZFgu6ntpMIrNLQPy+m43mNC26Fo9v7gqHIf6pDtsVn8rDGx3vTvtFpFG8BLKEtpr/D34C1j3oH/Cs2lqnyhvhd8+hqd4Hfx11G/wmKadh6vow8k69bpeVylgr2NO4Z2QUGAiZnl/nZxzRpuFp828jgtwQ3BNFXAXsyb+O6hwU0LRcaNNwZ1mLqb/XXBVGgUC5pkulhMNzUkokH5smbtRQ/Gi2iOi8B/YH5JuhUw/7MCsYad18UH4EL4M0wbmyP+F/er9BIUckXDXzyQ8uwGG53hv8Tdq1hV4AP1Htc5A+1pmPFlTPa7Cv/AefE/gA/YR8SGTtBfegdPlbvGwHlsk3P9QlKSq7N4Mc3dJisUXOCqhmIHPH8csqdCx1Qfgb3hfgx6ZEofhJwn/VTf0BO5qx1wQ/zbmAeTU/qQVRom6N+VMZlMP1ffjvcM+hps19ETC1+5wCL4i5yJ2xHb4XfyrWHzGXsu1HiX19Pg6c50nSPLNVTVh90ulEiPwcl1v2bPxVvSATzwq8boyDRXyD9nlQ7OVLbIzAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAYCAYAAABXysXfAAADf0lEQVR4Xq1XS4iOURh+j/tlM0YYtxqJplxKFspQo4RQs6EplGJjJknNQsTGrSSxkGyU0pQFslHDlHuzcA+FyMKShRUT0nje857v/871P9+feer5zvmf93Lu5/t+oiiULxgYPWIOpJRrIIwU6iTWpjp2FxHHiFQiYoxIcaQcU3rakEBkGRpNMYJ4Bg6j/c8oB1yqAcWlqmnvwT/Q2X+Y48B9RSJ/YMGYtOCqgY+NKhPk6e3gX/Ar2MpCKs5gAtgOn0tw5ME812omKA63t1VSVPE5SjzLiu74hgw2k14pWuaoqRYjK1Mipadg/CNhY8BB0ttG9UTsASyfXsWTEQkqpUjDEf8Rgs68EPwB/gJXOOYUZKbH4bHbs8SRGUDKHOiB4KC2X/eQHOpPYFMuKoTJUyUs45MxV8YNkgHd9A1x2FsoGMwa8Ck4CGkJyvUwXkP5EOSSNQP3AsgPJu/BmAJ+IbmpdvnGBjCWZGK45Gufc+6loheK3uDRZ+p2ESDQA6EOlJ5BfRl8Q32eb48ibGADuB/6NJKVvujZX4J93koW9UPgCUv1888H1zpKHLWosySdOOXoYaeTMB3dQpJnpWVqFU0lVl6txmN52FatD+dJBuz1x54aNxjvDvUE5aTQ6Dqm2jQ4B35Xcv0XOAj+BJvSWQsh2rcXIA84Bx01m+TNPtfTI8i+uV+BV63f7P4OvGJyXkDRbOqLwAPgfXBiEVBCdZFcHkMkn1dducYng4/JftdkAmx4rrMo3E4tSrbdDrANvGyCRhOfMcEH4m2mETTOZ5EHksUo8BbCeZ97CJJGJQ+rwLfgdE8/STJhp0m/zzR4O/Nlwas0pOQWpEgjHHOkwo6gYyTfaRFEQkVqBa8XQsSrUWwC7/qilfgR6h0kg22uqR464cQvMlWlQ8aHD3U/6jsdYwo6yETa9VJg8GQetwx8fovPJd45Q3Dl87SNZJuSji1TqaV4PCD70AVtBkNsg8Qr8huc6hurIcjJEp+HjZZlO8kZW2B+4+KgHnBrGK1oDp4fSQ5kB4QOKQOuA7vBM/DhG6X4c3abGGFme7Is1YVnGU/ysctfIQbag1emJfCOgFek6FhlqrLenW8i1Y2a2gkeJvl/1O96K77l+MLguqX/J3SqBvI5XbLqNozeC75G/R7KGZ5xJri4ptVFqhWNukYXlUeauvGMGhhTU+I6/gMd6YTxT1xegQAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAYCAYAAADH2bwQAAAA0ElEQVR4XmNgZEAAZDZ2ARgASzCiyiM4UBZMAKoYO4CIY3EFCoVmIobdUDEYAWejKELWjQlQrYgC4iNA/Aso8BsoKgFTxgTEc4D4NFDCHUizgEWRDKwAcj4BRRSRnQMDHECR90D6P1DiP5D9H8xmYPCFqdSACjSCeRjeZmA0Y4AoyIVLoNkjDMR/gbgLJoHibajYdCDxCMgSgMoIAcXjoCrBJMhbzUB8AYiXA3EfEBshyaMai2IFMo2QQGUTA7AYC7UaCeC0AFUHDmVYnATVAACyexWYv0vlbgAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAVCAYAAAByrA+0AAABEUlEQVR4XoVSuw5BQRCdaTwKj0Yp0Wh9BaVOpfMhSlEr/IlEQqWQiEYrkeh8gYRChDP7nHvX48TMnntmzuzuvYgUWD8EgZXOQYuFmPKLYropQeLK4ecOHlrRTS5p7jFAvBBPxBGxRlxRF+2B2IFvXJ3EuEAew1+zgpm1Ijuk7TWgI4UqyDJIdtcSlhvWU1Zn8xuBDWOvYT2S6UzzKJlcltTGQzF38wnZ48jd1H25oKcGBr4lu0M9X8kY3KQK0gN8n+10SDSmPvIL+jQtfsaMrKH71+COdEC6g5UyegpzhxaZr8trZ9blsDbI/hUkLmReJ8srPaMIjZuJ2Z9F8yh9OVBocjzBV7PeSMMbkoLBZ1XwBkh6JWn84m0zAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAYCAYAAACoaOA9AAAECElEQVR4XrWWW6hOQRTH17jGyXEpl4gHIiVH4skllCJEueTyILeQeJCn4/aiJJQSJ8eDQ14oXigppRw5IvFA8oIT5UFCKYp0/Nes2XvP3jOzv9nn+/xqfXv2f625rZlvZhMxSv8mj1KyGJWLD9UN6ZpSZxinWm3BQ0wMxYWVxpQ6/yM1+g25tR5ylhO3I2xi4v16XnXa6eUsomtEBTpBjuAhJqY6ToKiyS9qYzEthxbLI/kpDSx1hqlQrUJoBPnWZsC6IP3Bswf2Ied1WUESx/YG1hZKrp98YHQ1TbXoSgSbltl14OcRyaQHZLpBysNhB0liToXaC+k2MTF+el+zHl7AWkkmPpkFzzA2wg4TxyhaWnSGqLazPKGOEEOoUkgvYIW1wC7D1pMkZ0nitGJWw8bB7sN+wgZnrtpEDilMdIYbFZOxH7YZNoskObttJ5oagsdalPj5G3bP9ntx+s8LNdwFvM6FsBuwLrinEy9o8k70DDYRNg12BXaHZMwrdU0Hb/spt2BjYUOVJOcEi1aVVpT74rmMxH8kcTjN2ivsOKvjNCFCk+LzUVF/lN8SXyKK9qZeopew57B22CCjtcPbbcq1SLvlDh5bjk+w69b7FpIdxZxENU7O3MytvzUmobBY3nz41ZQabg/LSc7HkSSLdSHv1snh3TPQ0s7DunOLp1FbSXJgXjMPswDCaeu9E/bUlJtgvCIJnMTvsH6FVs6R3GIOhb54Ery9c6aKZaXLO7NqwextIEnOHEsbb7QD8ppmg3dYRy4tRGPw+EF2csxqJxxT8u1iPPpg/mZed1BWsZnkW+imebfhLTw/e81PJh1e47kI+0p6sVK2kSSHLxmDWqREW5VpWl+Dn7t5TZOO9gms2Rr7UZKGcHOpFmtivJVZ35dGyu3WCfcvkpXn9+rEJM4MpBD6DnZNPCmXYO+lmKqsfSH5ftsF48+Q27CPsFembCH1ZhP/D+WwTeBbq0elSUg7aGMdNiURjIc7sm4vo8ZM2IO1GEW1CN9EPJ7tOVXpCfNBbPOa5NYaSnKbJTvtIcmtJ5huOCm8Y/6SdMCHcPLX4v8vT7aPeT9DkmH+8Etizxofc4qs28s/jwJ1x+gdtJD0qqtRVugI2GfYvExi1AbEPEDhKpkPXDCM5LyxD+1I7MGVDjTNPp9NPLjQ8nslTUi3iYlxyFcqNGHv+NG2oz6yXnh38XnD3xKboE9NPb3COVO8ZDFx8Q6yePztdhxl7Bx1yPWH8Dg9UgJv1T2wdSUxHqpFMzKnoloDp5IpK5pJ8vHLt9sEy1NGREhC2o+fkB6Nd05VqBbt4PRfZ3t5amz7Umd9RDftJKCgNYa4Fv1RJokep0f6bzS2r+jWwoH+pDhCBarXtWv8A5THi0qrFcbkAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAABGElEQVR4Xp1RLY8CMRCdIZsQ8CgkFg3cP8DyP0BgMfgjgT8ACoU8hUDgOHuKsFmDw/EXuNdtu51pV9zxkvnovDcz3S0xAcE5uLymJLSaTevuEGRyajRaHoE9rIDNylMyyCMMNA0vWDchVbetDdmImeaCVtoYd9i3Pwhaf7ULK7JX6XmlmicbgA8qxbwIpIhVFjYUcD9ao1D9JgQ+s71KX9Qr1o9owHawLeyF0jIZ/S+xAWoZ3AHpxjXncDdPSjRR+UK8gMgc+0n2rwys1o5oIZyQX5F3qnaikRXT2hfMpx4RnlT/CDnbFy1LbfgH0klyMQOmKZyZPo6pCLKZ/ZUFVTNcQHf/XesKYUO0MiHeRWhXg/Q6uU3KfgHssCP2aEuibgAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAYCAYAAADkgu3FAAAB/klEQVR4XqVTTStEURh+j88hIjVkoRTKXlnIysrGyuQjSshHKaWUUlKKshtKNnZsNCVjg2aJjYWV8ifsJEuec8+5977n485cPNNzz3ue9/Oce4fIC4GfD1oNFn+EiTQxZeCk88aO85+Ia/PTe5qFgUkD8EKWUBFOqJvmKiGSpi5bz+hYIdKRKoDXdk4WwunpBDmCV0oBz4tNDZ2bKo0f2bJMydQ9UTEs5zr4Bq2AtRXcAi8R9Iz1FHqLEc3BC/FB3YFEPx55aL1Yv+F6hT2kgzqlBq7yjL9iDxwExykoKmYCVU2UFd5GYhiPJ/taZJKWmmDPxm7zZEfYfsCojSWaI9VInpajHcx5+oSYhH3HPAZewKJ1tbfggzVRAoz/XB7c9qV1kJp8jWndSJbaMlgNXgSqoE08irBG4tAI8n2XwE/EPGLdN92Cpsm9Inl8qXWBC6Qa4iOhAejnSJpnsRzNImhEGbmxD4Up6SbaKa9MKMC+h70BVpFqmoH+Dj1rl9G7MbBkN9BIkP2YAK9JXXddlBuXOAB3tCRjDGewibdmsqVfgTnwEGwA66GvYG3TQfIDGgV7QPnV+sA7J57yBJ4zrFM6Rhb8AhcDr6AlPI5hyeY1SmJIOk1K9JG60jRIVzmIcg++CzZy4RfzekKcLtFOvpMK4MnOFN6ijh0jKV7QD9lsNo7JOQBUAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAABE0lEQVR4Xo2SPU4DMRCFxwJFooAGcYFU3IASCdHRpMgBli6R4AJ0FHALKiS0iC7dltuEDnrOkDtsPmOv7fXYgpFm3vObH49XK2LN/EYHPnhJWyz6g5csJvwNunKqqPwolLtTYVJZMLVwNjpqBe6s1K2tmvlPooF/gt+or+AF/o63+E0oBBvgAT+AH4ED6hd4ivfihrhKbGPi8ufGFhu5h9vGFn6ZLRe+wJowwOehPYTM0N6An3hRCkYOCU+wJT7Dd/hLMm4B3LoWkSsO9kF38JW4xz37gWfwDjz2teZE3CN6sKPoGr5F/+D8CLdfJF259rNnejzUGxJIiyaJxDJB5b1NBqmpnqv1RiE9lG+orlHrMrIHKNwbfdDDKEcAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAByElEQVR4Xo1Tvy9EQRCeFSEuokBBQndIqCTcH6DSSO50EiLRqDSqEwqNXK0QolcIV4hGqDRUEoLQSUhUGi3F+fa9ebszb/fkvuTbnffNN7PzfhHlYdyiEegm5koRJFgIdIbvrRyRKmkKC1pBdOwyuOL1iEPNEkyhpENwMhOCppFahVzBLdim9WYwulgWIB7AdpbXPWJqMKWL5sF1LcWgkqrbMLgL3kF4x34KHoNj0toCzCy8XwhqYC94wcUl8BPxROJyfgk9XQmXvwiqLHSDJ+55GTrCUg8aSUHEdfCJ+M0BM2BVPMc9LB/uiqgD2wLYLzzJ2gP+gJsisWVsQx4L2z2Ca04ugfvQG9CL7OcU0bhNYK+I4nMEBbZNY7H5bU5naCDmZg6mgPw3glUW2sFLrhoEHxA/Yu9if9bPHlD0B/hTNsBnsp+GoSkkdiiZmF4Q38A2JKbOipNmXM/wDSuU/j6v4Bslzc0a0p3OIWDSyUY4FrIGPgnqs4HPcKStttmov8yS+p6v1FUs9reZ/hmk/l/3LOzYByzl4DxzWGomfZvWuxxYubiMbTGrCkwsBLpCcB8a4taEwIgdLL3/CC5ullFomohCfndEf+PkNlqsvdiwAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAYCAYAAADkgu3FAAACKElEQVR4XpVVvUveMRBORBRFOqiDQoUifoBOBe0f4OQitG5CRejSycVJUcFFnDuUipO2dBB1kC6lLhaKToWKVrsJLXRycdVBn0uOJJfc7/X1gSe5e+5ylwvvhzE5bC6oUg08lK3EnRSXoCpS1BV4vTiVe5ogoV8o9xKluKQxL8E3UVcSCyjNhKTX2IDwXEpKYmFrN9G0iGOEG3IxP1S7hILsYl3Y9qX+iJL1p9pJ5M6xLSOJVflSNWFND5Z3OPcT3l9wF9wGB2ViMX265Ju3kgPj4BW4BraDXzn4AvwPDodMh3rGKHOo2C0C8+y3wd5JLvkZ3ONYeZwT06FSeMlH97CdGfcJc8IY6JryuffgP286tQnLFNjpvLA49IKTwUvwBLxB5qLz/IElQ83i4V/gdw5PY/kA8w52X9KlCxtd6BDc1CYbgnaH/VUY35gvYCvHR6FTfMW7oQJpfWk9tpcNNSp1V/AafMt+I/jNm7Ybywl4CrZw3Ee4USgTJ5ONou6wAP4G8fE2I+AqOARegEfg05AZwY0IYi7RSLwgO3g6cwz+AS+hUuNZBJvzRN6oUb8oRLA0kd2KepaRTLgDdqSxEi6RGg3khax/uo9s52EP1g6kypDTEKhR8YsBLCPnk2hQNrP0FOvBi3qaO2H8rwcaWcqdYZ2+JjTND/CcJ3vGsQL0R/e6vECJ9Fn0y9dTRaCqWjplVVHxEtWFdFTlK7aAck4XVMnUN5kx99Z7QlWa4T/QAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAB9klEQVR4Xn2UMUiWQRjH75UkDW3IRcHGCMpFyKBNnBwcUnALhIScImhKNGiJRmkII6eGkMiEaBEdxEWnoMhIcFBSmlpaddDf3T33Pnfv3ecfnnv+93/+99zzfu/3fcZYVG4NKeOZkBUb0HqVe52QqrG/BQqNBHE/5a3cpjjdffjDWg5o2aikKd5RHmyKTUS940cTro+zw9IWSerJUFK1US/Ll1RPG+nFQU02CSaIp7pN28STxim2XSdeE99Q/5BXKH4k31RL4eIwUDTYKOkf+RVxDb4m1rvEX+K2WiNW6G0PnBLPZN+F6ZMl4v1AfJaaU6Mhsn7WuEu4NwdGsITGFm+II+EDxLLx/kUadXvZt7xKPiHPRTfMG9ewxnc8W3Jgk+iDt7Nbhb+vJ2W9xXrGZtyfc/JX4orshmydeCGa/VzHxDrFYmvhiSpr+E/MyA2XWNaFM4H5Af8J75QD8Wf0HL5fq65QmVnWX8Z/Ne4QLwk78W+K23j6/YEEHcQhZ6cbums5TtqB78EP4DSvHsMvJy4dawn+JJkz3ThuvxI9kajQbo9Ik8JH6hZZK2M2LFM9udniHrGAPkxlGL6YTSfbG/C3hQsC6yYdG/92Q6zpwAFe4c/QPAhCNlOGlg4ppEmRXV/yiyk77HDxb9Cj0U1NhVOFDtmQgnNtVzx2PBUHewAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAYCAYAAADkgu3FAAACTUlEQVR4XqVWPWhUQRDeNYeiBgW1SMB0dx5oJeiBYGVlE0jSBQwBGysxaAiKFjYSLC3EkNoUQVOIBEK0SZNrIuTv0E5QSBUQW0Uu38zs2533dpfj8INvZ/abeTO7+/ZxZ0wGNuH9N9JFEw1Yyuk52HJY1wh+okKUoxEJFSmK+1WMgXd1OO0n1LLpiUVkXvWzPh9OIv2wbWM45mcqEiFaRDY7yhwCP3jJDcHvAV1O/OxTE+DDqlhF7ulYLysj4CvwC/gDfA8ug00Kpo8lLinI59zG9BDaPPxz4JrItoXhALziM/M1UkJJamHyF/axCwxifBfCdgnDSqqGwgDC943bPYPzi09FxhVw3/ANY+GWcU1d7dfgT/ZEOA4zCXtBwoyzCHZhif/AGRVjnMHDf5D0VC34mZVmBbaRs+H8KfAN2EVOXeWgkXkL3rBUM4HLRlYxrrSPaHzK+dfdSp8X23MGmlWN7CD0l6UMtXICFfwN7Z4L1GDWnT8M7oB74ElWWOehC1Mn30mnMa4aOfLPhhaWwBNkdgxdb2uuwb4wstOv4CZ4sagW+vAp6KOrmfBeaFG/wJs8k3y/TTq6NmbfYL+DHSu36IQkRKBGjTDV58T+FjinxAh0rc9XxQSo0SVyXAvsxtIHXmAXfOBnYR3e++QlDb1gAd26pjpLKjrnfNIPYRvqqB1EoaNYUEKha2kUnIdPO1qANC1B/sgfQZ81dJkMv2tGvEbLP3R3qrIgThf00HNhQTkabz9GyKn8RwjIyCbRIJ9aQXY3/f6g5RNzkSPOS0etidDyxgAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAB40lEQVR4Xo2UTyhuQRjG50j5kywoUZSNuKzUZWdjZWGBpayULGRjRZeyuDdlcctCiYUoC/lT7t0IGxtWihAbKcrqbu6Wxec3c95zzsyZ+b489cy88zzv+87Mma9PqTyiL8TFpTwiK8ntZlZeB0/IQarCTV1kGzg5boGshuC4Y6SQDH8fX5Hd1pm6i9dphNUMmX9JXOZKdpxTw32N2giPdOw1KinkINYI0UyW6zY18ATlaC1wBeEK9YV4H+7Cdu8wNjxBqQH4Dy7BOngsei+5bxR0pZn21QKNdMEH86y4NUx7lr+DfmD3KIUDeAfNy4F+ZRqrpHgVvkrcyrTJ/Buewg6jCmrhO/xhafORbpjd5JrpXDx9/VHR9em3RTfohAWM4UQg/stQLS/Yw1AgXhTzD8Os7LMFD+2bU6T+w0m9wChnOpG4ieGG6JZllXXSpO5JxZ8kVsWYg/cq/ml8h7+UPnGkHsi4IG5OC2IMqvhU+ttVJqLlm2tewkf4jK6bT8OKNMPHCg3OlHyPEPYw6vOiyY3MYy3AbyJPoBfQG7IkNzotsVFrFL98n6yX5QYC96O2wbXUCzedgj+FG+ToX0MCJ13/GY7ZQqhb4DYWbC3kpyhuFnHce4fh7u6n+ooDY2eDrab4BByBPDk04ZkpAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAB7klEQVR4Xo1UTShFQRSeeSmR/KYoFhZS9JTixdbbsPOzxMbGysZCXoiSLKwUpbeVhbAQCz8rFqwU8WKnKKVsbFnwzcy5956ZuRdffXO+c86cM+fdO/cJ8RskM6Q9+EleFZc3+Fc8ysc0JXg9qEPcTJG2Gw2A45R24DTywBsZnYfpCPMu+Oks4ppAX8GkWCiCfzCHF6kDD9ygRkIHbxRmhqCmYuKWckJWohFcA68Rf4bdA3fAlsRhEtAHvoMrYDV4TPEM+Aq22SMmv80MEl/IzpBfBr3L8tvgPvN9sM5q4z0iKYr1QgeNAbmB5YW0qsPzFKvQOeh+3qgc/ARnaa9a5oRuGOIGPCedxZ4t0tPQJ6Q1WlH8DTvInsUhRCnpLmjk5aLeLcUFlizpSixNpPWKIvEBZ4IiReAp5eux3ELegSVgldCNxQLiS9hxJNSLoSEC5MCCNFejE4lloSaW4gH2EmygfWlpmo0ph57dGeUsqJ+pPp9H8AksgJMoKA5OpsNUs3aao0f5Ul2lYLBoSr2oK1ETejbUd/oGpo0ru/WkUtQaN9wXKj22cwDfqK7JMOkRUL9N62Cqa8ay6UTdESvAdXBe6Csig+dpwKZTf4ajVjGfjofiAomJP3XctxmT8Ibxqwx4rTWd14F7pLymduAHfA89av2ghBUAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAYCAYAAADkgu3FAAACSklEQVR4Xo1VPWgWQRCdFYkomkItFLSLCokWggYsBLEQCwW1kUA0YpNCbMRCiYiNWFuIwVKwEE0hNmKwsFEsApEkKDaCgVRpLCwSC30zO3u7czer9+DtzL6Z+WZu7+ejQIAsJTqCRZ9wNycYrRvvgV5FtSRHLyd1wtQOVq/MwTnwqnhuOKMT7gj/AHKfwBxu64LKuBH2diQtLRZR+ghuMJKTKiiPq+WJ79eJugt8ZSULRyJXTZITYlwAb6RNJcegT07CXvAhOAf+AF+Cz8EDHGwdhvX/16WIn4a/CvsA6nbYN5oximUFiSOc7f+erzLaQ4xi8xv2lu63Ivoih8MzLDOdqtS408cKcRfXGXCR8hN2kqSpFgR6hGVZfcYAOAZ/Z0xICHexzEL/BHvKDAE7CLMOThVz3KHYLGEefK9Vl8DH4B/shvIsdBlmSnOOgWvgwRTkdZiLwPOyjXgNblH/KHI4fi+HiYuhhaFCuU3xvUv4RnHgBvyDP8FJ3W8E36q/G/wMLoCbRcmXzc1zIz0nDW8Cf4FXmrgC04SlEB/vI/Dvww6j6gv8D/D3FA0SbKMGkngNZonyqRjw0fGlfwW/g5x4neJ0HrjRPvHs1RyC/454OIXeorRtfH6sdxSqixAb7W/JeC3oKbhN9yeiCbaPYrYtODkMbiRfDAWnTcOcodjgItl7ZN52PorpHMko/LMkXw9pxLkTGrypWsnjrTGbDf/RjRcBg1zijCCm9KsnUcJJcSRB2aCU1KuVVeCkd6Z2cgS9mtVS6nqt8V8OckiJ96VeNgAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAABYklEQVR4XoVSMU5DMQx1VIRUobIgLsACN2BEQjCxMHCAlgkkuAAbA1wCMbVCRWxsbLDABiMSCysDd4CX+Cex46T/Sa/2f372d9JPlOByqnKJmidEJ0oyT9IiwVQTzND8U28zJoOg7yN8Ih+xyxxBo6LPoH5DH1SKPXfB+AGvSpEWNDC4uAn+gVuqltA6f5aOwefCo9fOQW/U5TfgRKnCNAbfIHwgTqFvI7+HY47ng86DC6QVTmMnxzF4gXwAYej8OR29Q1sDX8gPjhup7TMeKZf8BfmLOgeHIDZwO/KYjRmpcEo8YKPmrEgMUbgDv4xatMrjLDn+MI7AZfAXvBUbHSKf5IZOFfkuhZXdGeIJNL/+dVdfx+8TOIoNclDcaRWcO75tb95D/or4ANMl+X+iaOCnxoN8Qwu5UnjMIKNqcEWuJ9/cs0VALJu3mRktXUSda6cZlIu1QmtQ6UpCjymg+f3TP79KI9bOOhUhAAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAYCAYAAABtGnqsAAAE1UlEQVR4XrWYXchmUxTH/xsxGFM+x4ULDIWS5mpywTwzuRBpXCjSxDu+hjKaMmmIDKEpJSShkLw+hkQjH2PK2yDyneSCiyHFlTI3KBLrv9d+n7M/1t7POe/7vL9az3POf6299zr77LPOfh4gwsUnJpMj6jE1veVBcMYRzeiAi6L6xA9lSJ8DYgeENrGny1b7UmtR0wuKwEQovFPHj9Ccg3jVjKUL5XNnLi+eYqRAqoez28Q2Jg4yH1rrqsaC4o1GhpSzSuwDCTw0d8T06GexHCQ2JyOtyx0TaSYXnM2YPtgdHCb2pdjqWOxC7UZLCG/mr2JHxWI1i6pjjBERS7VlZ1CJ2iL6h7k4kSKHqbJb+tyWi5VxUrWIKYQJhPhaM0P/VGwmF6sYHSwB14j9ItYsKREDsipCCyGlvVJOE/tP9JWWU6npS4rmBazPHTEPiT2caZsl33czTZmwshaEw1XyeWDKfdpCn/xT549itxq6h4X7gDjuzXQ+Tq/podFq+vAmfpaLETNin4h9Lfa82BqxXWIvi13chXU0F7zDlfKx1/GNDzwldgc0B4s9Yo8ZvXhG0CW6IdIOF/tHbLvdZHFU+nwFmmhHFzgDvcCDobkx36/Efax874NO7BAeFPtZ7JQwCOuc9OluT6LGuF0S9VKugo3FcSc0oZWR4zyvOazLLne5szaXGcUEqXA0NNGCEP8GdDVZ7EbX7RnQfLdAJ5Nt1gafiW/YJXWRHLP9tWMFOB/a59owJ+P48MUVulcPS94S+z7TtkNXIBOMuVwsqovFVKV07vug5WB/owVXIK1KaHsT9GJPjX39cVytf4mtiESu7r8RXW+W5xNi76eSwt3272LPZvqb8DXQd3P2vOj0RcPBFsJIbH8uejTbJzF+hBvTDLwo9kMu2i2KPxdWQFffO2NZ4eryE2T34x/f13ORnAW9m3dHTY8AXyrAo2JHin0EfWw4yB/h/P50qMqwSDwjV5tAZWdYHTmHwI+Hy6B7sd/Eno78l4ptis6lVLhNTutlhlsOvd57/JmKx0BXHzX6ed05nHBLxw3QDucLJPt8XOxPMRbUGWiNJPw5IxPoloXzjvr8xYzQnkBOAicnZ31YNTeL3QjN94HgOx66auOfWrMhfmukxcxJuo+EY96cWWifG0W/nt90ZJfEFc+xC56D7rJZe7iE+Shzw8hi/7nYM9AVSS5BWUhvEburYvkFjKD7qRLNljWNF3Jm4tNaxRfFPuhkXSD2sdir4Kpx/k0cw3r7L/RRt56Tk6El6m0R+OLiC3OPHPPmchWmc+dwIjQvbpsKfkIYqAe865wYEr+xbcpVOYKO14ITnP7uLPvpw3HQxdHEdz25/w0S8511K05C2A7YfeQ3wtcB/k+3So6vTpz9GEH3XinpMCwb33SndmY94J72ulxMKLouJyh8zbryafJcAd3rnZM7KrA+sJBuhtaOIbB+vQCtrTugk1kgibK+8lE6N/cNgJf9ntgJuaMP2byeLvat2LJivqG/7b7IxeliDZuT3G1+cfLm0PMmdSOMj1jD+aSY9MkowC0enzruVKaLkXTjuB9ZC25NdqRSrc/0BsQYkqfQC8EzvwupuBvEEzS4cYupdlbDGMSQChYV4x2G15BqDAhtMrmfoTfViDYkm/pY/wO64rDyrbh4aQAAAABJRU5ErkJggg==>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE8AAAAYCAYAAAC7v6DJAAAE3UlEQVR4XrWYa6hmUxjHn+WSjHFNQzKaD5TLzLjPGGma4xaFUVNCasxxKY1SSpFIQkiUy0dM4YNLaNAwhy9kDPmiSC6ZopHLB6GIEv//ftbee12evfZ6z3v86jl77//zrGdd9lprr/OKVOBSIcQ7w5hi/AhDZYf0jGKg0cpi/Kh7gTDaZQuZcwQjvk4qMml8g1Vo0q7VxISMx5st2Bv2Emxp5inQxJjpxulDu7uDYa/CDuhcHVFiZxXOsGOG43t8TE2o8gTs0lScD/VVmpwDewtZ9mye8mThwE1OsWxxzGwVXAnPy6mYkpXOhBqqCj0FuzUVZaR3npqYGDvUUHNpD9hu2AWWs0wcn5XOhDpQbDUuP8L2bQWb0DEYND196qySS2BfW45JmKzwUHSk74LdGAr/P77+oeYZPAN7LBWVoSxDeg11WxZituDyjr+3qnTX4s/HsH9pzl+98cs3Tp50EDvUfSETveGht5MJMUW36bwG9msqtjwO+wv2POxp2B+wP2F3eDujDy1jVl3Hgf6Fre+UONkaCHO4fig6C1Y5buZOnoN+WxNRU3kfsxT3mOXufdzvFP26f97oGW6taNsOTz1XiA7U8iDzrGjw8a1QZKTRtjtTTxStc23mEVkhuqQXazH3Li6/4J5l7hcttzjOaSzJXjha9MN0N4zHkP1hn8B+Ez1jpmVXitaxPNRZ8BvYI4FGzhQNnkl0skl8BTUElfF2FYxv0eJ00To5UCmPwo7y98zzLewF//wg7Hp/X8sO2FewvQIN5znZbg661s22rdNHjVjjxQ2BRtgY6O6ITlE4bX+XdvCMWhoy3bGeJ2FfwmYzt9LOPF4DsugTRNvGlzgfLoJxT7830DiI7NddgRZypGjbeGjuuNqLdIa8Bvs+0QgH+c1ULBN1nnvWbLK6WrjXNMu2UxJ86A2inV8W+pRsoD2R/oBoPWcFs2yGmksGJwBbWvPCTgsyuYtV5PLtZP9m5aZWaHDNgH4H+1R0cPOmZkLGHGK4n1rsJ1ovZ0bKVcL9SfNvdVxyfV3LcJ9sO+58sZc/4eD9I/HqeUj0g7kIdrPoHheCgW7atiQUGcxD6dn+mc4PRP8lEWM03oO0rnvK3LFkuHXmmTTRuyR9aSL7iHaWJwEOyN/Cdihcbs/CTg3qOk60oz+IvTefK+o/VB8d92EuWXxpHePZRpe0fiPsp1DwuGPx5xXY27DPYLfE7u7uINFK2JlWP0n644xlp3ShGs+G8cw0BF8aB6PDV8/Zwu2Cs34Gts0bZ0w6w5ag0M+ig8zlZsHZxf5uQ/6Hcb0QNbUvaHVXa993HuW2dk9jGLMG/2/KnNcPiyOM6JDezcHjYTxHYzaJvuFsxvQpquviF/qY3jEVH8EuH6m5CA+i94nOvNsTn4nRYQ7edbEUwd/OuNwuSx0WWYpYeCN6askKjeCaF8CPpz/WtAma63C2pPMni05d7lntmWuQJOtKCFzGu50uN96nMS2bYa+nYhcdFjIkD39guDMV58kW4Z5n1Bs9JINl3E+DkdNOzZ+l+HveealjjCDdPRIfgEewGyK6v/I/m4YmajDUwOiyTdFZQ/RrNn87exGWHtLLTN2GiEOcflR4Ilmo1AuUxqI4IYf1IjWFamLGmCZHMGuqZuuQ3jMeUSIrnQnzxOf5D/05rIhyNw0YAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAABEElEQVR4XpWSsUoDQRCGb6sEQUgRSJXCyheQVFZeeRAI5AWsA/ZiZetTBCv7gAgSkLNIm2fwEQRJmXy3M5ubvd0r/OG/mfnnn51duKI4w5mguQq2iouzkKgB8UltCLrVNMuMpEXv1nwjt1oVu9rCVGrST3qIR29DEJrGdAEfyb8RdkgP1Bu4kLZ8h7CG7/BS59fwCG+9owHqi4juOgjgCR7IB/82jikOdGv7KMIHYdtKrqj4HJ08xAtO7vxH/ixj4qycXLoUm8ccNsMlwgx11Ygjil/iUicncN8Y4RTTK/EqHHEHPwu51xvSDfEHfsF774ihi5Mq0mNTivalXambZ36KvNBBbz9dYfJoW2a1heonx+4gmvKmiTQAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAABDElEQVR4XnVTIQ4CMRBscwkgEHgEIYFfIBBIXoDAYVFYBIIEQcIDkCBAYHAIFJoPoLCg4AUwvbbX7e12kt1OZ2fb7SWnFIEmKfAS5AIVXHNJC4vAiZIagxbJKl7jOD885fVCysQk1uyv44VIqiO2iA/kFwojX3CmHaJlNhnSBfIPwh3xRRg+cPYu0slypcdoOKKh6c6pIFbgB2fYgPf9GEtlxyheok2DVg/wGoSrtemiHhD4HnyGdUI88RcIXM+RnuCNwiM53W6qzOkUdIKAXDUjDJmDHk5wRrQFPZrVLFWQNyKTvDlCQfeQbqU3sI0HHqcXgm5BTjVpjaUTP8Rzpgm3MSECbZB+2BTcbX8OQBghkhDLLAAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAYCAYAAACr3+4VAAAEWklEQVR4Xq1XXcifYxz+3VjmY7TN17w48DXM5sS2E2S+IosdiSVrHJlFcUBqSKxIxIiotZqSyFdYNg7WkI8c+CjJmx1oMkTIAaXXdd2/+//cv/vr/zzv673q+j/Pff0+7u/7uf8iDbhcKBA8+h17EVPMVjKbZ/Q+W7lbmLlxGIqOBVSkBH12YozPHBhfhseJuWEQisRBKPSAtJPzwdfAI0bC/0Crxg5PgWtag2xRN/d0zKNthOViPN4BD0zUwWj5pvpa8JVEmSZa1QyBid2K97ticTw4/XeLjs5XSPMtnp+Bd0oyWl0FB4D78H552dohsxQwxLVljPpKcD94SKZHeM3J9fj9HnwBXNLpIivAX/C+C8+DswxXgZOQamkraLvVLTZxeK87jrAX3DjO8WFwCrxJi4XTOlH7vZm+Ha5bMm12UTQlRWbeBr6XShHsHDvx4Egocjs5VtTn00z/Bj8bE61DkaVDYSmEquRh9YoP+/J7LhJngH+B3wk/ERlMIr5O4fenKMmRXhO52mg5OEDbRZf9bnDTmFVVgXfkJ4ur5gPwI3AN+HXQc1wg2qbjMt1tcdqBDZnBwFd2lmiC943hHK85n7yGueAX4JWhY5eJjvZEtad16VQ89oH3ix6G88DPwT/ETkyMXSa+ne5sozsG8pRiB5hwHG4U9eP30gM5luOHSZdGt1CjPngKMoYHFnERuCk6DcKHIE/+g0wUvwg7u1Ka7STROi+04qFBZGdT+ODkKN3ldBT9aRyAGXWM58zWcDT4q8Q6ngePiebe/q4Gp5w5O4QdFvkTvK8RfYJofbxAJDXgWyk/m3INK50G38BCDPbXPeqtpUtwyd8Dvgr+Cz6TmhWNRj8kmv88o60Kmu9ICp+FS5ar7NzM6EeLgRO5IYD77EvwZhbCRI9wmGjs6kpD7wB/AE83Gm9Pz5oywX27NFnx8Ycd5eDYQ/IR8G/xq9HdLtyTtlHODwrbZFaOggGTeD4JngJuFn4rnZ+FkxH+NN55yhkk3doL3maFYMYFX/ZIuKVAWoTHx+CyGO3OFG3Uj3ifY/TR4xLaUToqGHhx4bLlicvO8yR3WXvWgeHLkOgei6A9iufboo15HdwJt8dEr3jEQvhw/Stijq2iey/HYnCH6L+K50RnZ4kdfNFRx7Zx/4guuRo4a++COxDGNl4Bf87yi6KHXQKnE/ZGrmudZceJW0WXyQi8LM+r+K+HwBEsvsF0rKcu8AR4Wi4WGJbsE/DaAa6dC/fYpNMGXAc+3rmk4J8ALD25JjdY1CsOqpO3TKlAS4/oPNhWngs8mQdA49g5Xga4h/jBnhhT4y3gm7k4brl0cP4by/OgFzFTK6fbJrpHp4XDRTf7b+D5Ze5E4D7miXqpFTsUsRbuARkwA60URl8letWMaAUNR5ohlHi6vgQeb219ldlZ6nGVMT4LhAeW8xcgg8zbF4sMQUgfydtMUI+uqBXJojAX7ZxFtJJOV+/HzCOJ/wDjh6IOYOFMJQAAAABJRU5ErkJggg==>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAYCAYAAAAoG9cuAAAA3klEQVR4Xo2OPw5BQRDGZy9AJXEEB1CrNBISvTiEQqOlEYlz6EWlxAUk4gga0amfn919b/+9FV8yM998883silRQjgadikcOga4bm5KFUgjOViXi2W2f5y75BwOTG9Q8XSrJ1YRnEV0CXcqBeiRO8FFs6pAeREt3IgvSzYycaU56E237vxlpXI2NU/UpBbyA3+Fr400xQF5RL/JdEJlo1Vp3xNUXwIuYasG8JE/S1rvdg5/RG/7WkDjQ7qkbBkt404zcpbK1iIR6k4U//O2U3BOR02k1J2okD3897UPJB3wMFnrVZIbrAAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAZCAYAAAAFbs/PAAABPklEQVR4XpWRsUoFMRBFJ4riNwhaWNhbaasgVg8sBBsRnmAjthaCiF+gIBa2gr3YWVkpiIIfYPFaBf9Ai+fJ20wy2U0UB2bm5t65k7ArYsLF0mHLYomrUCasGjYkSlFt6PfVIeyTA0pmVxMilVpHVENOaWlxAVm9cJE5JGaZvIF4oL+QB+SYXWpji/wg58N5jvxk6GR0cnJEXVPPIvkF7oVhvfmS8g4eBz+CJ1R9pbzpcDSIHIKG4G3weXrSPw0L5JC8SJNh3Mm+1zg902caUmQTAYPrK5Gulj1plu3YRSuNQXqJizccewOnJSV8TNEH9LOGHAnTSKfw/p98gzfA6wizOsE3d9f0W4R78BVDq8G8S3mCv6NPptd6KZZaqBif3BYMzrZ1DIVozWYopwq8Ch3J8lrskurCv8O68pt/AMFfJ4dAS5o5AAAAAElFTkSuQmCC>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAABR0lEQVR4Xo1Sq05EMRBtySrAENSGoAlZtw7kGvQKvgJPSPgCNIpg+IZduXYDwWxISDBYgsaAQVxO6bTz6l2YZDqnZ2Y6p703hGQxLxQkZbAkirlmlf3DGrWO6lOx3khR3ZXFWIvT16mUjoSPER6BvhE7EM/AC+Al/B24i5nvsB/qVrbUkIr2rR7sThE+gTZUgmwTiS8UPfTcYwBfGa4+zAnJuxTJPSQvSMUW/JZT1EXhKjUjjolN9A2SZ7WmLFIZ4Xtqtj6qNeo6vNkBxkvHueBHCG9GndgwMw35M5xzMhzA73JatbrtdfiVGI84Gbex7IoabjKTX+AfIAb+bpnwsjM6jPnHmNkuUePOm4T8Rz0FftnXmO5pKwvhZf1vmuSV8YFygisja01uj6mEoxqMMH2aku0GSQF9h7aS6lSBHV+a1042yDb/AO+oK24myc4VAAAAAElFTkSuQmCC>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAYCAYAAAAGXva8AAACc0lEQVR4Xo1UO2hVQRCd9YMfrCLYSaxEiEX8lKJiI6kEi5QSFJIqQsoUpnhYiKiFnYVaKWiZQvyACiEJJOmCSREIATUkTaIWCorEM3f23p3dnb3PA+fe3TOfnf0SKTijFaOkx7DzRLGu4JT2LEGjZPR6MmaGSjL0gFZjg8wrEyLo6vRK6IKMGXSpNrfkSjGf4RrQaozM8WyAXnAC/ACugrPgInjd2++BF3yb8QBcAHfAP+A7cArcALfAt+BpcbWrugX+hGkG9sto7/P6brADfZ7t4P46gAG9j6pB3SOtA6fAT+AP8JjXokV8QlLtqFmRoz34fgZf1YLCsJPYAS36VXxKUtBImvUmrBw0kRokd7MFr9Eey3yIXoLfXFiZGug7LvQ3EpzRhiMwfMd/BdyrDYxoLYju4Hc80XkFeO9eJN4n8XkGboNDvvYGfBB4lmNB8kg9LTg6TxI/iTafiYeI+IL/JngbPFw7aiwTB7lq0wWlwWypQ7I1J5R6FJ916DeCFsB7wAG/SE5ogB9AjT8IcrIU0yQzo6Qqvja8tHFejzWSgQ+EkGxKB0nu366qF8w9JCfzeaMIeCC+Jn9dYdAOyaBXY7nJjMRuEv9Lylibr1BzzaJS+fFgvVqBbArAIfAN+BXkQ6FxjqprQv12KPFjwMnPJuaLXl/z/X6Y76Yp+NgPgR9hmCM56vwUXsNg0evjcZ+kSE7MnEK+x5GHo3F8l8D3JP68RdpuIXuTTcdK0nrmY+QpIstWQheforlUqW8X4xRafbrlCRMsuuRoLTQTavzPusceJf+saKOgqFNKlEEn1ZJu65z+o3x8J4sy2q0wgozYf0EwXFa0K6dPAAAAAElFTkSuQmCC>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAABEElEQVR4XmNggANGIARTMAJOoXBwyiMLYEggA2QTUFRC2F5AHAfEGkgyWEESUMN/oC4gZtgF1YxhHBsQCwOxGRA7AIWksVnNDcTucB4UoHueGUhNALJYMGWhAOqxSiDpiyaFAeSBKs+hC8IAsl/qgFQjTBhhK6b9p4HYA10CXRkfEIPCTA8hhGwqQoMqAyRgvVGFMXkcQPwXiKejSkBZaPYfA+L/QLEZQFoNbjW6I4HAEij2jgEav0D2DiDbCiSBogHKUQcSwJhhuAjRwPANiFXgKrFbwZjAAAoJRgZ/MA9FCkLoAAlQnIM8+B2I5bBZDSJAiXUJkH0USCcjTEKxGjWQEQAmjl0Wm2ZcKpFlGBkA3bsdRhWpY9YAAAAASUVORK5CYII=>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABe0lEQVR4Xo1UsS5FQRCdedGIhk/Q8QHyCoVCo9Oo+ANfoFEoJNSiUip0Ep3kJYiOgs4fvEg0aCQi8py7s/fuzOzufU4yd+eec2Z3dvfdR1QB+1wTEYbPCmq5riqL/ZyvKnsMoinmms9QnS0KeqLktV3UcsUErCJGiE/EBPwXxnvEhrj4Go+XoBH9Ih4Rx21xhOniksW8WOhom4LGe2l905bp+A3xpDU12znJIiuJV6PKhyTGI8MKZhAfiFfwA91pDqZ9PCbQ1xPVYY3l7C4cbxGF5nCbjmKwyrvYVX6FxMyTGG86poXs4y50RLTs1AxbJMaD8KYPgWmO5MrHhVay9k5JttOdj3JsNhrezoyiF2tfWH5s34hZ52xwQnIJO351B14i2datVwKYnkk6WnB8Nw7xeED2w3K1Y+QjlusfIK4Q75RusPmEDmN5hGlRf7RTYLfcj2SzC/SXZ86a3RqNtVwXiUz8T25Qm8hSSWFvc5gipkmNsf5Xm6FA/QGMHj5u2dm6/QAAAABJRU5ErkJggg==>
