# Final Project Instructions

## CSE 6730, Spring 2026

**Your goal:** Your final project is to build your very own computer simulation, a.k.a. a simulation model.
That sounds vague, but hang in there.
If you choose well and devote yourself to it, it will be more fun than you think, and it's a great opportunity to enhance your project portfolio for your future job search with something that might help you "stand out" from the crowd!

**Scope: Simulation models vs. data models.** As a reminder, you can divide the world of models into two broad camps: simulation models and data models.

* In a simulation model, you design a set of rules that express how you believe the real-world system behaves.
You can think of these rules as capturing your prior knowledge about the system.
* In a data model, you use data to try to learn the model.
You minimize your use of prior knowledge, letting the data tell you what specific form the model should take.

In reality, a model may be a combination of these two styles.
For instance, the rules may be parameterized, and you use data to estimate those parameters.
Or, you might design a data model to inspect its structure and infer the rules governing its behavior.

Your project should focus on __creating a simulation model rather than a data model__.
Given the hybrid nature of real-world modeling, it doesn't mean you can't use data in your project.
But since it's a class about simulation and numerics, your project should _skew its emphasis accordingly_.

The structure of a typical project has these components.

* (**Inspiration**) It starts by "taking inspiration" from natural or engineered time-varying systems from the world at large.
It's often helpful to pose one or more questions about a system you think a simulator could help answer.
Of course, real-world systems are complicated!
Part of the "art" of your project is deciding what aspects of elements of that system you will focus on modeling and simulating.
 
* (**Model**) It then develops an abstract model of this system's behavior.
Think of this phase as the "rule generation" part or "conceptual modeling" task.
Data might inform your design.
 
* (**Simulator**) Code up your simulator to create a computer simulation of this abstract model.
Your simulator __needs to incorporate some of the numerical methods and analysis we covered in class__.
Some projects naturally lend themselves to a team developing additional features, such as a front end for interacting with the simulation or a backend visualization tool.
 
* (**Experiments**) Conduct computer experiments with your simulator.
Your goals are to validate your model and, assuming that the process looks reasonable, try to answer the questions you posed about the system or phenomena of interest.
You should use data in some way, either qualitatively to inform the design of your abstract model or quantitatively as part of your experiments and validation.
Data may again play a role in validating your results.
 
* (**Reporting**) To help us understand what you did, you'll write up a report and what you did and give a short presentation that explains your project.

**Teaming**

You'll work in small teams of _3-6 students each_.

Inevitably, conflicts within the team may arise. Please communicate with one another, and if needed, work with the teaching staff to resolve problems early on. The deadline for "divorce" is Checkpoint 2 -- after that point, we will consider all teams "final," and you will just have to "make it work" until the end.

**Milestones and grade weights**. The following list summarizes what you need to turn in and when. For "how" to submit, see later sections of this document. *(The percentage values shown indicate the fraction of your final course grade and add up to 50%.)*

1. (1%) Form project teams: You need to form a team and "declare it" on Canvas.  
2. (2%) Literature survey: You need to search for papers, books, or materials related to your project topic and summarize them in a 1- or 2-page review.  
3. (3%) Project checkpoint 1: You need to show preliminary progress on your project by explaining your abstract or conceptual model and any preliminary progress on implementing this model.  
4. (4%) Project checkpoint 2: You need to show that you have made significant progress on implementing your simulator. If you've hit any snags, you need to describe them at this point.  
5. (35%) Final project report: Your project is due on the date in the syllabus. You will need to submit a written report and your fully *documented and commented* simulation code (via your GitHub repository above).  
6. (5%) Final project presentations: Presentation of final projects. Your group will present your project on one of these two dates. You have 5 minutes per group. Everyone in the group should present at least part of one slide; otherwise, you can divide the presentation however you like. 

**Where to find some inspiration!**

* [Sayama's book (2015)](https://textbooks.opensuny.org/introduction-to-the-modeling-and-analysis-of-complex-systems/) -- lots of examples, exercises, and mini-projects that you can go deeper into or extend  
* [Scientific computing with case studies](https://epubs.siam.org/doi/book/10.1137/9780898717723), by Dianne O'Leary (2009), SIAM.  
* [Mathematical modeling of zombies](https://people.maths.ox.ac.uk/maini/PKM%20publications/384.pdf), by Robert Smith? (2014). You might find inspiration from some of the mathematical techniques for conceptual modeling that Smith? applies to consider various aspects of how zombies might behave "in the real world."  
* [The structure and dynamics of networks](https://ebookcentral.proquest.com/lib/gatech/detail.action?docID=787357), by Mark Newman, Duncan J. Watts, and Albert-László Barabási (2006).  
* [Mathematical models: mechanical vibrations, population dynamics, and traffic flow](https://epubs.siam.org/doi/book/10.1137/1.9781611971156), by Richard Haberman (1998), SIAM.  
* [Tutorial on the dynamics of biological neurons](http://jackterwilliger.com/biological-neural-networks-part-i-spiking-neurons/) (spiking models) by Jack Terwilliger (2018).  
* [A primer on mathematical models in biology](https://epubs.siam.org/doi/book/10.1137/1.9781611972504?mobileUi=0), by Segel & Edelstein-Keshet (2013).  
* *The nature of code*, by Daniel Shiffman. [Section 7.9](https://natureofcode.com/book/chapter-7-cellular-automata/) has an interesting list of application ideas for cellular automata (as does Sayama's book).  
* *Traffic and related self-driven many-particle systems*, by Dirk Helbing. [Link via [GT Library Proxy](https://journals-aps-org.prx.library.gatech.edu/rmp/pdf/10.1103/RevModPhys.73.1067)]  
* *Programmable matter*:  
  * Talk by Dana Randall @ GT -- [https://www.youtube.com/watch?v=nPCjWIoK5KI](https://www.youtube.com/watch?v=nPCjWIoK5KI)   
  * A recent article on self-organization in this context -- [https://science.sciencemag.org/content/371/6524/90](https://science.sciencemag.org/content/371/6524/90)   

**Evaluation**. Roughly speaking, when we assess your project report (and summary presentation) and code artifact, we'll be considering these dimensions:

* Exposition: How well do the report and presentation explain the problem(s) and solution technique(s)?
Is it concise but also clear, readable, and precise?
Does a prospective reader take away insight?
 
* Code: Is it clear and readable?
Does the implementation pay attention to efficiency considerations?
 
* Results: Does the project contain simulation examples to help illustrate the problems and techniques?
Does it use visualization appropriately?

There is no hard "weighting" among these categories, but do try to pay equal attention to all criteria.
Your report should highlight where you put a lot of effort and acknowledge any weak points in what you did.

Peer assessment.
The entire team receives the same base grade.
However, at the final submission, we may ask each of you to submit your own "peer assessment" in which you evaluate the contributions of your teammates.
In the event of discrepancies, a small additional component of the grade may be based on teammate evaluations.
This assessment aims to create an incentive for you to find ways to work well together and contribute equally to the overall product.

For all milestones: You will use the GT GitHub instance (github.gatech.edu) to submit some of your milestones.
Start by creating an initial repository there to hold the materials for you and your teammates.
You can make this private if you wish, but then you will need to share it with all of the teaching staff so we can grade it.

## Team Formation

For this milestone, you just need to "declare" your team.
Go to Canvas and complete the following two steps:

1. Under "People," look for the "Groups" or "Project Groups" tab, find an empty Project Group, and sign yourselves up!  
2. Under "Assignments," submit the "form project teams" assignment. List your teammates and contact info. Optionally, if you have a topic idea already, please mention that.

## 1st milestone: Literature Review 

Write a short document summarizing what others have done related to your project for your literature review (1-2 pages, plus unlimited additional space for references).
That is, are there existing models, simulators, or simulation techniques?
Will you adopt one of these approaches as a basis for your project?

* What will you do that will be new or different from what you've found?
You don't have to do completely new research.
Still, if you are attempting to replicate an existing simulator or simulation study, you should distinguish what you will attempt to do from what has been done before.

## Checkpoint 1

**Submission**: A single PDF document, which includes a pointer to your GitHub repo. 
There are three components for the submission for the checkpoint:
   
**1. A clear and detailed description of your project (about 2-4 pages, excluding references).**
At this stage, you should have a clearer idea of the project, and the details you expect will be part of the final submission.

Some things to include:

* An abstract summarizing the system and the goals of the project  
* Description of the system being studied  
* A conceptual model of the system  
* Platform(s) of development  
* Literature review (paste in what you submitted before, possibly updated if you discovered new things in the interim -- the literature review does not count against the 2-4 page guideline)  
   
**2. An update of the current state of the project and initial results, if any (max 2 pages)**

Some things to include:

1. A "show of progress" via some working code, analysis, or initial modeling attempts
2. If there have been any major changes in direction or "course corrections" since your original proposal, you can describe them here.
3. Division of labor: How will you divide up the remaining work among your team?
In particular, we will be looking to see that you've given thought to how to ensure your project justifies a multi-person effort.
   
**3. A GitHub repository for the final submission**

This checkpoint component will ensure you have set up a git repository on the Georgia Tech GitHub repository.
This is where you will be sharing your final project implementation with the instructors.
   
Things we will evaluate:

* A GT GitHub link to the project repository is included in the report  
* The repository has correct permissions that are shared with all the teammates, instructors, and TAs. **See step 8 below.**  
* At least one file, e.g., README or the Project Checkpoint report, is in the repository.
 
__Note: Although we recommend it, you don't have to use git for your development. You just have to share your work with us on GitHub.__
 
In case you have not interacted with GT GitHub or Git in general. The following is a brief tutorial for setting up a GT git repository and pushing (read, saving) your local copy on the GitHub server.

1. Log on to GT GitHub ([https://github.gatech.edu/](https://github.gatech.edu/)) using your GT credentials.  
2. Name your new repository (or "repo").  
3. Choose the privacy setting for the repo. If you choose to make it public, anyone with a link and a GT account would be able to access your repo and its contents. This can be updated later.  
4. Once the repos are set up, work as usual in your project directory. When it is time to push changes:  
5. `git add --a`
  to add all the new files to git. 	                            	      
  **ProTip**: use git status to view the current state of all files added to git or updated since the last commit.  
5. `git commit -m second commit`
6. `git push`
7. Collaborators can be added through the settings section on your repo.  
   [https://github.gatech.edu/](https://github.gatech.edu/)
   `<gt_username>/<repo_name>/settings/collaboration`

    **ProTip**: You can search and add people using GT usernames.

8. Add **the following TAs** *and* **Spencer** as collaborators on your repo. Their IDs are:  
  * sbryngelson3 - Spencer Bryngelson  
  * bwilfong3 - Benjamin Wilfong
  * aradhakr34 - Anand Radhakrishnan
  * araghavan68 - Aditya Raghavan 
  * yqiu327 - Yuan Qiu
   
You are always welcome to request a discussion with the TAs via Piazza in case you find something unclear.

## Checkpoint 2

We want to see that you have made significant progress toward creating your simulator for this checkpoint. The TAs will inspect your code repos to see how much progress you've made. Please submit a short (1- to 2-page) summary of how you've divided the work, what you have completed, and what work remains.

## Final Report 

Here is what you need to do for your final project submission.
There are two main parts. I highly recommend you follow them closely!

### Part 1

Each team will upload a single PDF into the assignment on Canvas named __Final project__  for your final project submission.
This PDF should contain the following information.

* A single cover page that shows the title of your project or tutorial, your team number, a list of your team's members, and a link to your GitHub repo so we can verify what you implemented. (Presumably, this repo is the same as what you indicated in your Project Checkpoint.)  
* The project report or tutorial itself  
* After the report or tutorial contents, provide a brief description of how you split the work among the team members.

Please also place a copy of this PDF in your repository.

Suggested outline: Most of you are doing projects that involve simulating a real-world system.
In such cases, here is a suggested outline for your report.
(The top-level bullets are suggested section headings for the report; the sub-bullets are only there to explain what goes in the corresponding section.)

* **Abstract**
  * 100-200 word summary of the overall project  
* **Project description**
  * What is the goal of your project?  
  * If you are modeling some real-world phenomenon, what aspects of that phenomenon are most relevant to capture in your model?  
  * Assume a reader knows little about this phenomenon. Use clear language to explain it, minimize jargon, and define terms as needed. Illustrations or sketches are always helpful!  
* **Literature review** (include from your earlier submission)  
* **Conceptual model**
  * Describe the abstract or mathematical (conceptual) model you have developed. Clearly explain how the model's features reflect the phenomenon of interest as described under the "project description."
* **Simulation/simulator/simulation model**
  * Summarize the simulation and numerical methods you implemented. Include all details about why you chose various methods, why they were a good choice, and what choices you could have made but didn't because you didn't have enough time or they just weren't good choices for you use case.
* **Experimental results and validation**
  * Explain what studies you did that used your simulator. Clearly describe your experimental procedure.  
  * How did you attempt to validate the simulator? Justify how you modeled the inputs to the program.  
  * Clearly explain how you analyze your outputs. Report confidence intervals as appropriate.  
* **Discussion, conclusions, summary**  
  * What did you learn about the system you were modeling and simulating?  
  * What would you suggest they do if someone were to extend this work?  
* **References** (from your literature survey)  
* **Appendix: Division of labor**  
  * Briefly explain how you divided the work. Who did what?

### Part 2

**Separate** from the above submission, which only needs to be done once for the team, each of you must **individually** submit a teammate assessment (see Canvas). It is simply a text entry box in which you will enter the following information:

* Your team number and list of teammates  
* Assign a letter grade (A through F) to yourself and each of your teammates based on assessing their contributions to the project. In each case, explain your grading. (Note: Do not interpret this grade as one for your project; instead, think of it only as a grade based on the effort expended to complete the project.)  
* We will not share your teammate's assessment with anyone else on your team. It will only be used for us to identify issues and try to resolve them after the fact.

### Final Presentation

You will present your work with your teammates in 5 minutes. Upload your slides to Canvas in the "Final Presentation" assignment. Find more info on the presentation in the course syllabus.

## AI / Generative Tools Policy (CSE 6730: Modeling & Simulation — Spring 2026)

This course emphasizes **conceptual modeling, numerical methods, and implementation**. AI tools (e.g., ChatGPT, Claude, Copilot, Gemini, CodeWhisperer, etc.) can be helpful for learning and productivity, but they can also short-circuit the skills you’re here to develop. This policy aims to allow **responsible use** while protecting **academic integrity**, **fair grading**, and **deep learning**.

### 1) Core principles
1. **You are responsible for your work.** Anything you submit—text, math, code, figures, citations, results—must be understood by you and defensible in discussion.
2. **AI output is not automatically correct.** Treat AI as an *unreliable assistant* that can hallucinate math, citations, and code behavior.
3. **Transparency is required.** If you use AI meaningfully, you must disclose how.
4. **Learning objectives come first.** If AI use replaces the thinking the assignment is designed to assess, it’s not allowed.

### 2) Definitions (for this course)
- **Generative AI / AI tools**: systems that generate or transform text, code, images, audio, or data (LLMs, coding copilots, AI summarizers, etc.).
- **Meaningful use**: any use that contributes directly to the content you submit (e.g., generating code, rewriting paragraphs, generating derivations, proposing model equations, writing experiment plans, producing plots/figures, or summarizing papers that you cite).

### 3) Quizzes (individual; open-book/open-internet)
Quizzes are **individual assessments**. They are open-book/open-internet, but they are **not collaborative** (including “collaboration” with an AI that is effectively solving the quiz for you).

**Allowed on quizzes**
- Looking up course notes, textbooks, documentation (NumPy/SciPy/PETSc/etc.), and general reference material.
- Using AI for *low-level help* that does not provide the solution:
  - Clarifying a concept or definition (“What is conditioning?”)
  - Explaining syntax errors or language/library usage (“Why does `solve_ivp` complain?”)
  - Finding relevant documentation pages or examples
  - Debugging help where you supply your own code and reasoning

**Not allowed on quizzes**
- Asking AI to **solve quiz questions**, derive the main steps, compute final numeric answers, or choose multiple-choice answers.
- Copying AI-generated solutions, reasoning, or code that effectively substitutes for your own work.
- Uploading quiz questions verbatim into an AI system and requesting answers.

**Rule of thumb:** If an AI could plausibly answer the quiz for you, that usage is **not allowed**.

### 4) Final project (team-based simulation project)
The final project is a **team effort** with **individual accountability**. You may use AI tools in ways that support implementation and writing—*as long as the team’s work remains original and you document the use*.

#### Allowed uses (project)
You may use AI for:
- **Brainstorming** modeling approaches, numerical methods, validation strategies, or experiment design (with team judgment).
- **Implementation support**
  - Debugging, refactoring, writing tests, performance tips
  - Example usage of libraries (SciPy, sparse solvers, plotting libraries)
  - Boilerplate (CLI parsing, plotting scaffolds, README structure)
- **Writing support**
  - Editing for clarity, grammar, structure
  - Producing outlines or suggesting section organization
- **Documentation support**
  - Generating docstrings or comments from your own code (you must verify correctness)

#### Not allowed uses (project)
You may **not** use AI to:
- Submit a project where the **core conceptual model** (rules/equations/assumptions) was generated by AI and you cannot justify it.
- Generate a full simulator or substantial module that your team did not design, understand, and validate.
- Fabricate or “auto-generate” **citations** you have not read and verified.
- Produce results/plots without being able to explain **exactly** how they were generated and validated.
- Circumvent the requirement to build a **substantial amount of original simulation code** (i.e., using AI to effectively outsource the main deliverable).

**Expectation:** AI can accelerate work, but the project must clearly reflect your team’s **own modeling decisions**, **numerical method choices**, and **software implementation**.

### 5) Required disclosure (project + written submissions)
For any milestone report, the final report, and the final repository, include an **AI Use Statement** if you used AI meaningfully.

**Minimum disclosure requirements**
- Tool(s) used (e.g., ChatGPT, Copilot)
- What you used it for (e.g., debugging, code refactoring, writing edits, brainstorming)
- Which files/sections were materially affected
- What you did to verify correctness (tests, cross-checks, unit tests, derivations, citations)

**Where to put it**
- Reports (literature review, checkpoint PDFs, final report): include a short **AI Use Statement** near the end (e.g., after references or in an appendix).
- Code repository: include a short section in the **README** (or a separate `AI_USAGE.md`) and, if relevant, brief file-level notes in comments or commit messages.

#### Suggested AI Use Statement template (copy/paste)
> **AI Use Statement (required if applicable).** We used [Tool(s)] to assist with [tasks]. Specifically:  
> - **Code:** [describe modules/files, e.g., “refactoring solver loop in `solver.py`, debugging `solve_ivp` usage, generating docstrings”].  
> - **Writing:** [e.g., “editing for clarity in Sections 2–4; outline suggestions”].  
> - **Verification:** We verified all AI-assisted outputs by [tests/derivations/benchmarking/reading cited sources], and we take responsibility for correctness and originality.

### 6) Citation and attribution rules (project reports)
- You must cite **all external sources** used in your report (papers, books, websites, documentation, repos).
- If AI helped you find sources, you must still **read them** and cite the **original source**, not the AI.
- Do **not** cite AI as an authoritative source for scientific claims. If you must mention AI use, do so in the AI Use Statement, not as a bibliographic reference (unless you are explicitly studying the AI system as part of the project, which is unusual here).

### 7) Data, privacy, and intellectual property
- Do not upload private, sensitive, or proprietary data/code to third-party AI tools unless you have permission and understand the tool’s data policy.
- If you use AI tools on codebases or datasets with restrictive licenses, you are responsible for complying with those licenses.

### 8) Enforcement and integrity
Violations of this policy are treated as **academic integrity** issues and may be reported to the Office of Student Integrity consistent with Georgia Tech policy.

When evaluating suspected violations, course staff may:
- Ask you to explain your code/model/results in detail.
- Ask for intermediate artifacts (git history, notes, drafts, experiment logs).
- Ask for short oral checks or live walkthroughs of parts of the implementation.

**Best practice:** Commit early/often, keep notes, and write tests—this protects you as much as it helps grading.

### 9) Practical guidance (how to use AI responsibly)
AI can be useful if you treat it like:
- A **documentation assistant** (“show me a minimal example of sparse CG with preconditioning in SciPy”)
- A **debugging partner** (“what does this error mean and how do I test the fix?”)
- A **writing editor** (“make this paragraph clearer without changing technical meaning”)

Avoid using it like:
- A **solution engine** for quizzes
- A **model designer** you can’t justify
- A **citation generator**

