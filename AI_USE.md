# AI Use Statement

This document describes how AI tools were used in this project, and where they were not. All modeling decisions, analysis, and conclusions are our own. However, we did utilize AI significantly in our process, especially considering the complexity of our project, oftentimes to clarify our results or understand our data better. As such, we have provided a comprehensive summary of where exactly AI was used and how we used it to perform rigorous science.

**Tools used:** Claude Code (Sonnet 5, Medium Effort), GPT 6 Astra

## Summary

| Area | AI used? |
|---|---|
| Understanding the data | Yes |
| Graph styling, titles, axis labels | Yes |
| Displaying results from model runs | Yes |
| Editing report diagrams | Yes |
| Grammar and wording in the report | Yes |
| Debugging code | Yes |
| Structure (layout and headings) of documentation files such as `README.md`, `AI_USE.md`, and markdowns in Jupyter Notebooks | Yes |
| Model ideation | **No** |
| Model implementation | **Yes** |
| Choosing which graphs to make and what they analyze | **No** |
| Interpreting figures and deciding what to report | **No** |
| Content of documentation files | **Yes** |

## Where AI was used

### 1. Understanding the data
AI helped us get oriented in the dataset: what each column means, how the files relate, and what the value ranges and types look like. In particular, it was useful for correlating what Emily, one of our advisors, provided us in our task document with what was actually present in the referenced files. We used it to speed up exploration, but not to decide how the data should be used.

> *Example prompt:* "Here is the output of `df.describe()` and `df.info()`. What do these columns represent, and are there any that look like they have unusual values or missing data?"

> *Example prompt:* "Here, we can see the outliers in the data. Some of them are extreme values, but others (such as in energy flux) are negative. Is it possible for these values to be negative?"

### 2. Visual style, titles, and axes of graphs
We decided what each graph shows. AI helped with presentation: color choices, font sizes, legends, titles, and axis labels.

> *Example prompt:* "This matplotlib plot has overlapping x-axis labels and a legend that covers the data. How do I fix the layout and make the title and axis labels clearer?"

> *Example prompt*: "This graph has clear results, but is not labeled clearly. Can you move the legend and select visually-distinct colors to make the comparison more obvious?

### 3. Displaying data and results from model runs
After running models ourselves, we used AI to help format and present the results, such as building summary tables and plotting metrics in a readable way.

> *Example prompt:* "I have a dictionary of accuracy and F1 scores for each model. Help me turn it into a clean comparison table and a bar chart."

### 4. Editing diagrams for the report
AI helped clean up and revise diagrams (for example, figures that were previously) that appear in the report. The content of each diagram reflects our own design.

> *Example prompt:* "Here is the source for my graph of model results. Can you give me the LaTEX code to place this into a good place in the report, so that it is next to the analysis that is present for it?"

### 5. Grammar and wording in the report
We wrote the report. AI was used for grammar fixes, clarity, and tightening wording. We reviewed each suggested edit before keeping it.

> *Example prompt:* "Fix grammar and improve clarity in this paragraph, but keep my meaning and don't add new claims."

### 6. Debugging
When code raised errors or behaved unexpectedly, we used AI to help diagnose the problem, such as shape mismatches, indexing mistakes, and library usage errors.

> *Example prompt:* "I'm getting `ValueError: Found input variables with inconsistent numbers of samples` when fitting this model. Here is the traceback and the relevant code. What's causing it?"

### 7. Structure and content of documentation files
AI helped organize sections, headings, and formatting in files such as `README.md` and `AI_USE.md`. We also used AI to draft some of the text.

> *Example prompt:* "Draft a structure for an AI_USE.md that lists where AI was and was not used in a data science project, with example prompts."

> *Example prompt:* "I have written this function that defines a pipeline for a Random Forest model and tests it on the training and validation datasets provided. Can you provide a short title and description for the Jupyter Notebook to make it easier to understand?"

### 8. Model implementation
We used AI to help write code for the models and architectures we chose.

## Where AI was not used

- **Model ideation.** We chose the models, features, and architectures to test, including MLPs, XGBoost, random forests, and neural network ensembles.
- **Choosing the graphs and their purpose.** We decided which visualizations to produce and what question each one should answer. This meant that we independently verified each result that was produced by our code and determined which ones were key to include into our report.
- **Interpreting results.** We analyzed the resulting figures ourselves and decided what findings belong in the report. AI did not draw the conclusions. Our process was: we looked at each figure, wrote out our own interpretation as complete-sentence bullet points, and then asked AI to join those bullets into paragraphs (adding transitions and connecting words). AI also sometimes asked clarifying questions about our points, which we answered ourselves. It did not add findings, claims, or explanations that were not in our bullets.

## Verification

We read and tested all AI-assisted output before using it.

- **Code.** Every AI-suggested code change was tested by us. We worked primarily in Jupyter Notebooks, so each change was executed in a cell right away. We compared the output against the results we expected and fixed anything that did not match before moving on. No AI-suggested code was kept without being run.
- **Text.** We reviewed every wording edit to make sure it preserved our intended meaning.
- **Documentation.** We reviewed the AI-assisted documentation for accuracy.
