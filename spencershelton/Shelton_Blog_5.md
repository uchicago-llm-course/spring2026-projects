# Revising Modularity Paper

In response to feedback, some of the figures from the draft paper have been revised.

Figure 4 currently plots loss, exact move accuracy, and top-4 token accuracy on the same y-axis making the visual comparison confusing as the metrics have different units and therefore different interpretations with lower loss being better, while higher accuracy is better. The cleaner version splits the metrics into separate panels with their own axes with the caption stateing whether the plotted value is a gain, a drop, or an absolute metric.

Figure 2's current schematic can make it look like the shared layers are a separate component from the transformer rather than the non-modular parts of the same transformer stack. The revised version labels the backbone and modular insertions more explicitly with the goal being to show that the model is not a collection of independent models; it is a shared transformer with selected modular capacity at defined points.

The discussion section has been and will continue to be revised on how to achieve stronger modularity. The current results show partial functional localization without full clean symbolic isolation. It should be noted that this is somewhat anticipated and even somewhat the desired result, the goal was more the functional result of in place changes to the model having lowered side effects, but this will be made more cclear in the discussion.

## Extra Testing and Refinement

Currently I am additionally working on training a chess transformer without the additional training time modularization pressure from switching rules to show how that model reacts when modification is attempted to target a different ruleset. The training is currently running on the inital model and should be done by tomorrow morning. This will be added to the results and discussion with some more figures to show training requirement differences to train a new rule and any regresssion in general performance after training the new rule.