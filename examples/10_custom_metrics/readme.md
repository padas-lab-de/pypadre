# Writing custom metrics for components in the pipeline
In a scenario where the user needs custom metrics, it is possible to add that to the existing framework.
For that, the user needs to follow the following steps
- Write a function that computes the metrics and 
saves the metrics as a dictionary with the key being the metric name. The function should return a metric object
 with the result as the computed metric
- Write a wrapper class that provides a name and also what the class consumes, for example confusion matrix
- Define the reference to the metric for tracking the metric source code
- create an object of the metric class and add it to the metric_registry