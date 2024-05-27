<h2>Anomaly Detection using Isolation Forest and Local Outlier Factor (LOF)<br /><br /></h2>
<h3>Data Overview</h3>
<p>The dataset includes the following columns:</p>
<ul>
<li><strong>Timestamp</strong>: Time of occurrence of the event.</li>
<li><strong>traceID</strong>: Unique ID of an execution path through the system.</li>
<li><strong>spanID</strong>: Unique ID associated with the execution of a logical unit.</li>
<li><strong>parentSpanID</strong>: spanID of the parent span that called a given span.</li>
<li><strong>serviceName</strong>: Name of the microservice associated with the span.</li>
<li><strong>Name</strong>: Method/function/endpoint name associated with the span.</li>
<li><strong>durationNano</strong>: Time taken in nanoseconds to execute the span.</li>
</ul>
<h3>Algorithms Used</h3>
<ol>
<li>
<p><strong>Isolation Forest</strong>: An algorithm specifically designed for anomaly detection. It isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The logic is that anomalies are few and different, and they are more susceptible to isolation.</p>
</li>
<li>
<p><strong>Local Outlier Factor (LOF)</strong>: LOF measures the local density deviation of a given data point with respect to its neighbors. It considers points that have a substantially lower density than their neighbors as anomalies.</p>
</li>
</ol>
<h3>Results</h3>
<ul>
<li><strong>Isolation Forest</strong> detected several anomalies characterized by unusually high or low execution durations.</li>
<li><strong>Local Outlier Factor</strong> identified similar anomalies, with some additional points flagged due to localized density deviations.</li>
<li>Both models provided complementary insights, confirming the presence of performance anomalies in the microservice traces.</li>
</ul>
