<p><strong>Deep Learning with PyTorch: GradCAM:</strong></p>
<ul>
<li><strong>Data description &ndash; </strong>
<ul>
<li><strong>Images &ndash; 186</strong></li>
<li><strong>Class &ndash; 3</strong></li>
<li><strong>Each Image &ndash; 3X227X227 dimensional</strong></li>
</ul>
</li>
<li><strong>Model Description &ndash;</strong>
<ul>
<li><strong>Feature_extractor -</strong>
<ul>
<li><strong>Convolution block &ndash; </strong>
<ul>
<li><strong>5X5 convolution layer</strong></li>
<li><strong>ReLu layer</strong></li>
<li><strong>4X4 Maxpool layer </strong></li>
</ul>
</li>
<li><strong>Convolution block is repeated 4 times.</strong></li>
</ul>
</li>
<li><strong>Linear layer &ndash; </strong>
<ul>
<li><strong>Linear layer 6400X2048</strong></li>
<li><strong>ReLu layer</strong></li>
<li><strong>Linear layer 2048X3</strong></li>
</ul>
</li>
<li><strong>Job Description: </strong>
<ul>
<li><strong>How the gradient is changing for each class of the images.</strong></li>
<li><strong>For each class a bar plot is created to check the prediction of the model.</strong></li>
<li><strong>A Heatmap is generated to check the gradients.</strong></li>
</ul>
</li>
</ul>
</li>
</ul>
<ul>
<li><strong>Results and Discussion &ndash;</strong>
<ul>
<li><strong>Loss &ndash; </strong>
<ul>
<li><strong>Train loss &ndash; 0.00</strong></li>
<li><strong>Validation &ndash; 0.08</strong></li>
</ul>
</li>
<li><strong>Heatmap &ndash;</strong>
<ul>
<li><strong>The actual class identification in gradients are identifies in red colour.</strong></li>
</ul>
</li>
</ul>
</li>
<li><strong>Reference &ndash; </strong>
<ul>
<li><strong>https://coursera.org/share/e2b10b2507391044890dfe2bff2e1737</strong></li>
</ul>
</li>
</ul>
