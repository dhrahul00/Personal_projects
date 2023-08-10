<p><strong>Multi-Class classification with Pytorch :</strong></p>
<p><strong>Description:</strong></p>
<p>&nbsp;In this project a RNN model is built to analyse text and predict an emoji accordingly.</p>
<p>A small training set is used to map the classifier text to emojis.</p>
<ul>
<li><strong>Model:</strong></li>
</ul>
<ul>
<li><strong>Description:</strong>
<ul>
<li>RNN model
<ul>
<li>Embedding layer &ndash; Embedding (400001, 50)</li>
<li>LSTM layer &ndash; LSTM (50, 128, num_layers=2, batch_first=True, dropout=0.5)</li>
<li>Linear layer &ndash; Linear (in_features=128, out_features=64, bias=True)</li>
<li>Droupout -Dropout (p=0.5, inplace=False)</li>
<li>Linear layer &ndash; Linear (in_features=64, out_features=5, bias=True)</li>
</ul>
</li>
</ul>
</li>
</ul>
<ul>
<li><strong><a href="https://www.kaggle.com/datasets/alvinrindra/emojify/download?datasetVersionNumber=2" target="_blank">Data</a>&nbsp;Description:</strong>
<ul>
<li>Embedding - <a href="https://www.kaggle.com/datasets/watts2/glove6b50dtxt/download?datasetVersionNumber=1" target="_blank">glove 6b 50d</a></li>
<li>Train Data &ndash;
<ul>
<li>Sample &ndash; 131</li>
</ul>
</li>
<li>Test Data &ndash;
<ul>
<li>Sample &ndash; 5</li>
</ul>
</li>
<li>Class &ndash;
<ul>
<li>Sample - 4 {0: '❤️', 1: '⚾', 2: '😄', 3: '😔', 4: '🍴'}</li>
</ul>
</li>
</ul>
</li>
</ul>
<ul>
<li><strong>Train Accuracy:</strong>
<ul>
<li>Train accuracy 95% (may vary)</li>
</ul>
</li>
<li><strong>Test Accuracy: </strong>
<ul>
<li>Test accuracy 71% (may vary)</li>
</ul>
</li>
</ul>
<ul>
<li><strong>Epoch:</strong>
<ul>
<li>50</li>
</ul>
</li>
</ul>
<ul>
<li><strong>Optimizer:</strong>
<ul>
<li>Adam &ndash;
<ul>
<li>Learning rate = 0.001</li>
<li>L2/weight_decay = 0.01</li>
</ul>
</li>
</ul>
</li>
</ul>
<ul>
<li><strong>Loss Function:</strong>
<ul>
<li>Cross entropy loss</li>
</ul>
</li>
</ul>
<ul>
<li><strong>Miss labeled :</strong><br />
<ul>
<ul>
<li>In training data -</li>
</ul>
</ul>
<a href="https://imgbox.com/VZLGfJkF" target="_blank"><img src="https://thumbs2.imgbox.com/9c/ee/VZLGfJkF_t.png" alt="image host" /></a></li>
</ul>
<ul>
<li><strong>References:</strong>
<ul>
<li><a href="https://www.deeplearning.ai/program/deep-learning-specialization/" target="_blank">https://www.deeplearning.ai/program/deep-learning-specialization/</a></li>
</ul>
</li>
</ul>
