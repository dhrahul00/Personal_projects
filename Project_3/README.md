<p><strong>Hexadecimal Image Captcha classification using CNN  </strong></p>
<ul>
<li><strong>Model &ndash; </strong>
<ul>
<li>ResNet_18</li>
</ul>
</li>
<li><strong>Tools &ndash; </strong>
<ul>
<li>Python</li>
<li>PyTorch and Sklearn</li>
<li>Matplotlib</li>
<li>Opencv</li>
<li>Numpy & Pandas</li>
</ul>
</li>
<li><strong>Data &ndash; </strong>
<ul>
<li>2000 captcha images - input</li>
<li>2000 labels in a csv file &ndash; target</li>
</ul>
</li>
<li><strong>Train &amp; Test split &ndash;</strong>
<ul>
<li>Test data &ndash; 1800</li>
<li>Train data &ndash; 200</li>
</ul>
</li>
<li><strong>Description&ndash; </strong>
<ul>
<li>Data is loaded by data-loader with 32 batch size.</li>
<li>ResNet_18 CNN model is trained for this problem.</li>
<li>Adam optimizer and Binary cross entropy loss are considered as optimizer and loss function.</li>
<li>With 7 epoch 100% accuracy is achieved in both test and train samples</li>
<li>Predicted result is compared with actual result by visualization (green means they are same red indicates they are different)</li>
</ul>
</li>
<li><strong>Reference &ndash; </strong>
<ul>
<li><a href="https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624">https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624</a></li>
<li><a href="https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/">https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/</a></li>
<li><a href="https://arxiv.org/abs/1512.03385">https://arxiv.org/abs/1512.03385</a></li>
</ul>
</li>
</ul>
