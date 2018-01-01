# ReadingComprehension_AttNN
<p>Based on the recurrent neural network and its variant, we closely and correctly predict the sentence relatedness.
And the finial score of the three models lists as follows:</p>
----------
<p>Model name Peason Score  MSE score</p>
<p>SimpleLSTM 0.215939774023 0.265131676145</p>
<p>SimpleGRU 0.21431587721 0.36135241366</p>
<p>Att-RNN 0.445342835253 0.311997173107</p>
<p>Att-GRU 0.696281333858 0.145266796306</p>
<p>Att-LSTM 0.606448015981 0.157967669988</p>
<p>We use the BCELoss (Cross Entory) to calculate the relatedness of the model.The score is properly described by BCELoss function .</p>
 <p>Addtionaly, We use the mean loss to evelucate the model.And some of the other information see the code.</p>
----------
<br>## Note! </br>
<p>The model must be placed on a workstation with an integrated GPU and trained to automatically select free GPU resources to prevent resource preemption.</p>
<p>The application the model trained on is a machine named PSC-HB1S.</p>
<p>The details of the application are following below:</p>
<p>Four CPUs: Intel® Xeon® E5-2600 v3/v4</p>
<p>Four GPUs for cuda environment</p>
<p>To get the details of the trainig processing and eveluation ,you see the essay. </p>
