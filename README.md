# miccai17-rnn-theano
This is the implementation of BiLSTM for our project about semantic segmentation in 3D ultrasound volumes.

BiLSTM here is developed to refine the semantic segmentation results from 3D FCN's prediction.

With our proposed cube-wise serialization and deserialization manner, we prove that BiLSTM can be more preferable than DenseCRF and the cascaded Auto-Context scheme in refining the labelling results for volumetric ultrasound segmentation.

More information can be found from the project page and the published conference paper:http://appsrv.cse.cuhk.edu.hk/~xinyang/3d_is_proj/3d_is_proj_index.html.

The implementation is originally adapted from uyaseen:https://github.com/uyaseen/theano-recurrence.

Library: Theano.
