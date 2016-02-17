addpath matlab
caffe.reset_all();

 models= '../temporal_kd_TH14_test.prototxt';
 weights = '../temporal_model_TH14_kd/_iter_90000.caffemodel'
 net = caffe.Net(models,'test')
 net.copy_from(weights);
 conv1 = net.params('conv1',1).get_data();
 a=conv1;

 models= '../temporal_cls_zbw.prototxt';
 weights = '/home/bwzhang/TH14_mvs_standard.caffemodel'
 net = caffe.Net(models,'test')
 net.copy_from(weights);
 conv1 = net.params('conv1',1).get_data();


