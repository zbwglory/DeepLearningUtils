 clear;
 caffe.reset_all();
 models= '../temporal_zbw.prototxt';
 weights = '../temporal_kd_adapted_list01_mvs.caffemodel'
 net = caffe.Net(models,'test')
 net.copy_from(weights);
 conv1_mvs = net.params('conv1',1).get_data();
  
 models= '../temporal_zbw.prototxt';
 weights = '../temporal_model_no_addition/_iter_90000.caffemodel'
 net_ori = caffe.Net(models,'test')
 net_ori.copy_from(weights);
 conv1 = net_ori.params('conv1',1).get_data();
 
 models= '../temporal_zbw.prototxt';
 weights = '/home/bwzhang/DeepLearning/project/caffe/temporal_model/_iter_90000.caffemodel'
 net = caffe.Net(models,'test')
 net.copy_from(weights);
 conv1_ori = net.params('conv1',1).get_data();
  

img_show=[];
 
for i = 1:size(conv1_ori,3)
    img_show_row=[];
    for j = 1:size(conv1_ori,4)
        abc = mat2gray(conv1_ori(:, :, i, j));
%         abc = imresize(abc, size(abc));
        t(i,j)= entropy(double(abc));
        img_show_row = [img_show_row, zeros(size(abc,1), 1), abc];
    end
    img_show = [img_show; zeros(1, size(img_show_row,2)); img_show_row];
end

img_show = [img_show; zeros(1, size(img_show, 2))];

for i = 1:size(conv1,3)
    img_show_row=[];
    for j = 1:size(conv1,4)
        abc = mat2gray(conv1(:, :, i, j));
%         abc = imresize(abc, size(abc));
        t(i,j)= entropy(double(abc));
        img_show_row = [img_show_row, zeros(size(abc,1), 1), abc];
    end
    img_show = [img_show; zeros(1, size(img_show_row,2)); img_show_row];
end

img_show = [img_show; zeros(1, size(img_show, 2))];

for i = 1:size(conv1_mvs,3)
    img_show_row=[];
    for j = 1:size(conv1_mvs,4)
        abc = mat2gray(conv1_mvs(:, :, i, j));
%         abc = imresize(abc, size(abc));
        t(i,j)= entropy(double(abc));
        img_show_row = [img_show_row, zeros(size(abc,1), 1), abc];
    end
    img_show = [img_show; zeros(1, size(img_show_row,2)); img_show_row];
end


imshow (img_show),colormap('gray');
