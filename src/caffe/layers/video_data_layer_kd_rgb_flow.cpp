#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
VideoDataKDRFLayer<Dtype>:: ~VideoDataKDRFLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataKDRFLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int new_height  = this->layer_param_.video_data_kdrf_param().new_height();
	const int new_width  = this->layer_param_.video_data_kdrf_param().new_width();
	const int new_length  = this->layer_param_.video_data_kdrf_param().new_length();
	const int num_segments = this->layer_param_.video_data_kdrf_param().num_segments();
	const string& source = this->layer_param_.video_data_kdrf_param().source();

	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	string filename;
	string dir_mvs;
	string dir_tvl1;
	int label;
	int length;
	while (infile >> dir_mvs >> dir_tvl1 >> filename >> length >> label){
		lines_dir_.push_back(std::make_pair(dir_mvs, dir_tvl1));
		lines_.push_back(std::make_pair(filename,label));
		lines_duration_.push_back(length);
	}
	if (this->layer_param_.video_data_kdrf_param().shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleVideos();
	}

	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	lines_id_ = 0;

	Datum datum;
	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
	int average_duration = (int) lines_duration_[lines_id_]/num_segments;
	vector<int> offsets;
	for (int i = 0; i < num_segments; ++i){
		caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		int offset = (*frame_rng)() % (average_duration - new_length + 1);
		offsets.push_back(offset+i*average_duration);
	}
	if (this->layer_param_.video_data_kdrf_param().modality() == VideoDataKDRFParameter_Modality_FLOW)
		CHECK(ReadSegmentFlowToDatum_KDRF(lines_[lines_id_].first, lines_dir_[lines_id_].first, lines_dir_[lines_id_].second, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum));
	else
		CHECK(ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true));
	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.video_data_kdrf_param().batch_size();
	if (crop_size > 0){
		top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
	} else {
		top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
		this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	}
	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

	top[1]->Reshape(batch_size, 1, 1, 1);
	this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

    LOG(INFO) << "Hi there";
	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    LOG(INFO) << "Hi There two";
	this->transformed_data_.Reshape(top_shape);
}

template <typename Dtype>
void VideoDataKDRFLayer<Dtype>::ShuffleVideos(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

template <typename Dtype>
void VideoDataKDRFLayer<Dtype>::InternalThreadEntry(){

	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	VideoDataKDRFParameter video_data_kdrf_param = this->layer_param_.video_data_kdrf_param();
	const int batch_size = video_data_kdrf_param.batch_size();
	const int new_height = video_data_kdrf_param.new_height();
	const int new_width = video_data_kdrf_param.new_width();
	const int new_length = video_data_kdrf_param.new_length();
	const int num_segments = video_data_kdrf_param.num_segments();
	const int lines_size = lines_.size();

	for (int item_id = 0; item_id < batch_size; ++item_id){
		CHECK_GT(lines_size, lines_id_);
		vector<int> offsets;
		int average_duration = (int) lines_duration_[lines_id_] / num_segments;
		for (int i = 0; i < num_segments; ++i){
			if (this->phase_==TRAIN){
				caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
				int offset = (*frame_rng)() % (average_duration - new_length + 1);
				offsets.push_back(offset+i*average_duration);
			} else{
				offsets.push_back(int((average_duration-new_length+1)/2 + i*average_duration));
			}
		}
		if (this->layer_param_.video_data_kdrf_param().modality() == VideoDataKDRFParameter_Modality_FLOW){
			if(!ReadSegmentFlowToDatum_KDRF(lines_[lines_id_].first, lines_dir_[lines_id_].first, lines_dir_[lines_id_].second, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum)) {
				continue;
			}
		} else{
			if(!ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true)) {
				continue;
			}
		}
		int offset1 = this->prefetch_data_.offset(item_id);
    	this->transformed_data_.set_cpu_data(top_data + offset1);
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
		top_label[item_id] = lines_[lines_id_].second;
		//LOG()

		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_size) {
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if(this->layer_param_.video_data_kdrf_param().shuffle()){
				ShuffleVideos();
			}
		}
	}
}

INSTANTIATE_CLASS(VideoDataKDRFLayer);
REGISTER_LAYER_CLASS(VideoDataKDRF);
}
