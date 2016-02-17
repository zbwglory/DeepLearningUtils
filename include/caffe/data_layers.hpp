#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
public:
	explicit BaseDataLayer(const LayerParameter& param);
	// LayerSetUp: implements common data layer setup functionality, and calls
	// DataLayerSetUp to do special data layer setup for individual layer types.
	// This method may not be overridden except by the BasePrefetchingDataLayer.
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {}
	// Data layers have no bottoms, so reshaping is trivial.
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {}

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

#ifdef USE_MPI
	/**
	 * @brief call advance_cursor() for `step` times to offset the data access for parallel training
	 */
	inline virtual void OffsetCursor(int step){
		if (Caffe::parallel_mode() == Caffe::MPI){
			for (int i = 0; i < step; ++i) this->advance_cursor();
		}
	}
#endif

protected:

#ifdef USE_MPI
	/**
	 * @brief The core utility for parallel based data access
	 *
	 * This move the "cursor" defined in each data layer one step forward
	 */
	inline virtual void advance_cursor(){
		LOG(FATAL)<<"Data must implement advance_cursor() method to be involved in the parallel training";
	}
#endif

	TransformationParameter transform_param_;
	shared_ptr<DataTransformer<Dtype> > data_transformer_;
	bool output_labels_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
		public BaseDataLayer<Dtype>, public InternalThread {
		public:
	explicit BasePrefetchingDataLayer(const LayerParameter& param)
	: BaseDataLayer<Dtype>(param) {}
	// LayerSetUp: implements common data layer setup functionality, and calls
	// DataLayerSetUp to do special data layer setup for individual layer types.
	// This method may not be overridden.
	void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual void CreatePrefetchThread();
	virtual void JoinPrefetchThread();
	// The thread's function
	virtual void InternalThreadEntry() {}

		protected:
	Blob<Dtype> prefetch_data_;
	Blob<Dtype> prefetch_label_;
	Blob<Dtype> transformed_data_;
};

template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
	explicit DataLayer(const LayerParameter& param)
	: BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~DataLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Data"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline int MaxTopBlobs() const { return 2; }


protected:
	virtual void InternalThreadEntry();

#ifdef USE_MPI
	inline virtual void advance_cursor() {
		if (cur_input_mode_ == SEQUENCE) {
			cursor_->Next();
			if (!cursor_->valid()) {
				DLOG(INFO) << "Restarting data prefetching from start.";
				cursor_->SeekToFirst();

				if (this->layer_param_.data_param().shuffle() == true){
					LOG(INFO)<<"Entering shuffling mode after first epoch";
					cur_input_mode_ = SHUFFLE;
					shuffle(shuffle_key_pool_.begin(), shuffle_key_pool_.end());
					shuffle_cursor_ = shuffle_key_pool_.begin();
				}
			}
		}else if (cur_input_mode_ == SHUFFLE){
			//NO OP
		}
	}
#endif

	shared_ptr<db::DB> db_;
	shared_ptr<db::Cursor> cursor_;

	enum InputMode{
			SEQUENCE, SHUFFLE
	};
	InputMode cur_input_mode_;
	vector<string> shuffle_key_pool_;
	vector<string>::iterator shuffle_cursor_;
};

/**
 * @brief Provides data to the Net generated by a Filler.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class DummyDataLayer : public Layer<Dtype> {
public:
	explicit DummyDataLayer(const LayerParameter& param)
	: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	// Data layers have no bottoms, so reshaping is trivial.
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {}

	virtual inline const char* type() const { return "DummyData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int MinTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

	vector<shared_ptr<Filler<Dtype> > > fillers_;
	vector<bool> refill_;
};

/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5DataLayer : public Layer<Dtype> {
public:
	explicit HDF5DataLayer(const LayerParameter& param)
	: Layer<Dtype>(param) {}
	virtual ~HDF5DataLayer();
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	// Data layers have no bottoms, so reshaping is trivial.
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {}

	virtual inline const char* type() const { return "HDF5Data"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int MinTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void LoadHDF5FileData(const char* filename);

	std::vector<std::string> hdf_filenames_;
	unsigned int num_files_;
	unsigned int current_file_;
	hsize_t current_row_;
	std::vector<shared_ptr<Blob<Dtype> > > hdf_blobs_;
	std::vector<unsigned int> data_permutation_;
	std::vector<unsigned int> file_permutation_;
};

/**
 * @brief Write blobs to disk as HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5OutputLayer : public Layer<Dtype> {
public:
	explicit HDF5OutputLayer(const LayerParameter& param)
	: Layer<Dtype>(param), file_opened_(false) {}
	virtual ~HDF5OutputLayer();
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	// Data layers have no bottoms, so reshaping is trivial.
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {}

	virtual inline const char* type() const { return "HDF5Output"; }
	// TODO: no limit on the number of blobs
	virtual inline int ExactNumBottomBlobs() const { return 2; }
	virtual inline int ExactNumTopBlobs() const { return 0; }

	inline std::string file_name() const { return file_name_; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void SaveBlobs();

	bool file_opened_;
	std::string file_name_;
	hid_t file_id_;
	Blob<Dtype> data_blob_;
	Blob<Dtype> label_blob_;
};

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
	explicit ImageDataLayer(const LayerParameter& param)
	: BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~ImageDataLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ImageData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
	shared_ptr<Caffe::RNG> prefetch_rng_;
	virtual void ShuffleImages();
	virtual void InternalThreadEntry();

#ifdef USE_MPI
	inline virtual void advance_cursor(){
		lines_id_++;
		if (lines_id_ >= lines_.size()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.image_data_param().shuffle()) {
				ShuffleImages();
			}
		}
	}
#endif

	vector<std::pair<std::string, int> > lines_;
	int lines_id_;
};

/**
 * @brief Provides data to the Net from video files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class VideoDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
	explicit VideoDataLayer(const LayerParameter& param)
	: BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~VideoDataLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "VideoData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
	shared_ptr<Caffe::RNG> prefetch_rng_;
	shared_ptr<Caffe::RNG> prefetch_rng_2_;
	shared_ptr<Caffe::RNG> prefetch_rng_1_;
	shared_ptr<Caffe::RNG> frame_prefetch_rng_;
	virtual void ShuffleVideos();
	virtual void InternalThreadEntry();

#ifdef USE_MPI
	inline virtual void advance_cursor(){
		lines_id_++;
		if (lines_id_ >= lines_.size()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.video_data_param().shuffle()) {
				ShuffleVideos();
			}
		}
	}
#endif

	vector<std::pair<std::string, int> > lines_;
	vector<int> lines_duration_;
	int lines_id_;
};

/**
 * @brief Provides data to the Net from video files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class VideoDataKDLayer: public BasePrefetchingDataLayer<Dtype> {
public:
	explicit VideoDataKDLayer(const LayerParameter& param)
	: BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~VideoDataKDLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "VideoDataKD"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
	shared_ptr<Caffe::RNG> prefetch_rng_;
	shared_ptr<Caffe::RNG> prefetch_rng_2_;
	shared_ptr<Caffe::RNG> prefetch_rng_1_;
	shared_ptr<Caffe::RNG> frame_prefetch_rng_;
	virtual void ShuffleVideos();
	virtual void InternalThreadEntry();

#ifdef USE_MPI
	inline virtual void advance_cursor(){
		lines_id_++;
		if (lines_id_ >= lines_.size()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.video_data_param().shuffle()) {
				ShuffleVideos();
			}
		}
	}
#endif

	vector<std::pair<std::string, int> > lines_;
	vector<std::pair<std::string, std::string> > lines_dir_;
	vector<int> lines_duration_;
	int lines_id_;
};

template <typename Dtype>
class VideoDataKDRFLayer: public BasePrefetchingDataLayer<Dtype> {
public:
	explicit VideoDataKDRFLayer(const LayerParameter& param)
	: BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~VideoDataKDRFLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "VideoDataKD"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
	shared_ptr<Caffe::RNG> prefetch_rng_;
	shared_ptr<Caffe::RNG> prefetch_rng_2_;
	shared_ptr<Caffe::RNG> prefetch_rng_1_;
	shared_ptr<Caffe::RNG> frame_prefetch_rng_;
	virtual void ShuffleVideos();
	virtual void InternalThreadEntry();

#ifdef USE_MPI
	inline virtual void advance_cursor(){
		lines_id_++;
		if (lines_id_ >= lines_.size()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.video_data_param().shuffle()) {
				ShuffleVideos();
			}
		}
	}
#endif

	vector<std::pair<std::string, int> > lines_;
	vector<std::pair<std::string, std::string> > lines_dir_;
	vector<int> lines_duration_;
	int lines_id_;
};




/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MemoryDataLayer : public BaseDataLayer<Dtype> {
public:
	explicit MemoryDataLayer(const LayerParameter& param)
	: BaseDataLayer<Dtype>(param), has_new_data_(false) {}
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "MemoryData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

	virtual void AddDatumVector(const vector<Datum>& datum_vector);
	virtual void AddMatVector(const vector<cv::Mat>& mat_vector,
			const vector<int>& labels);

	// Reset should accept const pointers, but can't, because the memory
	//  will be given to Blob, which is mutable
	void Reset(Dtype* data, Dtype* label, int n);
	void set_batch_size(int new_size);

	int batch_size() { return batch_size_; }
	int channels() { return channels_; }
	int height() { return height_; }
	int width() { return width_; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	int batch_size_, channels_, height_, width_, size_;
	Dtype* data_;
	Dtype* labels_;
	int n_;
	size_t pos_;
	Blob<Dtype> added_data_;
	Blob<Dtype> added_label_;
	bool has_new_data_;
};

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class WindowDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
	explicit WindowDataLayer(const LayerParameter& param)
	: BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~WindowDataLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "WindowData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
	virtual unsigned int PrefetchRand();
	virtual void InternalThreadEntry();

#ifdef USE_MPI
	inline virtual void advance_cursor(){
		//TODO: remove this
		PrefetchRand();
		this->transform_param_.mirror() && PrefetchRand();
	}
#endif

	shared_ptr<Caffe::RNG> prefetch_rng_;
	vector<std::pair<std::string, vector<int> > > image_database_;
	enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
	vector<vector<float> > fg_windows_;
	vector<vector<float> > bg_windows_;
	Blob<Dtype> data_mean_;
	vector<Dtype> mean_values_;
	bool has_mean_file_;
	bool has_mean_values_;
	bool cache_images_;
	vector<std::pair<std::string, Datum > > image_database_cache_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
