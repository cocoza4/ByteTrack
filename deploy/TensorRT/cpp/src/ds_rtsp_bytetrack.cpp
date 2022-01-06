#include <gst/gst.h>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "cuda_runtime_api.h"
#include "BYTETracker.h"

#define DEVICE 0 // GPU id
#define NMS_THRESH 0.7
#define BBOX_CONF_THRESH 0.1

/* By default, OSD process-mode is set to CPU_MODE. To change mode, set as:
 * 1: GPU mode (for Tesla only)
 * 2: HW mode (For Jetson only)
 */
#define OSD_PROCESS_MODE 0

/* Whether to display different color for each bounding box */
#define MULTI_BB_COLOR 0

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1088
#define MUXER_OUTPUT_HEIGHT 608

#define NET_INPUT_WIDTH 1088
#define NET_INPUT_HEIGHT 608

#define TILED_OUTPUT_WIDTH 1480
#define TILED_OUTPUT_HEIGHT 820

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

// stuff we know about the network and the input/output blobs
const char *INPUT_BLOB_NAME = "input_0";
const char *OUTPUT_BLOB_NAME = "output_0";

namespace po = boost::program_options;

typedef struct _CustomData {
  // std::unique_ptr<BYTETracker> tracker1;
  // std::unique_ptr<BYTETracker> tracker2;
  BYTETracker *tracker1;
  BYTETracker *tracker2;
} CustomData;

struct GridAndStride
{
	int grid0;
	int grid1;
	int stride;
};

static void generate_grids_and_stride(const int target_w, const int target_h, vector<int> &strides, vector<GridAndStride> &grid_strides)
{
	for (auto stride : strides)
	{
		int num_grid_w = target_w / stride;
		int num_grid_h = target_h / stride;
		for (int g1 = 0; g1 < num_grid_h; g1++)
		{
			for (int g0 = 0; g0 < num_grid_w; g0++)
			{
				grid_strides.push_back((GridAndStride){g0, g1, stride});
			}
		}
	}
}

static inline float intersection_area(const Object &a, const Object &b)
{
	Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

static void qsort_descent_inplace(vector<Object> &faceobjects, int left, int right)
{
	int i = left;
	int j = right;
	float p = faceobjects[(left + right) / 2].prob;

	while (i <= j)
	{
		while (faceobjects[i].prob > p)
			i++;

		while (faceobjects[j].prob < p)
			j--;

		if (i <= j)
		{
			// swap
			swap(faceobjects[i], faceobjects[j]);

			i++;
			j--;
		}
	}

#pragma omp parallel sections
	{
#pragma omp section
		{
			if (left < j)
				qsort_descent_inplace(faceobjects, left, j);
		}
#pragma omp section
		{
			if (i < right)
				qsort_descent_inplace(faceobjects, i, right);
		}
	}
}

static void qsort_descent_inplace(vector<Object> &objects)
{
	if (objects.empty())
		return;

	qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const vector<Object> &faceobjects, vector<int> &picked, float nms_threshold)
{
	picked.clear();

	const int n = faceobjects.size();

	vector<float> areas(n);
	for (int i = 0; i < n; i++)
	{
		areas[i] = faceobjects[i].rect.area();
	}

	for (int i = 0; i < n; i++)
	{
		const Object &a = faceobjects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++)
		{
			const Object &b = faceobjects[picked[j]];

			// intersection over union
			float inter_area = intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}
}

static void generate_yolox_proposals(vector<GridAndStride> grid_strides, float *feat_blob, float prob_threshold, vector<Object> &objects)
{
	const int num_class = 1;

	const int num_anchors = grid_strides.size();

	for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
	{
		const int grid0 = grid_strides[anchor_idx].grid0;
		const int grid1 = grid_strides[anchor_idx].grid1;
		const int stride = grid_strides[anchor_idx].stride;

		const int basic_pos = anchor_idx * (num_class + 5);

		// yolox/models/yolo_head.py decode logic
		float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
		float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
		float w = exp(feat_blob[basic_pos + 2]) * stride;
		float h = exp(feat_blob[basic_pos + 3]) * stride;
		float x0 = x_center - w * 0.5f;
		float y0 = y_center - h * 0.5f;

		float box_objectness = feat_blob[basic_pos + 4];
		for (int class_idx = 0; class_idx < num_class; class_idx++)
		{
			float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
			float box_prob = box_objectness * box_cls_score;
			if (box_prob > prob_threshold)
			{
				Object obj;
				obj.rect.x = x0;
				obj.rect.y = y0;
				obj.rect.width = w;
				obj.rect.height = h;
				obj.label = class_idx;
				obj.prob = box_prob;

				objects.push_back(obj);
			}

		} // class loop

	} // point anchor loop
}

static void decode_outputs(float *prob, vector<Object> &objects, float scale, const int img_w, const int img_h)
{
	vector<Object> proposals;
	vector<int> strides = {8, 16, 32};
	vector<GridAndStride> grid_strides;
	generate_grids_and_stride(NET_INPUT_WIDTH, NET_INPUT_HEIGHT, strides, grid_strides);
	generate_yolox_proposals(grid_strides, prob, BBOX_CONF_THRESH, proposals);
	//std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

	qsort_descent_inplace(proposals);

	vector<int> picked;
	nms_sorted_bboxes(proposals, picked, NMS_THRESH);

	int count = picked.size();

	//std::cout << "num of boxes: " << count << std::endl;

	objects.resize(count);
	for (int i = 0; i < count; i++)
	{
		objects[i] = proposals[picked[i]];

		// adjust offset to original unpadded
		float x0 = (objects[i].rect.x) / scale;
		float y0 = (objects[i].rect.y) / scale;
		float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
		float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

		// clip
		x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
		y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
		x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
		y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

		objects[i].rect.x = x0;
		objects[i].rect.y = y0;
		objects[i].rect.width = x1 - x0;
		objects[i].rect.height = y1 - y0;
	}
}

template <typename... Args>
static std::string format(const std::string& fmt, const Args&... args) {
	boost::format f(fmt);
	std::initializer_list<char> {(static_cast<void>(
			f % args
	), char{}) ...};

	return boost::str(f);
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
	GMainLoop *loop = (GMainLoop *)data;
	switch (GST_MESSAGE_TYPE(msg))
	{
	case GST_MESSAGE_EOS:
		g_print("End of stream\n");
		g_main_loop_quit(loop);
		break;
	case GST_MESSAGE_ERROR:
	{
		gchar *debug;
		GError *error;
		gst_message_parse_error(msg, &error, &debug);
		g_printerr("ERROR from element %s: %s\n",
							 GST_OBJECT_NAME(msg->src), error->message);
		if (debug)
			g_printerr("Error details: %s\n", debug);
		g_free(debug);
		g_error_free(error);
		g_main_loop_quit(loop);
		break;
	}
	default:
		break;
	}
	return TRUE;
}

static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
	NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(GST_BUFFER(info->data));
	// /* Iterate each frame metadata in batch */
	// for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
	// 	NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

	// 	nvds_clear_obj_meta_list(frame_meta, frame_meta->obj_meta_list);

	// 	NvDsObjectMeta *obj_meta = NULL;
	// 	NvDsObjectMetaList *l_meta = frame_meta->obj_meta_list;
	// 	for (NvDsObjectMetaList *l_meta = frame_meta->obj_meta_list; l_meta != NULL; l_meta = l_meta->next) {
	// 		obj_meta = (NvDsObjectMeta *)(l_meta->data);
	// 		std::cout << "unique_component_id: " << obj_meta->unique_component_id << std::endl;
	// 		// nvds_remove_obj_meta_from_frame(frame_meta, obj_meta);
	// 	}
	// }

	return GST_PAD_PROBE_OK;
}

static void add_obj_meta_to_frame(const vector<STrack>& output_stracks, NvDsInferTensorMeta *tensor_meta, 
																	NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta) {
	
	int size = output_stracks.size();
	g_print("DEBUG: source_id: %d, output_stracks: %d\n", frame_meta->source_id, size);
	int class_id = 0;

	// for (const auto& strack : output_stracks) {
	for (auto i = 0; i < size; i++) {
		// const auto& strack = output_stracks[i];
		std::vector<float> tlwh = output_stracks[i].tlwh;
		bool vertical = tlwh[2] / tlwh[3] > 1.6;
		if (tlwh[2] * tlwh[3] > 20 && !vertical) {

			int tracker_id = output_stracks[i].track_id;
			std::cout << "tracker_id: " << tracker_id << std::endl;

			auto rect{cv::Rect_<float>(tlwh[0], tlwh[1], tlwh[2], tlwh[3])};
			// g_print("xywh: (%f, %f, %f, %f)\n", rect.x, rect.y, rect.width, rect.height);

			NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
			
			obj_meta->unique_component_id = tensor_meta->unique_id;
			obj_meta->confidence = 0.;

			/* This is an untracked object. Set tracking_id to -1. */
			obj_meta->object_id = UNTRACKED_OBJECT_ID;
			obj_meta->class_id = class_id;

			NvOSD_RectParams &rect_params = obj_meta->rect_params;
			NvOSD_TextParams &text_params = obj_meta->text_params;

			/* Assign bounding box coordinates. */
			rect_params.left = rect.x * MUXER_OUTPUT_WIDTH / NET_INPUT_WIDTH;
			rect_params.top = rect.y * MUXER_OUTPUT_HEIGHT / NET_INPUT_HEIGHT;
			rect_params.width = rect.width * MUXER_OUTPUT_WIDTH / NET_INPUT_WIDTH;
			rect_params.height = rect.height * MUXER_OUTPUT_HEIGHT / NET_INPUT_HEIGHT;

			/* Border of width 3. */
			rect_params.border_width = 3;
			rect_params.has_bg_color = 0;
#if MULTI_BB_COLOR
			// Scalar s = tracker->get_color(tracker_id);
			// rect_params.border_color = (NvOSD_ColorParams){
			// 		s[0], s[1], s[2], s[3]};
			// g_print("s: (%f, %f, %f, %f)\n", s[0], s[1], s[2], s[3]);
#else
			rect_params.border_color = (NvOSD_ColorParams){1, 0, 0, 1};
#endif

			/* display_text requires heap allocated memory. */
			gchar *text = g_strdup_printf("%i", tracker_id);
			text_params.display_text = g_strdup(text);
			/* Display text above the left top corner of the object. */
			text_params.x_offset = rect_params.left;
			text_params.y_offset = rect_params.top - 10;
			/* Set black background for the text. */
			text_params.set_bg_clr = 1;
			text_params.text_bg_clr = (NvOSD_ColorParams){0, 0, 0, 1};
			/* Font face, size and color. */
			text_params.font_params.font_name = (gchar *)"Serif";
			text_params.font_params.font_size = 11;
			text_params.font_params.font_color = (NvOSD_ColorParams){1, 1, 1, 1};
			nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
		}
	}
}

/* This is the buffer probe function that we have registered on the src pad
 * of the PGIE's next queue element. PGIE element in the pipeline shall attach
 * its NvDsInferTensorMeta to each frame metadata on GstBuffer, here we will
 * iterate & parse the tensor data to get detection bounding boxes. The result
 * would be attached as object-meta(NvDsObjectMeta) into the same frame metadata.
 */
static GstPadProbeReturn pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
	static guint use_device_mem = 0;
	// const auto trackers = static_cast<std::vector<BYTETracker *> *>(u_data);

	CustomData *data = (CustomData *)u_data;
	NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(GST_BUFFER(info->data));

	/* Iterate each frame metadata in batch */
	int batch = 0;
	for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
		batch++;
		NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
		int img_width = frame_meta->source_frame_width;
		int img_height = frame_meta->source_frame_height;
		// g_print("DEBUG: frame width, height: %d, %d\n", img_width, img_height);

		// nvds_clear_obj_meta_list(frame_meta, frame_meta->obj_meta_list);
		// auto *tracker = trackers->at(frame_meta->source_id);
		// const auto& tracker = (*trackers)[frame_meta->source_id];
		g_print("DEBUG: batch_id: %d, source_id: %d, num_obj_meta: %d\n", frame_meta->batch_id, frame_meta->source_id, frame_meta->num_obj_meta);

		// TODO: change to array, use 1 or 2 sources for debugging only
		BYTETracker *tracker = NULL;
		int src_id = frame_meta->source_id;
		if (src_id == 0) {
			tracker = data->tracker1;
		} else if (src_id == 1) {
			tracker = data->tracker2;
		} else {
			g_printerr("ERROR: not supported\n");
		}
		/* Iterate user metadata in frames to search PGIE's tensor metadata */
		for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
			NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;

			/* only interested in raw tensor outputs */
			if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
				/* convert to tensor metadata */
				NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
				assert(meta->num_output_layers == 1);
				// g_print("meta->num_output_layers %d\n", meta->num_output_layers);

				// ByteTrack contains only 1 output
				NvDsInferLayerInfo *info = &meta->output_layers_info[0];
				for (unsigned int i = 0; i < meta->num_output_layers; i++) {
					// NvDsInferLayerInfo *info = &meta->output_layers_info[i];
					info->buffer = meta->out_buf_ptrs_host[i];
					if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
						cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
											info->inferDims.numElements, cudaMemcpyDeviceToHost);
						// g_print("%d, info->inferDims.numElements %d\n", i, info->inferDims.numElements);
						g_print("num_dims: %u, [%d, %d, %d]\n", info->inferDims.numDims, info->inferDims.d[0], info->inferDims.d[1], info->inferDims.d[2]);
					}
				}

				std::vector<Object> objects;
				float *probs = static_cast<float *>(info->buffer);
				// float scale = min(NET_INPUT_WIDTH / (img_width*1.0), NET_INPUT_HEIGHT / (img_height*1.0));
				float scale = 1.0;
				decode_outputs(probs, objects, scale, img_width, img_height);

				std::vector<STrack> output_stracks = tracker->update(objects);
				g_print("DEBUG: source_id: %d, tracker_id: %d\n", src_id, tracker->id);
				assert(src_id == tracker->id);
				add_obj_meta_to_frame(output_stracks, meta, batch_meta, frame_meta);
			}
		}
	}

	std::cout << "batch=" << batch << std::endl;
	use_device_mem = 1 - use_device_mem;
	return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn summary_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
	static guint use_device_mem = 0;
	// const auto trackers = static_cast<std::vector<BYTETracker *> *>(u_data);

	CustomData *data = (CustomData *)u_data;
	NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(GST_BUFFER(info->data));

	// TODO: display dataf from BYTETracker

	// /* Iterate each frame metadata in batch */
	// int batch = 0;
	// for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
	// 	batch++;
	// 	NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
	// 	int img_width = frame_meta->source_frame_width;
	// 	int img_height = frame_meta->source_frame_height;

	// 	// nvds_clear_obj_meta_list(frame_meta, frame_meta->obj_meta_list);
	// 	// auto *tracker = trackers->at(frame_meta->source_id);
	// 	// const auto& tracker = (*trackers)[frame_meta->source_id];
	// 	g_print("xxxxxxxxxxx DEBUG: batch_id: %d, source_id: %d, num_obj_meta: %d\n", frame_meta->batch_id, frame_meta->source_id, frame_meta->num_obj_meta);

	// 	for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
	// 			NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
	// 			// if (obj_meta->class_id == 0) {
	// 			// 	vehicle_count++;
	// 			// 	num_rects++;
	// 			// }
	// 			// if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
	// 			// 	person_count++;
	// 			// 	num_rects++;
	// 			// }
	// 	}
	// }

	// std::cout << "batch=" << batch << std::endl;
	// use_device_mem = 1 - use_device_mem;
	return GST_PAD_PROBE_OK;
}

static void cb_newpad(GstElement * decodebin, GstPad * decoder_src_pad, gpointer data) {
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp(name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref(bin_ghost_pad);
    } else {
      g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar *name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
}

static GstElement* create_source_bin(guint index, const gchar* uri) {
  GstElement *bin = NULL, *uri_decode_bin = NULL;
	const auto& bin_name = format("source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new(bin_name.c_str());

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add(GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

void create_multi_source_bin(GstElement *pipeline, GstElement *streammux, const std::vector<std::string>& sources) {
	for (int i = 0; i < sources.size(); i++) {
    GstPad *sinkpad, *srcpad;
    GstElement *source_bin = create_source_bin(i, sources[i].c_str());

    if (!source_bin) {
      throw std::runtime_error("Failed to create source bin. Exiting.");
    }
    gst_bin_add(GST_BIN(pipeline), source_bin);

		const auto& pad_name = format("sink_%d", i);
    sinkpad = gst_element_get_request_pad(streammux, pad_name.c_str());
    if (!sinkpad) {
      throw std::runtime_error("Streammux request sink pad failed. Exiting.");
    }

    srcpad = gst_element_get_static_pad(source_bin, "src");
    if (!srcpad) {
      throw std::runtime_error("Failed to get src pad of source bin. Exiting.");
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
      throw std::runtime_error("Failed to link source bin to stream muxer. Exiting.");
    }

    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
  }
}

void run(const std::vector<std::string>& sources, const bool display) {
	cudaSetDevice(DEVICE);

	GstBus *bus;
	GMainLoop *main_loop;
	guint bus_watch_id;

	GstElement *pipeline = NULL, *streammux = NULL, *pgie = NULL;

  int current_device = -1, pgie_batch_size = 1;
  int num_sources = sources.size();
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

	/* Initialize GStreamer */
	main_loop = g_main_loop_new(NULL, FALSE);

	/* Create gstreamer elements */
	/* Create Pipeline element that will form a connection of other elements */
	pipeline = gst_pipeline_new("people-tracking-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
		throw std::runtime_error("One element could not be created");
  }
  gst_bin_add(GST_BIN(pipeline), streammux);

  create_multi_source_bin(pipeline, streammux, sources);

  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
	if (!pgie) {
		throw std::runtime_error("nvinfer element could not be created");
	}

  g_object_set(G_OBJECT(streammux), "batch-size", num_sources, NULL);
  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
							 MUXER_OUTPUT_HEIGHT, "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

	/* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
	g_object_set(G_OBJECT(pgie), "config-file-path", "../src/pgie_config.txt", NULL);

	// TODO: I think model batch_size is wrong, verify each model
	if (num_sources == 1 || num_sources == 2 || num_sources == 4) {
		const auto& p = format("../../models/bytetrack_s_b%d.engine", num_sources);
		std::cout << "Engine file: " << p << std::endl;
		g_object_set(G_OBJECT(pgie), "model-engine-file", p.c_str(), NULL);
	} else {
		const auto& msg = format("ERROR: num_sources (%d) not supported\n", num_sources);
		throw std::runtime_error(msg);
	}

	/* Override the batch-size set in the config file with the number of sources. */
  g_object_get(G_OBJECT(pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set(G_OBJECT(pgie), "batch-size", num_sources, NULL);
  }

	/* we add a message handler */
	bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
	bus_watch_id = gst_bus_add_watch(bus, bus_call, main_loop);
	gst_object_unref(bus);

	// DEBUG
	CustomData data;
	data.tracker1 = new BYTETracker(0, 30, 30);
	data.tracker2 = new BYTETracker(1, 30, 30);
	// std::vector<BYTETracker *> trackers;
	// for (int i = 0; i < num_sources; i++) {
  //   auto tracker = new BYTETracker(i, 30, 30);
	// 	std::cout << "Creating ByteTrack for source " << tracker->id << std::endl;
	// 	trackers.emplace_back(tracker);
  // }

	/* Add probes */
	GstPad *pgie_src_pad = gst_element_get_static_pad(pgie, "src");
	if (!pgie_src_pad)
		g_print("Unable to get src pad\n");
	else {
		gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
				pgie_src_pad_buffer_probe, &data, NULL);
	}
	gst_object_unref(pgie_src_pad);

	/* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
	if (display) {
		/* Use convertor to convert from NV12 to RGBA as required by nvosd */
		GstElement *nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

		/* Create OSD to draw on the converted RGBA buffer */
		GstElement *nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

		GstElement *sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");

		if (!nvvidconv || !nvosd || !sink) {
			throw std::runtime_error("One element could not be created. Exiting.\n");
		}

		// g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
  	//     "display-text", OSD_DISPLAY_TEXT, NULL);
  	g_object_set(G_OBJECT(sink), "qos", 0, NULL);

		/* Add queue elements between every two elements */
		GstElement *queue1 = gst_element_factory_make ("queue", "queue1");
		GstElement *queue2 = gst_element_factory_make ("queue", "queue2");
		GstElement *queue3 = gst_element_factory_make ("queue", "queue3");
		GstElement *queue4 = gst_element_factory_make ("queue", "queue4");
		GstElement *queue5 = gst_element_factory_make ("queue", "queue5");

		GstElement *tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
		guint tiler_rows = (guint)sqrt(num_sources);
		guint tiler_columns = (guint)ceil(1.0 * num_sources / tiler_rows);

		/* we set the tiler properties here */
		g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns,
				"width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

		gst_bin_add_many(GST_BIN(pipeline), queue1, pgie, queue2, tiler, queue3,
			nvvidconv, queue4, nvosd, queue5, sink, NULL);

		/* we link the elements together
		* nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
		if (!gst_element_link_many(streammux, queue1, pgie, queue2, tiler, queue3,
					nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
			throw std::runtime_error("Elements could not be linked");
		}

	} else {

		GstElement *queue1 = gst_element_factory_make ("queue", "queue1");
		GstElement *queue2 = gst_element_factory_make ("queue", "queue2");
		// GstPad *osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");

		// gst_bin_add_many(GST_BIN(pipeline), queue1, pgie, queue2, streamdemux, NULL);

		// /* Lets add probe to get informed of the meta data generated, we add probe to
		// 	* the sink pad of the osd element, since by that time, the buffer would have
		// 	* had got all the metadata. */
		
	  // if (!osd_sink_pad)
	  //   g_print ("Unable to get sink pad\n");
	  // else
	  //   gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
	  //       summary_src_pad_buffer_probe, NULL, NULL);
	  // gst_object_unref(osd_sink_pad);
		

		GstElement *streamdemux = gst_element_factory_make("nvstreamdemux", "stream-demuxer");

		if (!streamdemux) {
			throw std::runtime_error("streamdemux element could not be created");
		}

		gst_bin_add_many(GST_BIN(pipeline), queue1, pgie, queue2, streamdemux, NULL);

		/* we link the elements together
		* nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
		if (!gst_element_link_many(streammux, queue1, pgie, queue2, streamdemux, NULL)) {
			throw std::runtime_error("Elements could not be linked");
		}

		/* Add probes */
		GstPad *demux_sink_pad = gst_element_get_static_pad(streamdemux, "sink");
		if (!demux_sink_pad)
			throw std::runtime_error("Unable to get src pad\n");
		else {
			g_print("add demuxer probe\n");
			gst_pad_add_probe(demux_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
					summary_src_pad_buffer_probe, &data, NULL);
		}
		gst_object_unref(demux_sink_pad);
	}

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
	// std::vector<std::unique_ptr<BYTETracker>> trackers;
	// for (int i = 0; i < num_sources; i++) {
  //   auto tracker = std::make_unique<BYTETracker>(i, 30, 30);
	// 	std::cout << "Creating ByteTrack for source " << tracker->id << std::endl;
	// 	trackers.emplace_back(std::move(tracker));
  // }

	

	


  /* Set the pipeline to "playing" state */
  g_print("Now processing:");
  for (auto& source : sources) {
    g_print(" %s,", source.c_str());
  }
  g_print ("\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

	/* Wait till pipeline encounters an error or EOS */
	g_print("Running...\n");
	g_main_loop_run(main_loop);

	/* Out of the main loop, clean up nicely */
	g_print("Returned, stopping playback\n");
	gst_element_set_state(pipeline, GST_STATE_NULL);

	g_print("Deleting pipeline\n");
	if (data.tracker1 != NULL) {
		delete data.tracker1;
	}
	if (data.tracker2 != NULL) {
		delete data.tracker2;
	}
	gst_object_unref(GST_OBJECT(pipeline));
	g_source_remove(bus_watch_id);
	g_main_loop_unref(main_loop);

}

int main(int argc, char **argv) {
	try {
    po::options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
			("source,s", po::value<std::vector<std::string>>(), "Input sources.")
      ("display,d", po::bool_switch()->default_value(false), "Enable on-screen display.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
		} else {
			const auto display = vm["display"].as<bool>();
			const auto& sources = vm["source"].as<std::vector<std::string>>();
      std::cout << "Enable on-screen display: " << vm["display"].as<bool>() << std::endl;
			gst_init(&argc, &argv);
			run(sources, display);
		}
  }
  catch (const po::error &ex) {
    std::cerr << ex.what() << std::endl;
  }
	
}
