#include <gst/gst.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "cuda_runtime_api.h"
#include "BYTETracker.h"

#define CHECK(status)                          \
	do                                           \
	{                                            \
		auto ret = (status);                       \
		if (ret != 0)                              \
		{                                          \
			cerr << "Cuda failure: " << ret << endl; \
			abort();                                 \
		}                                          \
	} while (0)

#define DEVICE 0 // GPU id
#define NMS_THRESH 0.7
#define BBOX_CONF_THRESH 0.1

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1088
#define MUXER_OUTPUT_HEIGHT 608

#define PGIE_NET_WIDTH 1088
#define PGIE_NET_HEIGHT 608

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/** set the user metadata type */
#define NVDS_USER_FRAME_META_EXAMPLE (nvds_get_user_meta_type("NVIDIA.NVINFER.USER_META"))

// stuff we know about the network and the input/output blobs
static const int INPUT_W = 1088;
static const int INPUT_H = 608;
const char *INPUT_BLOB_NAME = "input_0";
const char *OUTPUT_BLOB_NAME = "output_0";


Mat static_resize(Mat &img)
{
	float r = min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
	// r = std::min(r, 1.0f);
	int unpad_w = r * img.cols;
	int unpad_h = r * img.rows;
	Mat re(unpad_h, unpad_w, CV_8UC3);
	resize(img, re, re.size());
	Mat out(INPUT_H, INPUT_W, CV_8UC3, Scalar(114, 114, 114));
	re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
	return out;
}

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

float *blobFromImage(Mat &img)
{
	cvtColor(img, img, COLOR_BGR2RGB);

	float *blob = new float[img.total() * 3];
	int channels = 3;
	int img_h = img.rows;
	int img_w = img.cols;
	vector<float> mean = {0.485, 0.456, 0.406};
	vector<float> std = {0.229, 0.224, 0.225};
	for (size_t c = 0; c < channels; c++)
	{
		for (size_t h = 0; h < img_h; h++)
		{
			for (size_t w = 0; w < img_w; w++)
			{
				blob[c * img_w * img_h + h * img_w + w] =
						(((float)img.at<Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std[c];
			}
		}
	}
	return blob;
}

static void decode_outputs(float *prob, vector<Object> &objects, float scale, const int img_w, const int img_h)
{
	vector<Object> proposals;
	vector<int> strides = {8, 16, 32};
	vector<GridAndStride> grid_strides;
	generate_grids_and_stride(INPUT_W, INPUT_H, strides, grid_strides);
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
		// x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
		// y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
		// x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
		// y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

		objects[i].rect.x = x0;
		objects[i].rect.y = y0;
		objects[i].rect.width = x1 - x0;
		objects[i].rect.height = y1 - y0;
	}
}

const float color_list[80][3] =
		{
				{0.000, 0.447, 0.741},
				{0.850, 0.325, 0.098},
				{0.929, 0.694, 0.125},
				{0.494, 0.184, 0.556},
				{0.466, 0.674, 0.188},
				{0.301, 0.745, 0.933},
				{0.635, 0.078, 0.184},
				{0.300, 0.300, 0.300},
				{0.600, 0.600, 0.600},
				{1.000, 0.000, 0.000},
				{1.000, 0.500, 0.000},
				{0.749, 0.749, 0.000},
				{0.000, 1.000, 0.000},
				{0.000, 0.000, 1.000},
				{0.667, 0.000, 1.000},
				{0.333, 0.333, 0.000},
				{0.333, 0.667, 0.000},
				{0.333, 1.000, 0.000},
				{0.667, 0.333, 0.000},
				{0.667, 0.667, 0.000},
				{0.667, 1.000, 0.000},
				{1.000, 0.333, 0.000},
				{1.000, 0.667, 0.000},
				{1.000, 1.000, 0.000},
				{0.000, 0.333, 0.500},
				{0.000, 0.667, 0.500},
				{0.000, 1.000, 0.500},
				{0.333, 0.000, 0.500},
				{0.333, 0.333, 0.500},
				{0.333, 0.667, 0.500},
				{0.333, 1.000, 0.500},
				{0.667, 0.000, 0.500},
				{0.667, 0.333, 0.500},
				{0.667, 0.667, 0.500},
				{0.667, 1.000, 0.500},
				{1.000, 0.000, 0.500},
				{1.000, 0.333, 0.500},
				{1.000, 0.667, 0.500},
				{1.000, 1.000, 0.500},
				{0.000, 0.333, 1.000},
				{0.000, 0.667, 1.000},
				{0.000, 1.000, 1.000},
				{0.333, 0.000, 1.000},
				{0.333, 0.333, 1.000},
				{0.333, 0.667, 1.000},
				{0.333, 1.000, 1.000},
				{0.667, 0.000, 1.000},
				{0.667, 0.333, 1.000},
				{0.667, 0.667, 1.000},
				{0.667, 1.000, 1.000},
				{1.000, 0.000, 1.000},
				{1.000, 0.333, 1.000},
				{1.000, 0.667, 1.000},
				{0.333, 0.000, 0.000},
				{0.500, 0.000, 0.000},
				{0.667, 0.000, 0.000},
				{0.833, 0.000, 0.000},
				{1.000, 0.000, 0.000},
				{0.000, 0.167, 0.000},
				{0.000, 0.333, 0.000},
				{0.000, 0.500, 0.000},
				{0.000, 0.667, 0.000},
				{0.000, 0.833, 0.000},
				{0.000, 1.000, 0.000},
				{0.000, 0.000, 0.167},
				{0.000, 0.000, 0.333},
				{0.000, 0.000, 0.500},
				{0.000, 0.000, 0.667},
				{0.000, 0.000, 0.833},
				{0.000, 0.000, 1.000},
				{0.000, 0.000, 0.000},
				{0.143, 0.143, 0.143},
				{0.286, 0.286, 0.286},
				{0.429, 0.429, 0.429},
				{0.571, 0.571, 0.571},
				{0.714, 0.714, 0.714},
				{0.857, 0.857, 0.857},
				{0.000, 0.447, 0.741},
				{0.314, 0.717, 0.741},
				{0.50, 0.5, 0}};

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

/* This is the buffer probe function that we have registered on the src pad
 * of the PGIE's next queue element. PGIE element in the pipeline shall attach
 * its NvDsInferTensorMeta to each frame metadata on GstBuffer, here we will
 * iterate & parse the tensor data to get detection bounding boxes. The result
 * would be attached as object-meta(NvDsObjectMeta) into the same frame metadata.
 */
static GstPadProbeReturn nvinfer_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
	static guint use_device_mem = 0;
	BYTETracker *tracker = (BYTETracker *)u_data;

	NvDsBatchMeta *batch_meta =
			gst_buffer_get_nvds_batch_meta(GST_BUFFER(info->data));

	/* Iterate each frame metadata in batch */
	for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
			 l_frame = l_frame->next)
	{
		NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

		/* Iterate user metadata in frames to search PGIE's tensor metadata */
		for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list;
				 l_user != NULL; l_user = l_user->next)
		{
			NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
			if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
				continue;

			/* convert to tensor metadata */
			NvDsInferTensorMeta *meta =
					(NvDsInferTensorMeta *)user_meta->user_meta_data;
			// g_print("meta->num_output_layers %d\n", meta->num_output_layers);

			// ByteTrack contains only 1 output
			NvDsInferLayerInfo *info = &meta->output_layers_info[0];
			// g_print("numElements: %d\n", info->inferDims.numElements);

			for (unsigned int i = 0; i < meta->num_output_layers; i++)
			{
				NvDsInferLayerInfo *info = &meta->output_layers_info[i];
				info->buffer = meta->out_buf_ptrs_host[i];
				if (use_device_mem && meta->out_buf_ptrs_dev[i])
				{
					cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
										 info->inferDims.numElements, cudaMemcpyDeviceToHost);
				}
			}

			int class_id = 0;
			std::vector<Object> objects;
			float *probs = static_cast<float *>(info->buffer);
			//    float scale = min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
			float scale = 1.0;
			decode_outputs(probs, objects, scale, INPUT_W, INPUT_H);

			vector<STrack> output_stracks = tracker->update(objects);

			for (int i = 0; i < output_stracks.size(); i++)
			{
				vector<float> tlwh = output_stracks[i].tlwh;
				bool vertical = tlwh[2] / tlwh[3] > 1.6;
				if (tlwh[2] * tlwh[3] > 20 && !vertical)
				{
					int tracker_id = output_stracks[i].track_id;
					Scalar s = tracker->get_color(tracker_id);

					auto rect{cv::Rect_<float>(tlwh[0], tlwh[1], tlwh[2], tlwh[3])};
					// g_print("xywh: (%f, %f, %f, %f)\n", rect.x, rect.y, rect.width, rect.height);

					NvDsObjectMeta *obj_meta =
							nvds_acquire_obj_meta_from_pool(batch_meta);
					obj_meta->unique_component_id = meta->unique_id;
					obj_meta->confidence = 0.;

					/* This is an untracked object. Set tracking_id to -1. */
					obj_meta->object_id = UNTRACKED_OBJECT_ID;
					obj_meta->class_id = class_id;

					NvOSD_RectParams &rect_params = obj_meta->rect_params;
					NvOSD_TextParams &text_params = obj_meta->text_params;

					/* Assign bounding box coordinates. */
					rect_params.left = rect.x * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
					rect_params.top = rect.y * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;
					rect_params.width = rect.width * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
					rect_params.height = rect.height * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;

					/* Border of width 3. */
					rect_params.border_width = 3;
					rect_params.has_bg_color = 0;
					// rect_params.border_color = (NvOSD_ColorParams){
					// 		1, 0, 0, 1};
					rect_params.border_color = (NvOSD_ColorParams){
							s[0], s[1], s[2], s[3]};
					// g_print("s: (%f, %f, %f, %f)\n", s[0], s[1], s[2], s[3]);

					/* display_text requires heap allocated memory. */
					gchar *text = g_strdup_printf("%i", tracker_id);
					text_params.display_text = g_strdup(text);
					/* Display text above the left top corner of the object. */
					text_params.x_offset = rect_params.left;
					text_params.y_offset = rect_params.top - 10;
					/* Set black background for the text. */
					text_params.set_bg_clr = 1;
					text_params.text_bg_clr = (NvOSD_ColorParams){
							0, 0, 0, 1};
					/* Font face, size and color. */
					text_params.font_params.font_name = (gchar *)"Serif";
					text_params.font_params.font_size = 11;
					text_params.font_params.font_color = (NvOSD_ColorParams){
							1, 1, 1, 1};
					nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
				}
			}
		}
	}
	use_device_mem = 1 - use_device_mem;
	return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
													gpointer u_data)
{
	GstBuffer *buf = (GstBuffer *)info->data;
	guint num_rects = 0;
	NvDsObjectMeta *obj_meta = NULL;
	guint vehicle_count = 0;
	guint person_count = 0;
	NvDsMetaList *l_frame = NULL;
	NvDsMetaList *l_obj = NULL;
	NvDsDisplayMeta *display_meta = NULL;

	NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

	for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
			 l_frame = l_frame->next)
	{
		NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
		int offset = 0;

		// g_print ("frame (width, height): (%d, %d) \n", frame_meta->source_frame_width, frame_meta->source_frame_height);

		// for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
		//         l_obj = l_obj->next) {
		//     obj_meta = (NvDsObjectMeta *) (l_obj->data);
		//     if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
		//         vehicle_count++;
		//         num_rects++;
		//     }
		//     if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
		//         person_count++;
		//         num_rects++;
		//     }
		// }
		//     display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
		//     NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
		//     display_meta->num_labels = 1;
		//     txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
		//     offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
		//     offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

		//     /* Now set the offsets where the string should appear */
		//     txt_params->x_offset = 10;
		//     txt_params->y_offset = 12;

		//     /* Font , font-color and font-size */
		//     txt_params->font_params.font_name = "Serif";
		//     txt_params->font_params.font_size = 10;
		//     txt_params->font_params.font_color.red = 1.0;
		//     txt_params->font_params.font_color.green = 1.0;
		//     txt_params->font_params.font_color.blue = 1.0;
		//     txt_params->font_params.font_color.alpha = 1.0;

		//     /* Text background color */
		//     txt_params->set_bg_clr = 1;
		//     txt_params->text_bg_clr.red = 0.0;
		//     txt_params->text_bg_clr.green = 0.0;
		//     txt_params->text_bg_clr.blue = 0.0;
		//     txt_params->text_bg_clr.alpha = 1.0;

		//     nvds_add_display_meta_to_frame(frame_meta, display_meta);
	}

	// g_print ("Frame Number = %d Number of objects = %d "
	//         "Vehicle Count = %d Person Count = %d\n",
	//         frame_number, num_rects, vehicle_count, person_count);
	// frame_number++;
	return GST_PAD_PROBE_OK;
}

int main(int argc, char **argv)
{
	cudaSetDevice(DEVICE);

	GstBus *bus;
	GstStateChangeReturn ret;
	GMainLoop *main_loop;
	guint bus_watch_id;
	GstPad *osd_sink_pad = NULL;

	GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
						 *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
						 *nvosd = NULL;

	/* Initialize GStreamer */
	gst_init(&argc, &argv);
	main_loop = g_main_loop_new(NULL, FALSE);

	/* Create gstreamer elements */
	/* Create Pipeline element that will form a connection of other elements */
	pipeline = gst_pipeline_new("dstest1-pipeline");

	/* Source element for reading from the file */
	source = gst_element_factory_make("filesrc", "file-source");

	/* Since the data format in the input file is elementary h264 stream,
    * we need a h264parser */
	h264parser = gst_element_factory_make("h264parse", "h264-parser");

	/* Use nvdec_h264 for hardware accelerated decode on GPU */
	decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");

	/* Create nvstreammux instance to form batches from one or more sources. */
	streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

	pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");

	/* Use convertor to convert from NV12 to RGBA as required by nvosd */
	nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

	/* Create OSD to draw on the converted RGBA buffer */
	nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

	sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");

	if (!pipeline || !source || !h264parser || !decoder || !streammux || !pgie ||
			!nvvidconv || !nvosd || !sink)
	{
		g_printerr("One element could not be created. Exiting.\n");
		return -1;
	}

	/* Set all the necessary properties of the nvinfer element,
    * the necessary ones are : */
	g_object_set(G_OBJECT(pgie),
							 "config-file-path", "../src/pgie_config.txt", NULL);

	/* we set the input filename to the source element */
	char *file = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264";
	g_object_set(G_OBJECT(source), "location", file, NULL);

	g_object_set(G_OBJECT(streammux), "batch-size", 1, NULL);

	g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
							 MUXER_OUTPUT_HEIGHT,
							 "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

	/* we add a message handler */
	bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
	bus_watch_id = gst_bus_add_watch(bus, bus_call, main_loop);
	gst_object_unref(bus);

	// queue = gst_element_factory_make("queue", NULL);

	gst_bin_add_many(GST_BIN(pipeline),
									 source, h264parser, decoder, streammux, pgie, nvvidconv, nvosd, sink, NULL);

	GstPad *sinkpad, *srcpad;
	gchar pad_name_sink[16] = "sink_0";
	gchar pad_name_src[16] = "src";

	sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);
	if (!sinkpad)
	{
		g_printerr("Streammux request sink pad failed. Exiting.\n");
		return -1;
	}

	srcpad = gst_element_get_static_pad(decoder, pad_name_src);
	if (!srcpad)
	{
		g_printerr("Decoder request src pad failed. Exiting.\n");
		return -1;
	}

	if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
	{
		g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
		return -1;
	}

	gst_object_unref(sinkpad);
	gst_object_unref(srcpad);

	/* we link the elements together */
	/* file-source -> h264-parser -> nvh264-decoder ->
    * nvinfer -> nvvidconv -> nvosd -> video-renderer */

	if (!gst_element_link_many(source, h264parser, decoder, NULL))
	{
		g_printerr("Elements could not be linked: 1. Exiting.\n");
		return -1;
	}

	if (!gst_element_link_many(streammux, pgie,
														 nvvidconv, nvosd, sink, NULL))
	{
		g_printerr("Elements could not be linked: 2. Exiting.\n");
		return -1;
	}

	/* Lets add probe to set user meta data at frame level. We add probe to
    * the src pad of the nvinfer element */
	BYTETracker tracker(30, 30);

	GstPad *infer_src_pad = NULL;
	infer_src_pad = gst_element_get_static_pad(pgie, "src");
	if (!infer_src_pad)
	{
		g_print("Unable to get source pad\n");
	}
	else
	{
		gst_pad_add_probe(infer_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
											nvinfer_src_pad_buffer_probe, &tracker, NULL);
	}
	gst_object_unref(infer_src_pad);

	/* Add probe to get informed of the meta data generated, we add probe to
    * the source pad of PGIE's next queue element, since by that time, PGIE's
    * buffer would have had got tensor metadata. */
	// GstPad *queue_src_pad = NULL;
	// queue_src_pad = gst_element_get_static_pad(queue, "src");
	// gst_pad_add_probe(queue_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
	// 									nvinfer_src_pad_buffer_probe, &tracker, NULL);
	// gst_object_unref(queue_src_pad);

	/* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
	//   osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
	//   if (!osd_sink_pad)
	//     g_print ("Unable to get sink pad\n");
	//   else
	//     gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
	//         osd_sink_pad_buffer_probe, NULL, NULL);
	//   gst_object_unref (osd_sink_pad);

	/* Set the pipeline to "playing" state */
	g_print("Now playing: %s\n", argv[1]);
	gst_element_set_state(pipeline, GST_STATE_PLAYING);

	/* Wait till pipeline encounters an error or EOS */
	g_print("Running...\n");
	g_main_loop_run(main_loop);

	/* Out of the main loop, clean up nicely */
	g_print("Returned, stopping playback\n");
	gst_element_set_state(pipeline, GST_STATE_NULL);
	g_print("Deleting pipeline\n");
	gst_object_unref(GST_OBJECT(pipeline));
	g_source_remove(bus_watch_id);
	g_main_loop_unref(main_loop);
}
