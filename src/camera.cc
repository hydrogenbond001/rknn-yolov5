#include <opencv2/opencv.hpp>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <fstream>

#include "RgaUtils.h"
#include "postprocess.h"
#include "rknn_api.h"
#include "preprocess.h"

#define PERF_WITH_POST 1

// Function prototypes
static void dump_tensor_attr(rknn_tensor_attr *attr);
double __get_us(struct timeval t);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i)
  {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp)
  {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0)
  {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL)
  {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp)
  {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
  FILE *fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++)
  {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}


int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s <rknn model>\n", argv[0]);
        return -1;
    }

    int ret;
    rknn_context ctx;
    size_t actual_size = 0;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    const float nms_threshold = NMS_THRESH;      // Default NMS threshold
    const float box_conf_threshold = BOX_THRESH; // Default confidence threshold
    struct timeval start_time, stop_time;
    char *model_name = (char *)argv[1];

    // Initialize RKNN context
    printf("Loading model...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_query error ret=%d\n", ret);
        return -1;
    }
    printf("SDK version: %s, Driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query error ret=%d\n", ret);
        return -1;
    }
    printf("Model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("Model is NCHW input format\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("Model is NHWC input format\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    printf("Model input height=%d, width=%d, channel=%d\n", height, width, channel);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;


    // 获取并打印OpenCV的构建信息
    //std::string build_info = cv::getBuildInformation();
    //std::cout << build_info << std::endl;
    // Initialize camera
    //cv::VideoCapture cap(0); // 0是默认摄像头的ID
    //cv::VideoCapture cap;
    const char* video_path = argv[2]; // 从命令行获取视频文件路径
    if (!std::ifstream(video_path))
    {
        printf("Video file does not exist: %s\n", video_path);
        return -1;
    }
    //printf("Attempting to open video file: %s\n", video_path);
    cv::VideoCapture cap(video_path); // 读取视频文件
    //cap.read(video_path);

    if (!cap.isOpened())
    {
        printf("Error opening video capture\n");
        return -1;
    }

    cv::Mat img;
    while (true)
    {
        cap >> img;
        if (img.empty())
        {
            printf("Empty frame\n");
            break;
        }

        //cv::Mat img;
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        cv::Size target_size(width, height);
        cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
        BOX_RECT pads;
        memset(&pads, 0, sizeof(BOX_RECT));
        float scale_w = (float)target_size.width / img.cols;
        float scale_h = (float)target_size.height / img.rows;

        if (img.cols != width || img.rows != height)
        {
            // Resize image
            float min_scale = std::min(scale_w, scale_h);
            scale_w = min_scale;
            scale_h = min_scale;
            letterbox(img, resized_img, pads, min_scale, target_size);
        }
        else
        {
            resized_img = img;
        }

        inputs[0].buf = resized_img.data;

        gettimeofday(&start_time, NULL);
        rknn_inputs_set(ctx, io_num.n_input, inputs);

        rknn_output outputs[io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < io_num.n_output; i++)
        {
            outputs[i].want_float = 0;
        }

        // Run inference
        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        gettimeofday(&stop_time, NULL);
        printf("Inference time: %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

        // Post-process
        detect_result_group_t detect_result_group;
        std::vector<float> out_scales;
        std::vector<int32_t> out_zps;
        for (int i = 0; i < io_num.n_output; ++i)
        {
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }
        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                     box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

        // Draw results
        for (int i = 0; i < detect_result_group.count; i++)
        {
            detect_result_t *det_result = &(detect_result_group.results[i]);
            char text[256];
            sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
            printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
                   det_result->box.right, det_result->box.bottom, det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3);
            cv::putText(img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        }

        // Show image
        cv::imshow("Detection", img);
        
        if (cv::waitKey(1) >= 0) break; // Break on any key press
    }

    // Cleanup
    rknn_destroy(ctx);
    cap.release();
    cv::destroyAllWindows();
    if (model_data)
    {
        free(model_data);
    }

    return 0;
}