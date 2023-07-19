#include <opencv2/opencv.hpp>

#define USE_SOBEL
#define USE_NEON
#define USE_MULTI_THREAD
#define USE_MULTI_FRAME
#define PERF

int main() {
  // 输出系统信息
  std::cout << "          NEON: ";
#ifdef USE_NEON
  std::cout << "ON" << std::endl;
  cv::setUseOptimized(true);
#else
  std::cout << "OFF" << std::endl;
  cv::setUseOptimized(false);
#endif

  std::cout << "  MULTI_THREAD: ";

#ifdef USE_MULTI_THREAD
  std::cout << "ON" << std::endl;
  cv::setNumThreads(4);
#else
  std::cout << "OFF" << std::endl;
  cv::setNumThreads(1);
#endif

  std::cout << "   MULTI_FRAME: ";
#ifdef USE_MULTI_FRAME
  std::cout << "ON" << std::endl;
#else
  std::cout << "OFF" << std::endl;
#endif

#ifdef USE_SOBEL
  std::cout << "EDGE DETECTION: SOBEL" << std::endl;
#else
  std::cout << "EDGE DETECTION: CANNY" << std::endl;
#endif

  std::cout << "          PERF: ";
#ifdef PERF
  std::cout << "ON" << std::endl;
#else
  std::cout << "OFF" << std::endl;
#endif

  // 捕获视频
  cv::VideoCapture capture("./data/video_1.mp4");
  // cv::VideoCapture capture("http://admin:admin@192.168.31.183:8081");

  // 设置窗口大小
  cv::namedWindow("EDGES", cv::WINDOW_NORMAL);
  cv::resizeWindow("EDGES", 1280, 720);

  cv::namedWindow("FRAME", cv::WINDOW_NORMAL);
  cv::resizeWindow("FRAME", 1280, 720);

  cv::namedWindow("TRANSFORMED", cv::WINDOW_NORMAL);
  cv::resizeWindow("TRANSFORMED", 1280, 720);

  // 检查视频是否成功打开
  if (!capture.isOpened()) {
    std::cerr << "Error opening video stream" << std::endl;
    return -1;
  }

  // 编码格式为YUYV422
  capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
  // 宽度<=1280
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  // 高度<=720
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  // 帧率为 30
  capture.set(cv::CAP_PROP_FPS, 30);

  cv::Mat frame, edges, gray, lines;

  int state = 0;
  int count = 0;

#ifdef PERF
  uint64_t frame_count = 0;
  uint64_t perspective_transform_time = 0;
  uint64_t edge_detection_time = 0;
  uint64_t hough_lines_time = 0;
  uint64_t total_time = 0;
#endif

  while (true) {
#ifdef USE_MULTI_FRAME
    // 多帧合一
    cv::Mat frame_1, frame_2;
    capture >> frame_1;
    capture >> frame_2;
    if (frame_1.empty() || frame_2.empty())
      break;
    frame = frame_1 + frame_2;
    frame = frame / 2.0f;
#else
    capture >> frame;
#endif
    if (frame.empty())
      break;

#ifdef PERF
    // 记录开始时间
    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
#endif
    // 透视变换
    cv::Point2f input_quad[4], output_quad[4];
    // 设置输入四边形的四个顶点
    input_quad[0] = cv::Point2f(0, frame.rows);
    input_quad[1] = cv::Point2f(frame.cols, frame.rows);
    input_quad[2] = cv::Point2f(0.65 * frame.cols, 0.65 * frame.rows);
    input_quad[3] = cv::Point2f(0.35 * frame.cols, 0.65 * frame.rows);
    // 设置输出四边形的四个顶点为整个图像的四个角
    output_quad[0] = cv::Point2f(0, frame.rows);
    output_quad[1] = cv::Point2f(frame.cols, frame.rows);
    output_quad[2] = cv::Point2f(frame.cols, 0);
    output_quad[3] = cv::Point2f(0, 0);
    // 获取透视变换矩阵
    cv::Mat lambda = getPerspectiveTransform(input_quad, output_quad);
    // 对原始图像应用透视变换
    cv::Mat frame_transformed;
    cv::warpPerspective(frame, frame_transformed, lambda, frame.size());

#ifdef PERF
    // 记录结束
    auto end = std::chrono::high_resolution_clock::now();
    // 计算透视变换时间
    perspective_transform_time +=
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
#endif

#ifdef PERF
    // 记录开始时间
    start = std::chrono::high_resolution_clock::now();
#endif
    // 边缘检测
#ifdef USE_SOBEL
    cv::Mat grad_x, grad_y;
    cv::Sobel(frame_transformed, grad_x, CV_16S, 1, 0, 3, 1, 0,
              cv::BORDER_DEFAULT);
    cv::Sobel(frame_transformed, grad_y, CV_16S, 0, 1, 3, 1, 0,
              cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);
    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, edges);
    // 二元值，使边缘更明显
    cv::threshold(edges, edges, 40, 255, cv::THRESH_BINARY);
    // 转换为灰度图像
    cv::cvtColor(edges, edges, cv::COLOR_BGR2GRAY);
#else
    // 转换为灰度图像
    cv::cvtColor(frame_transformed, gray, cv::COLOR_BGR2GRAY);
    // 高斯滤波
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    cv::Canny(gray, edges, 30, 80);
#endif

#ifdef PERF
    // 记录结束
    end = std::chrono::high_resolution_clock::now();
    // 计算边缘检测时间
    edge_detection_time +=
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
#endif

    cv::imshow("EDGES", edges);

#ifdef PERF
    // 记录开始时间
    start = std::chrono::high_resolution_clock::now();
#endif
    // 霍夫直线检测
    std::vector<cv::Vec4i> linesP;
    cv::HoughLinesP(edges, linesP, 2, CV_PI / 180, 50, 5, 100);

#ifdef PERF
    // 记录结束
    end = std::chrono::high_resolution_clock::now();
    // 计算霍夫直线检测时间
    hough_lines_time +=
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
#endif

    // for (int i = 0; i < linesP.size(); i++) {
    //   cv::Vec4i l = linesP[i];
    //   cv::line(frame_transformed, cv::Point(l[0], l[1]), cv::Point(l[2],
    //   l[3]),
    //            cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    // }

    // 对检测到的直线进行筛选
    std::vector<cv::Vec4i> selected_lines;

    // 计算图像 x 轴中点
    int mid = frame_transformed.cols / 2;
    // 左右直线最长长度
    double max_l = 0, max_r = 0;
    // 所选择的左右直线
    cv::Vec4i line_l, line_r;
    for (auto l : linesP) {
      double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
      // 计算斜率
      double slope = (y2 - y1) / (x2 - x1 + 1e-6);
      // 计算长度
      double length = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
      if (abs(slope) < 1 || abs(slope) > 6) {
        // 去除斜率过小或过大的直线
        continue;
      }
      if ((slope > 0 && x1 > mid && x2 > mid && length > max_r) ||
          (slope < 0 && x1 < mid && x2 < mid && length > max_l)) {
        selected_lines.push_back(l);
        if (slope > 0) {
          // 在右侧选择一条最长的直线
          max_r = length;
          line_r = l;
        } else {
          // 在左侧选择一条最长的直线
          max_l = length;
          line_l = l;
        }
      }
    }

    // 绘制右侧直线，蓝色
    line(frame_transformed, cv::Point(line_r[0], line_r[1]),
         cv::Point(line_r[2], line_r[3]), cv::Scalar(255, 0, 0), 3,
         cv::LINE_AA);
    // 绘制左侧直线，绿色
    line(frame_transformed, cv::Point(line_l[0], line_l[1]),
         cv::Point(line_l[2], line_l[3]), cv::Scalar(0, 255, 0), 3,
         cv::LINE_AA);

    // 对线进行逆透视变换，叠加到原始图像
    cv::Mat lambda_inv = getPerspectiveTransform(output_quad, input_quad);
    cv::Mat frame_inv;
    cv::warpPerspective(frame_transformed, frame_inv, lambda_inv, frame.size());
    cv::addWeighted(frame, 0.5, frame_inv, 0.5, 0, frame);

    // 判断是否偏离车道

    float slope_l =
        abs((line_l[3] - line_l[1]) / (line_l[2] - line_l[0] + 1e-6));
    float slope_r =
        abs((line_r[3] - line_r[1]) / (line_r[2] - line_r[0] + 1e-6));

    float slope_diff = slope_r - slope_l;

    int curr_state = 0;

    if (abs(slope_diff) < 2) {
      curr_state = 0;
    } else if (slope_r < slope_l) {
      curr_state = 1;
    } else {
      curr_state = 2;
    }

    // 状态计数，防止直线检测出现的抖动造成误报警
    if (curr_state == state) {
      count++;
    } else {
      count = 0;
      state = curr_state;
    }

    if (count > 3) {
      // 状态累计三帧以上认为是真实状态
      if (state == 1) {
        cv::putText(frame_transformed, "Turn Right", cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
      } else if (state == 2) {
        cv::putText(frame_transformed, "Turn Left", cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
      } else {
        cv::putText(frame_transformed, "Go Straight", cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
      }
    } else {
      // 状态累计三帧以下认为是误报警
      cv::putText(frame_transformed, "Go Straight", cv::Point(50, 50),
                  cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }

    // 显示原图像和变换后的图像
    cv::imshow("FRAME", frame);
    cv::imshow("TRANSFORMED", frame_transformed);

#ifdef PERF
    // 记录结束时间
    end = std::chrono::high_resolution_clock::now();

    // 计算总时间
    auto dur = end - total_start;
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

    total_time += ms;
    frame_count++;
#endif

    // 按'q'退出
    char c = (char)cv::waitKey(30);
    if (c == 'q')
      break;
  }

#ifdef PERF
  // 输出总时间和总帧数
  std::cout << "Total frames: " << frame_count << std::endl;

  std::cout << "Processing Time: " << std::endl;
  std::cout << "\tPerspective Trasform: " << perspective_transform_time << "ms"
            << std::endl;
  std::cout << "\tEdge Detection: " << edge_detection_time << "ms" << std::endl;
  std::cout << "\tHough Transform: " << hough_lines_time << "ms" << std::endl;
  std::cout << "Total Time: " << total_time << "ms" << std::endl;

  // 计算处理每一帧的时间
  std::cout << "Average Processing Time: " << std::endl;
  std::cout << "\tPerspective Trasform: "
            << (double)perspective_transform_time / (double)frame_count << "ms"
            << std::endl;
  std::cout << "\tEdge Detection: "
            << (double)edge_detection_time / (double)frame_count << "ms"
            << std::endl;
  std::cout << "\tHough Transform: "
            << (double)hough_lines_time / (double)frame_count << "ms"
            << std::endl;
  std::cout << "Average Total Time: "
            << (double)total_time / (double)frame_count << "ms" << std::endl;

#endif

  capture.release();
  cv::destroyAllWindows();

  return 0;
}
