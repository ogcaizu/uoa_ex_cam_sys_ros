/*!
 * @file  external_camera_node.cpp
 * @brief external camera system
 */

#include <ros/ros.h>
#include <external_camera/c_state.h>
#include <external_camera/c_req.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sys/time.h>
#include <time.h>
#include <signal.h>

sig_atomic_t stopFlag = 0; // 強制終了判定用のフラグ

/*!
 * @if jp
 * @brief 強制終了時の後処理(デストラクタ実行)用ハンドラ
 * @else
 * @brief exit handler
 * @endif
 */
void exitHandler(int s){
  ROS_INFO("caught signal %d, ...ros node will now exit...", s);
  stopFlag = 1;
}

/*!
 * @class State
 * @brief State class (Abstract class)
 */
class State{
public:
  // Stateデストラクタ
  virtual ~State(){
  }

  // カメラの状態に関するメッセージのpublish
  virtual void publishCameraState(char* date) = 0;
};

/*!
 * @class StandbyState
 * @brief StandbyState class (derived from State class)
 */
class StandbyState: public State{
private:
  const char* STANDBY; // camera mode
  ros::Publisher standby_pubcl; // ROS Publisher

public:
  // StandbyStateコンストラクタ
  StandbyState(ros::Publisher& standby_pubcl):
    STANDBY("Standby"){
    this->standby_pubcl = standby_pubcl;
    ROS_INFO_STREAM("StandbyState constructor was called.");
  }

  // カメラのStandby状態に関するメッセージのpublish
  void publishCameraState(char* date){
    external_camera::c_state msg;
    msg.time = std::string(date);
    msg.c_mode = STANDBY;

    standby_pubcl.publish(msg);
    ROS_INFO_STREAM("publish a standby state message." );
  }
};

/*!
 * @class MonitorState
 * @brief MonitorState class (derived from State class)
 */
class MonitorState: public State{
private:
  const char* MONITOR; // camera mode
  ros::Publisher monitor_pubcl; // ROS Publisher

  cv::VideoCapture cap; // ビデオキャプチャデバイス操作用オブジェクト

  const int WIDTH; // 背景画像や差分画像の幅 (pixel)
  const int HEIGHT; // 背景画像や差分画像の高さ (pixel)
  const int LATTICE_WIDTH; // 格子幅 (pixel) *画像上の幅であり、実幅は1m
  const double CENTER; // 格子の中心座標 (m) --- x,yそれぞれに適用

  cv::Mat frame, background, undistort_bg, undistort_fg, rotation_01, rotation_fg, parallel, intrinsic, distortion, mapx, mapy, gray, diff, result, affine, resz; // 各種画像変換用matrix
  cv::FileStorage fs; // カメラの歪みデータ読み込み用FileStorage
  const double THRESHOLD; // 画像の二値化時の閾値
  const double MAX_VALUE; // 画像の二値化時の最大値 (最小値は0)

  cv::Point2f img_center; // 画像の回転補正時の中心点
  const double ANGLE_DEG; // 画像の回転補正時の回転角度 (degree)
  const double SCALE; // 画像の回転補正時の拡大比率

  const int REPLACEMENT_TIME; // 背景画像差し替えのための待ち時間 (s)
  int cnt_flg, prev_cnt; // 背景画像変更のためのカウンター類

public:
  // MonitorStateコンストラクタ
  MonitorState(ros::Publisher& monitor_pubcl):
    MONITOR("Monitor"), WIDTH(1920), HEIGHT(1080), LATTICE_WIDTH(180), CENTER(0.5), THRESHOLD(50), MAX_VALUE(255), ANGLE_DEG(-3.0), SCALE(1.0), REPLACEMENT_TIME(10){
    // カメラデバイスの読み込み
    if(!cap.open(0)){
      // 例外1: カメラデバイスの準備ができていない
      throw "cannot open video capture device.";
    } 
    cap.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);

    // 背景画像をグレイスケールで読み込み
    background = cv::imread("./background.png", 0);
    if(background.empty()){
      // 例外2: 背景画像が読み込めない
      throw "cannot open background image.";
    }

    // カメラマウント設置時のズレ補正affine変換のためのmatrix取得
    img_center = cv::Point2f(static_cast<float>(background.cols/2), static_cast<float>(background.rows/2));
    cv::getRotationMatrix2D(img_center, ANGLE_DEG, SCALE).copyTo(affine);

    // カメラの歪みデータの読み込み
    if(!(fs.open("./camera.xml", cv::FileStorage::READ))){
      // 例外3: カメラの歪みを補正するxmlデータが見つからない
      throw "cannot open calibration data file.";
    }
    fs["intrinsic"] >> intrinsic;
    fs["distortion"] >> distortion;

    // 歪みデータを元に画像の歪みMAPを作成
    cv::initUndistortRectifyMap (intrinsic, distortion, cv::Matx33d::eye(), intrinsic, background.size(), CV_32FC1, mapx, mapy);
    // 歪みMAPを使用し背景画像の歪み補正
    cv::remap(background, undistort_bg, mapx, mapy, cv::INTER_AREA);
    // 画像拡張のための行列情報
    parallel = (cv::Mat_<double>(2, 3)<<1.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    // 各種カウンターの初期化
    cnt_flg = 0;
    prev_cnt = 0;

    this->monitor_pubcl = monitor_pubcl;
    ROS_INFO_STREAM("MonitorState constructor was called.");
  }

  // カメラのMonitor状態に関するメッセージのpublish
  void publishCameraState(char* date){
    external_camera::c_state msg;
    external_camera::p_info p_msg;

    msg.position.clear();

    // カメラによる静止画像キャプチャ
    cap >> frame;

    if(frame.empty()){
      // 例外4: 画像キャプチャに失敗
      throw "cannot capture the image.";
    }
    // 歪みMAPを使用し差分画像の歪み補正
    cv::remap(frame, undistort_fg, mapx, mapy, cv::INTER_AREA);
    // 差分画像をグレイスケールに変換
    cvtColor(undistort_fg, gray, CV_BGR2GRAY);
    // 背景画像と差分画像を用いて背景差分取得
    absdiff(gray, undistort_bg, diff);
    // 背景差分画像の二値化
    threshold(diff, result, THRESHOLD, MAX_VALUE, CV_THRESH_BINARY);

    // ぼかし＋収縮＋膨張 (環境光対策)
    medianBlur(result, result, 7); // 7回ぼかし
    erode(result, result, cv::Mat(), cv::Point(-1,-1), 4); // 4回収縮
    dilate(result, result, cv::Mat(), cv::Point(-1,-1), 4); // 4回膨張

    // affine変換で-3度回転 (二値化画像と差分画像)
    cv::warpAffine(result, rotation_01, affine, result.size(), cv::INTER_CUBIC); // 二値化画像の回転
    cv::warpAffine(undistort_fg, rotation_fg, affine, result.size(), cv::INTER_CUBIC); // 差分画像の回転 (見える化システム用)
    // 画像右端を拡張するためのmatrixの取得 (横幅や高さが格子幅で割り切れない場合の解決手段)
    cv::Mat extension = cv::Mat(cv::Size(LATTICE_WIDTH*11, LATTICE_WIDTH*6), CV_8UC3);
    // 画像右端の拡張 (拡張部分は黒の矩形で埋める)
    cv::warpAffine(rotation_01, extension, parallel, extension.size(), CV_INTER_AREA);
    // 1格子が1ピクセルになるよう画像をリサイズ
    resize(extension, resz, cv::Size(), (double)(extension.cols/LATTICE_WIDTH)/(double)(extension.cols), (double)(extension.rows/LATTICE_WIDTH)/(double)(extension.rows), CV_INTER_AREA);

    int cnt_p = 0; // 動体検出格子のカウンター
    char pos_c[32]; // 格子座標の格納用配列
    // 動体が検出された格子のカウントおよび見える化のための処理
    for(int y=0; y<resz.rows; y++){
      for(int x=0; x<resz.cols; x++){
        if(resz.at<unsigned char>(y, x) != 0){
          p_msg.x = (double)x + CENTER;
          p_msg.y = (double)y + CENTER;
          msg.position.push_back(p_msg);
          cnt_p++;

          // 該当の格子に枠付け (見える化のための処理)
          cv::rectangle(rotation_fg, cv::Point(LATTICE_WIDTH * x, LATTICE_WIDTH * y), cv::Point(LATTICE_WIDTH * (x+1), LATTICE_WIDTH * (y+1)), CV_RGB(0, 255, 0), 5, CV_AA);
          // 該当の格子に座標情報付与 (見える化のための処理)
          sprintf(pos_c, "(%3.1lf, %3.1lf)", p_msg.x, p_msg.y);
          cv::putText(rotation_fg, pos_c, cv::Point(x*LATTICE_WIDTH+10, y*LATTICE_WIDTH+100), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 0), 4, CV_AA);
        }
      }
    }

    // REPLACEMENT_TIME間、動体検出格子数cnt_pが同じであれば
    // 最後に撮影されたimageを背景画像として差し替え
    if(cnt_p == prev_cnt){
      cnt_flg++;
    }
    else{
      cnt_flg = 0;
    }
    prev_cnt = cnt_p;
    if(cnt_flg >= REPLACEMENT_TIME){
      cvtColor(undistort_fg, undistort_bg, CV_BGR2GRAY);
      cnt_flg = 0;
    }

    // 無線による画像データの送受信を考慮し、見える化画像を縮小
    resize(rotation_fg, rotation_fg, cv::Size(), 0.3, 0.3, CV_INTER_AREA);
    // 見える化画像(枠付け+座標表示済み画像)のnginxドキュメントルートへの書き出し
    cv::imwrite("/usr/share/nginx/html/latticed_1f1.png", rotation_fg);

    msg.time = std::string(date);
    msg.c_mode = MONITOR;
    msg.num_p = cnt_p;

    monitor_pubcl.publish(msg);
    ROS_INFO_STREAM("publish a monitor state message.");
  }
};

/*!
 * @class ErrorState
 * @brief ErrorState class (derived from State class)
 */
class ErrorState: public State{
private:
  const char* ERROR; // camera state
  ros::Publisher error_pubcl; // ROS Publisher

public:
  // ErrorStateコンストラクタ
  ErrorState(ros::Publisher& error_pubcl):
    ERROR("Error"){
    this->error_pubcl = error_pubcl;
    ROS_INFO_STREAM("ErrorState constructor was called.");
  }

  // カメラのError状態に関するメッセージのpublish
  void publishCameraState(char* date){
    external_camera::c_state msg;
    msg.time = std::string(date);
    msg.c_mode = ERROR;

    error_pubcl.publish(msg);
    ROS_INFO_STREAM("publish a error state message.");
  }
};

/*!
 * @class ExternalCameraNode
 * @brief ExternalCameraNode class (main class as ROS node)
 */
class ExternalCameraNode{
private:
  State* camera_state; // camera state instance

  // camera mode
  const char* STANDBY; // "Standby" mode
  const char* MONITOR; // "Monitor" mode
  const char* ERROR; // "Error" mode

  // ROS topic
  const char* TOPIC_CAMERA_1F1_REQ; // server request for 1F-1 camera
  const char* TOPIC_CAMERA_1F1_STT; // regular transmission of 1F-1 camera status
  //const char* TOPIC_CAMERA_1F2_REQ; // server request for 1F-2 camera
  //const char* TOPIC_CAMERA_1F2_STT; // regular transmission of 1F-2 camera status
  //const char* TOPIC_CAMERA_2F1_REQ; // server request for 2F-1 camera
  //const char* TOPIC_CAMERA_2F1_STT; // regular transmission of 2F-1 camera status

  ros::Subscriber req_subcl; // ROS Subscriber
  ros::Publisher state_pubcl; // ROS Publisher

  char date_tmp[20]; // string for timestamp
  struct tm *date; // 時刻要素を格納するための構造体
  time_t now; // システム時刻を保存するための変数
  int year, month, day, hour, minute, second; // 日時保存のための変数群

  const int FREQUENCY; // publish実行周波数
  const int QUEUE_SIZE; // ROSメッセージング用のQueueサイズ

  // cameraの状態変更 (state patternの要)
  void changeCameraState(State* camera_state){
    delete this->camera_state;
    this->camera_state = camera_state;
  }

  // "Standby"状態において、serverから"Monitor"状態への
  // 変更指示を受け取った際の変更手続き
  void receiveMonitorRequest(){
    try{
      changeCameraState(new MonitorState(state_pubcl));
    }
    catch(const char *e){
      ROS_ERROR("Exception: %s", e);
      receiveInternalError();
      return;
    }
    ROS_INFO_STREAM("changed the camera state into MONITOR.");
  }

  // "Monitor"状態移行時または実行時になんらかの内部エラーを
  // 確認した場合の"Error"状態への変更手続き
  void receiveInternalError(){
    changeCameraState(new ErrorState(state_pubcl));
    ROS_INFO_STREAM("changed the camera state into ERROR.");
  }

  // "Monitor"状態において、serverから"Standby"状態への
  // 変更指示を受け取った際の変更手続き
  void receiveStandbyRequest(){
    changeCameraState(new StandbyState(state_pubcl));
    ROS_INFO_STREAM("changed the camera state into STANDBY.");
  }

  // serverからのrequest取得用コールバック関数
  void requestCallback(const external_camera::c_req::ConstPtr& msg){
    char* command = const_cast<char*>(msg->c_cmd.c_str());
    
    // 1) serverからの指示が"Monitor"で、カメラが"Standby"状態であれば状態変更
    //    = "Monitor"指示が来てもカメラが"Standby"状態以外("Monitor" or "Error")の場合は無視
    // 2) serverからの指示が"Standby"で、カメラが"Monitor"状態であれば状態変更
    //    = "Standby"指示が来てもカメラが"Monitor"状態以外("Standby" or "Error")の場合は無視
    if(strcmp(command, MONITOR) == 0 && typeid(*camera_state) == typeid(StandbyState)){
      receiveMonitorRequest();
    }
    else if(strcmp(command, STANDBY) == 0 && typeid(*camera_state) == typeid(MonitorState)){
      receiveStandbyRequest();
    }
  }

  // timestampの作成
  void makeDateString(){
    time(&now);
    date = localtime(&now);

    year = date->tm_year + 1900;
    month = date->tm_mon + 1;
    day = date->tm_mday;
    hour = date->tm_hour;
    minute = date->tm_min;
    second = date->tm_sec;
    sprintf(date_tmp, "%4d-%02d-%02d %02d:%02d:%02d", year, month, day, hour, minute, second);
  }

public:
  // ExternalCameraNodeコンストラクタ
  ExternalCameraNode():
    STANDBY("Standby"), MONITOR("Monitor"), ERROR("Error"),
    TOPIC_CAMERA_1F1_REQ("sub_topic_1f_1"), TOPIC_CAMERA_1F1_STT("pub_topic_1f_1"),
    //TOPIC_CAMERA_1F2_REQ("sub_topic_1f_2"), TOPIC_CAMERA_1F2_STT("pub_topic_1f_2"),
    //TOPIC_CAMERA_2F1_REQ("sub_topic_2f_1"), TOPIC_CAMERA_2F1_STT("pub_topic_2f_1"),
    FREQUENCY(1), QUEUE_SIZE(100){

    ROS_INFO_STREAM("==> Constructor was called.");

    // メッセージPublish/Subscribeのためのpreset
    ros::NodeHandle nh("~");
    req_subcl = nh.subscribe<external_camera::c_req>(TOPIC_CAMERA_1F1_REQ, QUEUE_SIZE, &ExternalCameraNode::requestCallback, this);
    state_pubcl = nh.advertise<external_camera::c_state>(TOPIC_CAMERA_1F1_STT, QUEUE_SIZE);

    // カメラの初期状態は"Standby"
    camera_state = new StandbyState(state_pubcl);
    ROS_INFO_STREAM("set the camera state into STANDBY.");
  }

  // ExternalCameraNodeデストラクタ
  ~ExternalCameraNode(){
    ROS_INFO_STREAM("==> Destructor was called.");
    // camera_stateインスタンスの削除
    if(camera_state != NULL){
      delete camera_state;
      ROS_INFO_STREAM("camera state instance was deleted.");
    }
    // 見える化用画像データの削除(個人情報保護法対策)
    if(std::remove("/usr/share/nginx/html/latticed_1f1.png") == 0){
      ROS_INFO_STREAM("camera image was deleted.");
    }
    else{
      ROS_INFO_STREAM("camera image could not be found.");
    }
  }

  // ROS node 定期実行ルーチン
  void mainLoop(){
    ros::Rate loop_rate(FREQUENCY);

    // 強制終了時に処理を行うハンドラをセット
    signal(SIGINT, exitHandler);

    // publishの定期実行 (強制終了時はloopを抜け出しexitHandlerを実行)
    while(ros::ok() && stopFlag == 0){
      makeDateString();
      // "Standby", "Monitor", "Error"いずれかの状態でメッセージをPublish
      // "Monitor"状態時にthrowされたエラーはここでcatch
      try{
        camera_state->publishCameraState(date_tmp);
      }
      catch(const char *e){
        ROS_ERROR("Exception: %s", e);
        ROS_INFO_STREAM("Error_Date: " << date_tmp);
        receiveInternalError();
      }

      ros::spinOnce();
      loop_rate.sleep();
    }
  }
};

int main(int argc, char **argv){
  ros::init(argc, argv, "external_camera_node");
  ExternalCameraNode ecn_1F1;
  ecn_1F1.mainLoop();
}
