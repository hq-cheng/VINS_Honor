#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include "system.h"

#define LOG_TAG_NATIVE "native-lip.cc"
// debug logging
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG_NATIVE, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG_NATIVE, __VA_ARGS__)

using namespace std;

// System 实例, 全局变量
std::unique_ptr<System> pSystem;


extern "C" JNIEXPORT void JNICALL
Java_vins_1honor_mono_NDKHelper_VINSInit( JNIEnv* env, jclass obj ) {
    pSystem = std::unique_ptr<System>( new System );
    //LOGI( "Successfully Created VINS System Object!" );
    // 对整个 VINS 进行初始化, 包括设置相关参数, 开启 IMU 传感器, 开启后端处理线程等
    pSystem->Init();
}

extern "C" JNIEXPORT void JNICALL
Java_vins_1honor_mono_NDKHelper_VINSRelease( JNIEnv *env, jclass obj ) {
    LOGI("Stopping IMU SensorEvents Before Pause Triggered!");
    pSystem->ImuStopUpdate();
}

extern "C" JNIEXPORT void JNICALL
Java_vins_1honor_mono_NDKHelper_OnImageAvailable( JNIEnv* env, jclass obj,
        jlong imgTimestamp, jint imgWidth, jint imgHeight,
        jobject imgBuffer, jboolean isScreenRotated, jobject surface)
{
    // 创建 ANativeWindow 实例用于显示VIO处理之后的图像
    // Return the ANativeWindow associated with a Java Surface object for interacting with it through native code
    ANativeWindow * window = ANativeWindow_fromSurface(env, surface);
    ANativeWindow_acquire(window);
    // Struct that represents a windows buffer
    ANativeWindow_Buffer buffer;
    // Change the format and size of the window buffers.
    ANativeWindow_setBuffersGeometry(window, imgWidth, imgHeight, WINDOW_FORMAT_RGBA_8888);
    // Lock the window's next drawing surface for writing
    if (int32_t err = ANativeWindow_lock(window, &buffer, NULL)) {
        LOGE( "ANativeWindow_lock failed with error code: %d\n", err );
        ANativeWindow_release(window);
        return;
    }

    // 获取这一帧图像时间戳, 并进行简单的预处理
    double imgHeader = imgTimestamp / 1000000000.0;
    const double offset = 1.0 / 30.0 / 2.0 - 1.0 / 100.0 / 2.0;
    imgHeader += offset;

    // 获取这一帧图像数据, 并进行简单的预处理
    uint8_t *srcLumaPtr = reinterpret_cast<uint8_t *>(env->GetDirectBufferAddress(imgBuffer));
    if ( srcLumaPtr == nullptr ) {
        LOGE( "Image Direct Buffer Return NULL Pointer!" );
        return;
    }

    cv::Mat imgYUV( imgHeight, imgWidth, CV_8UC1, srcLumaPtr );
    if ( isScreenRotated ) {
        cv::rotate( imgYUV, imgYUV, cv::ROTATE_180 );
    }
    cv::Mat imgRgba( imgHeight, imgWidth, CV_8UC4 );
    cv::cvtColor( imgYUV, imgRgba, cv::COLOR_GRAY2BGRA ); // YUV -> RGBA
    //LOGI( "Start to Update Image via VIO" );
    pSystem->ImageStartUpdate( imgRgba, imgHeader, isScreenRotated );

    // copy to TextureView surface
    uint8_t * outPtr = reinterpret_cast<uint8_t *>(buffer.bits);
    cv::Mat imgToDisp( imgHeight, buffer.stride, CV_8UC4, outPtr );

    uchar *dbuf = imgToDisp.data;
    uchar *sbuf = imgRgba.data;
    for (int i = 0; i < imgRgba.rows; i++) {
        dbuf = imgToDisp.data + i * buffer.stride * 4;
        memcpy(dbuf, sbuf, imgRgba.cols * 4);
        sbuf += imgRgba.cols * 4;
    }

    // 释放 ANativeWindow 实例
    // Unlock the window's drawing surface after previously locking it, posting the new buffer to the display
    ANativeWindow_unlockAndPost(window);
    // Remove a reference that was previously acquired with ANativeWindow_acquire()
    ANativeWindow_release(window);
}

extern "C" JNIEXPORT void JNICALL
Java_vins_1honor_mono_NDKHelper_UpdateUIInfo( JNIEnv* env, jclass obj,
        jobject positionXText, jobject positionYText, jobject positionZText )
{
    // Get the method handles
    jclass tvClass = env->FindClass("android/widget/TextView");
    jmethodID setTextID = env->GetMethodID(tvClass, "setText", "(Ljava/lang/CharSequence;)V");

    pSystem->m_ui.lock();
    if( pSystem->tvXText.empty() == false ) {
        env->CallVoidMethod(positionXText, setTextID, env->NewStringUTF(pSystem->tvXText.c_str()));
        env->CallVoidMethod(positionYText, setTextID, env->NewStringUTF(pSystem->tvYText.c_str()));
        env->CallVoidMethod(positionZText, setTextID, env->NewStringUTF(pSystem->tvZText.c_str()));
    }
    pSystem->m_ui.unlock();
}


