package vins_honor.mono;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

import android.Manifest;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;

// for permission check
import androidx.core.app.ActivityCompat;

// View for SurfaceTexture, and Surface
import android.view.View;
import android.view.TextureView;
import android.view.Surface;

// SurfaceTexture and ImageFormat
import android.graphics.SurfaceTexture;
import android.graphics.ImageFormat;

// Android Camera2 API
import android.content.Context;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;

// Handler for Camera
import android.os.Handler;
import android.os.HandlerThread;

// Image and ImageReader to access image data in Surface
import android.media.Image;
import android.media.ImageReader;

// utils
import android.util.Log;
import android.util.Range;
import android.util.Rational;
import android.util.Size;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.lang.Math;

// widget
import android.widget.Toast;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity implements TextureView.SurfaceTextureListener {

    private static final String TAG = "MainActivity";

    // UI widgets
    private TextureView mTextureView;
    private TextView mPositionXText;
    private TextView mPositionYText;
    private TextView mPositionZText;

    // camera2 related
    private CameraDevice mCamera;
    private CaptureRequest.Builder mPreviewBuilder;
    private HandlerThread mHandlerThread;
    private Handler mHandler;
    private ImageReader mImageReader;
    private SurfaceTexture mSfTexture;
    private Surface mSurface;

    private NDKHelper mNDKHelper;

    /**
     * 在 Java 代码层面主要完成:
     *
     * Step1, 主线程中创建 TextureView 和 SurfaceTextureListener 接口, 在接口中开启相机的重复捕获图像请求,
     *    之后便可以执行 ImageReader 的回调函数, 它直接访问呈现到 Surface 中的图像数据, 并利用 NDK 将图像数据传递给 c++ VIO 代码进行处理
     * Step2, 主线程中通过 NDK 初始化整个 VINS 并开启 IMU 传感器, 并将 IMU 数据传递给 c++ VIO 代码进行处理
     * Step3, c++ VIO 代码处理完成之后, 回到主线程 ImageReader 的回调函数中更新 UI 信息
     */

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Camera2 step1: 初始化 Camera 所需的 Handler: 创建并启动线程 "camera_thread", 设置 looper
        mHandlerThread = new HandlerThread( "camera_thread" );
        mHandlerThread.start();
        mHandler = new Handler( mHandlerThread.getLooper() );

        // 绑定该 Activity 对应的 TextureView 和 SurfaceTextureListener 接口
        mTextureView = findViewById( R.id.texture_view );
        mTextureView.setSurfaceTextureListener( this );

        // IMU related widgets
        mPositionXText = findViewById( R.id.position_x );
        mPositionYText = findViewById( R.id.position_y );
        mPositionZText = findViewById( R.id.position_z );

        // 初始化整个 VINS 并开启 IMU 传感器
        mNDKHelper = new NDKHelper();
        mNDKHelper.VINSInit();
    }

    protected void onPause() {
        super.onPause();
        if ( mCamera != null ) {
            mCamera.close();
            mCamera = null;
        }
        if ( mImageReader != null ) {
            mImageReader.close();
            mImageReader = null;
        }
        mNDKHelper.VINSRelease();
    }

    protected void onDestroy() {
        super.onDestroy();
        if ( mCamera != null ) {
            mCamera.close();
            mCamera = null;
        }
        if ( mImageReader != null ) {
            mImageReader.close();
            mImageReader = null;
        }
        mNDKHelper.VINSRelease();
    }

    // 重写 TextureView.SurfaceTextureListener 的四个接口: onSurfaceTextureAvailable,
    // onSurfaceTextureDestroyed, onSurfaceTextureSizeChanged, onSurfaceTextureUpdated
    public void onSurfaceTextureAvailable ( SurfaceTexture surface, int width, int height ) {
        // surface, the surface returned by TextureView.getSurfaceTexture()
        // width/height, the width/height of the surface
        Log.i( TAG, "the width and height of the surfaceTexture: ( " + width + " , " + height + " )" );
        Toast.makeText(MainActivity.this, "the width and height of the surfaceTexture: ( " + width + " , " + height + " )", Toast.LENGTH_SHORT).show();

        try {

            // Camera2 step2: 配置获取的 mSfTexture, 并获取 Surface, 用于图像预览
            mSfTexture = surface;
            // 必须提前设置, 与图像预览窗口大小一致
            mSfTexture.setDefaultBufferSize( mTextureView.getWidth(), mTextureView.getHeight() );
            mSurface = new Surface( mSfTexture );

            // Camera2 step3: 创建 CameraManager 实例, 访问系统资源( 定义之后才能对 Camera2 api 进行操作 )
            CameraManager cameraManager = (CameraManager)getSystemService( Context.CAMERA_SERVICE );

            // 检查权限
            if ( ActivityCompat.checkSelfPermission( this, Manifest.permission.CAMERA )
                    != PackageManager.PERMISSION_GRANTED )
            {
                Log.d( TAG, "No Camera Permission" );
                Toast.makeText(MainActivity.this, "请打开相机权限！", Toast.LENGTH_SHORT).show();
                return;
            }

            // Camera2 step3: 打开相机(not the recording), 进入 cameraDeviceStateCallback 回调, 获取 CameraDevice 实例
            cameraManager.openCamera( "0", cameraDeviceStateCallback, mHandler );
        }
        catch ( CameraAccessException e ) {
            e.printStackTrace();
            Toast.makeText(MainActivity.this, "打开相机失败，请检查相机！", Toast.LENGTH_SHORT).show();
        }
    }
    public boolean onSurfaceTextureDestroyed( SurfaceTexture surface ) {
        return false;
    }
    public void onSurfaceTextureSizeChanged( SurfaceTexture surface, int width, int height ) {}
    public void onSurfaceTextureUpdated( SurfaceTexture surface ) {}

    // Camera2 step4: CameraDevice.StateCallback 回调函数
    private CameraDevice.StateCallback cameraDeviceStateCallback = new CameraDevice.StateCallback() {

        // 重写 onOpened, onDisconnected, onError
        public void onOpened( CameraDevice cameraDevice) {

            try {
                // Camera2 step5: 获取 CameraDevice 实例, 用于相机捕获图像
                mCamera = cameraDevice;

                try {
                    // Camera2 step6: 创建 CaptureRequest.Builder 构建者实例( 建造者模式 ), 用于构建 CaptureRequest 实例( 表示一个相机捕获图像的请求 )
                    mPreviewBuilder = mCamera.createCaptureRequest( CameraDevice.TEMPLATE_PREVIEW ); // // CameraDevice.TEMPLATE_PREVIEW 相机预览请求, 优先保证高帧率
                }
                catch ( CameraAccessException e ) {
                    e.printStackTrace();
                }

                // Camera2 step7: 设置 ImageReader 的监听器: onImageAvailableListener 回调函数, 允许直接访问呈现到 Surface 中的图像数据, 然后用于图像处理
                // !!!注意: 图像尺寸与 SurfaceTexture 大小比例一致，不然显示出来的预览图像会被拉伸变形, 例如这里图像宽高比4:3, 与 SurfaceTexture 一致
                mImageReader = ImageReader.newInstance( 640, 480, ImageFormat.YUV_420_888, 1 );
                mImageReader.setOnImageAvailableListener( onImageAvailableListener, mHandler );

                // Camera2 step8: 创建 CameraCharacteristics 实例, 用于获取对应的 CameraDevices 实例属性, 然后构建者可调整相关属性
                CameraManager cameraManager = (CameraManager)getSystemService( Context.CAMERA_SERVICE );
                CameraCharacteristics cameraProperties = cameraManager.getCameraCharacteristics( "0" );
                // 获取并检查相机曝光补偿能够改变的最小步长 CONTROL_AE_COMPENSATION_STEP
                Rational aeCompStepSize = cameraProperties.get( CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP );
                if ( aeCompStepSize == null ) {
                    Log.e( TAG, "Camera doesn't support setting Auto-Exposure Compensation" );
                    Toast.makeText(MainActivity.this, "相机不支持自动曝光补偿！", Toast.LENGTH_SHORT).show();
                    finish();
                }
                Log.i(TAG, "AE Compensation StepSize: " + aeCompStepSize);
                // 调整曝光补偿
                int aeCompensationInSteps = 0 * aeCompStepSize.getDenominator() / aeCompStepSize.getNumerator();
                mPreviewBuilder.set( CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, aeCompensationInSteps );
                // 设置相机帧率为 30 FPS
                mPreviewBuilder.set( CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<Integer>(30, 30) );

                // Camera2 step9: 在此相机捕获请求的目标列表中添加一个用于输出预览图像的 Surface
                // 它必须包含在 CameraDevice.createCaptureSession() 方法设置的输出 Surface 集合中
                mPreviewBuilder.addTarget( mImageReader.getSurface() );
                List<Surface> outputSurfaces = new ArrayList<>();
                outputSurfaces.add(mImageReader.getSurface());
                // Camera2 step10: 创建一个 CameraCaptureSession 会话,  用于 1 次相机捕获请求(a one-shot capture)或无尽的重复请求(an endlessly repeating use)
                // sessionStateCallback, 接收到关于相机捕获会话的状态更新后执行 CameraCaptureSession 回调函数
                mCamera.createCaptureSession( outputSurfaces, sessionStateCallback, mHandler );

                Toast.makeText(MainActivity.this, "相机准备捕获图像......", Toast.LENGTH_SHORT).show();
            }
            catch ( CameraAccessException e ) {
                e.printStackTrace();
                Toast.makeText(MainActivity.this, "相机暂时无法捕获图像，请检查相机！", Toast.LENGTH_SHORT).show();
            }

        }
        public void onDisconnected( CameraDevice camera ) {}
        public void onError( CameraDevice camera, int error ) {}
    };

    // Camera2 step11: 设置并建立一个相机重复捕获请求
    private CameraCaptureSession.StateCallback sessionStateCallback = new CameraCaptureSession.StateCallback() {
        public void onConfigured( CameraCaptureSession session ) {

            try {
                // CONTROL_AF_MODE 自动对焦模式选项
                // CONTROL_AF_MODE_CONTINUOUS_VIDEO 设置选项为自动对焦模式, 提供一个持续聚焦的 image stream
                mPreviewBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
                // 请求此捕获会话 session 无休止地重复捕获图像
                // CaptureRequest.Builder.build 使用当前的目标 Surface 来设置并建立一个相机重复捕获请求
                session.setRepeatingRequest(mPreviewBuilder.build(), null, mHandler);
                Toast.makeText(MainActivity.this, "相机正在重复捕获图像......", Toast.LENGTH_SHORT).show();
            }
            catch ( CameraAccessException e ) {
                e.printStackTrace();
                Toast.makeText(MainActivity.this, "相机暂时无法重复捕获图像，请检查相机！", Toast.LENGTH_SHORT).show();
            }
        }
        public void onConfigureFailed( CameraCaptureSession session ) {}
    };

    // 当设置好了相机的重复捕获请求模式, 便可以执行ImageReader 监听器: onImageAvailableListener 回调函数
    // 该函数可以直接访问呈现到 Surface 中的图像数据, 然后将图像进行处理并输入给VIO
    private ImageReader.OnImageAvailableListener onImageAvailableListener = new ImageReader.OnImageAvailableListener() {
        public void onImageAvailable(ImageReader reader) {
            // Step1: 获取最新的一帧图像
            Image image = reader.acquireLatestImage();

            if ( image == null ) {
                Toast.makeText(MainActivity.this, "获取到一帧 null 数据图像", Toast.LENGTH_SHORT).show();
                return;
            }
            // YUV图像, 亮度信号Y和色度信号U, V是分离的; 只有Y信号分量而没有U、V信号分量, 表示的图像就是黑白灰度图像
            if ( image.getFormat() != ImageFormat.YUV_420_888 ) {
                Log.e( TAG, "camera image is in wrong format" );
                Toast.makeText(MainActivity.this, "图像数据格式与 YUV_420_888 不符", Toast.LENGTH_SHORT).show();
            }

            // Step2: 取出这帧图像的数据
            Image.Plane imYPlane = image.getPlanes()[0]; // 一帧图像数组 Plane[] 的维度取决于这帧图像的格式, 可以使用 Image.getFormat()进行判断
            // Image.Plane imUPlane = image.getPlanes()[1];
            // Image.Plane imVPlane = image.getPlanes()[2];

            // 获取屏幕旋转的方向, 例如旋转手机竖屏至横屏来保持 landscape orientation, getRotation()返回角度值 Surface.ROTATION_90, 或 Surface.ROTATION_270
            // 这些角度实际是屏幕上所绘制图形的旋转, 与设备物理旋转的方向相反, 如果设备逆时针旋转90度，则要补偿渲染，图像将顺时针旋转90度, 则返回角度值为 Surface.ROTATION_90
            int curRotAngle = getWindowManager().getDefaultDisplay().getRotation();
            boolean isScreenRotated = curRotAngle != Surface.ROTATION_90;

            // Step3: 使用 NDK 将图像数据传递给 c++ 代码进行图像处理, 并输入给VIO, 再结合 ANativeWindow 显示处理完成之后的图像
            //Log.i( TAG, "Transform Image to C++ VIO Code via NDK Function!" );
            mNDKHelper.OnImageAvailable( image.getTimestamp(), image.getWidth(), image.getHeight(),
                                        imYPlane.getBuffer(),  isScreenRotated, mSurface );

            // 在主线程(UI线程)中更新UI信息
            runOnUiThread(new Runnable() {
                public void run() {
                    mNDKHelper.UpdateUIInfo( mPositionXText, mPositionYText, mPositionZText );
                }
            });

            // Step4: 释放这一帧图像数据
            image.close();
        }
    };

}
