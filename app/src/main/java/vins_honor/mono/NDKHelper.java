package vins_honor.mono;

import android.view.Surface;
import java.nio.ByteBuffer;
import android.widget.TextView;

public class NDKHelper {



    private static final String TAG = "NDKHelper";

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    // VINS System
    public static native void VINSInit();
    public static native void VINSRelease();

    // Camera Processing
    public static native void OnImageAvailable( long imgTimestamp, int imgWidth, int imgHeight,
                                                ByteBuffer imgBuffer, boolean isScreenRotated, Surface surface );

    // Update UI Information
    public static native void UpdateUIInfo( TextView positionXText, TextView positionYText, TextView positionZText );
}
