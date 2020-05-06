package com.demo.nightvision;

public class jniTools {

    static {
        System.loadLibrary("native-lib");
    }

    //jin图像方法
    public static  native void getEdge(Object bitmap);

}
