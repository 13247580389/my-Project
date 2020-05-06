#include <jni.h>
#include <opencv2/core/base.hpp>
#include <android/bitmap.h>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <jni.h>

#include <android/log.h>
#define TAG    "myhello-jni-test" // 这个是自定义的LOG的标识
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__) // 定义LOGD类型

#define pc(image, x, y, c) image->imageData[(image->widthStep * y) + (image->nChannels * x) + c]
#define INT_PREC 1024.0
#define INT_PREC_BITS 10
inline double int2double(int x) { return (double)x / INT_PREC; }
inline int double2int(double x) { return (int)(x * INT_PREC + 0.5); }
inline int int2smallint(int x) { return (x >> INT_PREC_BITS); }
inline int int2bigint(int x) { return (x << INT_PREC_BITS); }

using namespace std;
using namespace cv;


/* 函数: CreateKernel * 说明：创建一个标准化的一维高斯核；*/
vector<double> CreateKernel(double sigma)
{
    int i, x, filter_size;
    //方法返回的double数组
    vector<double> filter;
    double sum;
    // 为sigma设定上限
    if ( sigma > 300 ) sigma = 300;
    // 获取需要的滤波尺寸；
    filter_size = (int)floor(sigma*6) / 2;
    filter_size = filter_size * 2 + 1;
    // 计算指数
    sum = 0;
    for (i = 0; i < filter_size; i++)
    {
        //一维高斯函数的实现
        double tmpValue;
        x = i - (filter_size / 2);
        tmpValue = exp( -(x*x) / (2*sigma*sigma) );
        filter.push_back(tmpValue);
        //归一化
        sum += tmpValue;
    }
    // 归一化计算
    for (i = 0, x; i < filter_size; i++)
        filter[i] /= sum;
    return filter;
}

/* 函数: CreateFastKernel 说明：创建一个近似浮点的整数类型（左移8bits）的快速高斯核；*/
vector<int> CreateFastKernel(double sigma)
{
    vector<double> fp_kernel; // 存放创建内核
    vector<int> kernel; // 存放转换的整数类型的内核
    int i, filter_size;
    // sigma设置上限
    if ( sigma > 300 ) sigma = 300;

    // 获取需要的滤波尺寸，且强制为奇数；
    filter_size = (int)floor(sigma*6) / 2;
    filter_size = filter_size * 2 + 1;

    // 创建内核
    fp_kernel = CreateKernel(sigma);

    // double内核转为int型
    for (i = 0; i < filter_size; i++)
    {
        int tmpValue;
        tmpValue = double2int(fp_kernel[i]);
        kernel.push_back(tmpValue);
    }
    return kernel;
}

/* 函数：FilterGaussian * 说明：通过内核计算高斯卷积，内核由sigma值得到，且在内核两端值相等；*/
//#define pc(image, x, y, c) image->imageData[(image->widthStep * y) + (image->nChannels * x) + c]
void FilterGaussian(IplImage* img, double sigma)
{
    int i, j, k, source, filter_size;
    vector<int> kernel;
    IplImage* temp;
    int v1, v2, v3;
    // 设置上限
    if ( sigma > 300 ) sigma = 300;
    // 获取需要的滤波尺寸，且强制为奇数；
    filter_size = (int)floor(sigma*6) / 2;
    filter_size = filter_size * 2 + 1;
    // 创建内核
    kernel = CreateFastKernel(sigma);
    //创建与原图像大小一样的临时图像
    temp = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels);
    // X轴滤波
    for (j = 0; j < temp->height; j++)
    {
        for (i = 0; i < temp->width; i++)
        {
            // 内层循环已经展开
            v1 = v2 = v3 = 0;
            for (k = 0; k < filter_size; k++)
            {
                source = i + filter_size / 2 - k;

                if (source < 0) source *= -1;
                if (source > img->width - 1) source = 2*(img->width - 1) - source;

                v1 += kernel[k] * (unsigned char)pc(img, source, j, 0);
                if (img->nChannels == 1) continue;
                v2 += kernel[k] * (unsigned char)pc(img, source, j, 1);
                v3 += kernel[k] * (unsigned char)pc(img, source, j, 2);
            }

            // 处理后的数据放入临时图像中
            pc(temp, i, j, 0) = (char)int2smallint(v1);
            if (img->nChannels == 1) continue;
            pc(temp, i, j, 1) = (char)int2smallint(v2);
            pc(temp, i, j, 2) = (char)int2smallint(v3);

        }
    }

    // Y轴滤波
    for (j = 0; j < img->height; j++)
    {
        for (i = 0; i < img->width; i++)
        {
            v1 = v2 = v3 = 0;
            for (k = 0; k < filter_size; k++)
            {
                source = j + filter_size / 2 - k;

                if (source < 0) source *= -1;
                if (source > temp->height - 1) source = 2*(temp->height - 1) - source;

                v1 += kernel[k] * (unsigned char)pc(temp, i, source, 0);
                if (img->nChannels == 1) continue;
                v2 += kernel[k] * (unsigned char)pc(temp, i, source, 1);
                v3 += kernel[k] * (unsigned char)pc(temp, i, source, 2);
            }

            // 处理后的数据放入临时图像中
            pc(img, i, j, 0) = (char)int2smallint(v1);
            if (img->nChannels == 1) continue;
            pc(img, i, j, 1) = (char)int2smallint(v2);
            pc(img, i, j, 2) = (char)int2smallint(v3);

        }
    }
    //释放临时图像
    cvReleaseImage( &temp );
}
/*= 函数：FilterGaussian * 说明：通过内核计算高斯卷积，内核由sigma值得到，且在内核两端值相等；*/
void FilterGaussian(Mat src, Mat &dst, double sigma)
{
    IplImage tmp_ipl;
    tmp_ipl = src;
    FilterGaussian(&tmp_ipl, sigma);
    dst = cvarrToMat(&tmp_ipl);
}
/** 函数：FastFilter* 说明：给出任意大小的sigma值，都可以通过使用图像金字塔与可分离滤波器计算高斯卷积；*/
void FastFilter(IplImage *img, double sigma)
{
    int filter_size;
    // 设置上限
    if ( sigma > 300 ) sigma = 300;
    // 获取需要的滤波尺寸，且强制为奇数；
    filter_size = (int)floor(sigma*6) / 2;
    filter_size = filter_size * 2 + 1;
    // 如果3 * sigma小于一个像素，则直接退出
    if(filter_size < 3) return;
    // 处理方式：(1) 滤波  (2) 高斯光滑处理  (3) 递归处理滤波器大小  //高斯滤波方法FilterGaussian(img, sigma);
    if (filter_size < 10) {
#ifdef USE_EXACT_SIGMA
        FilterGaussian(img, sigma);
#else
        cvSmooth( img, img, CV_GAUSSIAN, filter_size, filter_size );
#endif
    }
    else
    {
        if (img->width < 2 || img->height < 2) return;
        IplImage* sub_img = cvCreateImage(cvSize(img->width / 2, img->height / 2), img->depth, img->nChannels);
        cvPyrDown( img, sub_img ); //使用Gaussian金字塔分解对输入图像向下采样
        FastFilter( sub_img, sigma / 2.0 ); //卷积处理
        cvResize( sub_img, img, CV_INTER_LINEAR ); //调整sub_img图像的尺寸与img一样
        cvReleaseImage( &sub_img ); //释放sub_img暂时图像
    }
}
void FastFilter(Mat src, Mat &dst, double sigma)
{
    IplImage tmp_ipl;
    tmp_ipl = src;
    FastFilter(&tmp_ipl, sigma);
    dst = cvarrToMat(&tmp_ipl);
}
/* 函数：Retinex * 说明：单通道SSR方法，基础Retinex复原算法。原图像和被滤波的图像需要被转换到*/
void Retinex(IplImage *img, double sigma, int gain, int offset)
{
    IplImage *A, *fA, *fB, *fC;

    // Initialize temp images
    // 初始化缓存图像
    fA = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fB = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fC = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);

    // Compute log image
    // 计算对数图像
    cvConvert( img, fA );
    cvLog( fA, fB );

    // Compute log of blured image
    // 计算滤波后模糊图像的对数图像
    A = cvCloneImage( img );
    FastFilter( A, sigma );
    cvConvert( A, fA );
    cvLog( fA, fC );

    // Compute difference
    // 计算两图像之差
    cvSub( fB, fC, fA );

    // Restore
    // 恢复图像
    cvConvertScale( fA, img, gain, offset);

    // Release temp images
    // 释放缓存图像
    cvReleaseImage( &A );
    cvReleaseImage( &fA );
    cvReleaseImage( &fB );
    cvReleaseImage( &fC );

}
void Retinex(Mat src, Mat &dst, double sigma, int gain, int offset)
{
    IplImage tmp_ipl;
    tmp_ipl = src;
    Retinex(&tmp_ipl, sigma, gain, offset);
    dst = cvarrToMat(&tmp_ipl);
}
/* 函数：MultiScaleRetinex* 说明：多通道MSR算法。原图像和一系列被滤波的图像转换到对数域，并与带权重的原图像做减运算。*/
void MultiScaleRetinex(IplImage *img, vector<double> weights, vector<double> sigmas, int gain, int offset)
{
    int i;
    double weight;
    int scales = sigmas.size();
    IplImage *A, *fA, *fB, *fC;

    // Initialize temp images
    fA = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fB = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fC = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);


    // Compute log image
    cvConvert( img, fA );
    cvLog( fA, fB );

    // Normalize according to given weights
    for (i = 0, weight = 0; i < scales; i++)
        weight += weights[i];

    if (weight != 1.0) cvScale( fB, fB, weight );

    // Filter at each scale
    for (i = 0; i < scales; i++)
    {
        A = cvCloneImage( img );
        double tmp = sigmas[i];
        FastFilter( A, tmp);

        cvConvert( A, fA );
        cvLog( fA, fC );
        cvReleaseImage( &A );

        // Compute weighted difference
        cvScale( fC, fC, weights[i] );
        cvSub( fB, fC, fB );
    }

    // Restore
    cvConvertScale( fB, img, gain, offset);

    // Release temp images
    cvReleaseImage( &fA );
    cvReleaseImage( &fB );
    cvReleaseImage( &fC );
}
void MultiScaleRetinex(Mat src, Mat &dst, vector<double> weights, vector<double> sigmas, int gain, int offset)
{
    IplImage tmp_ipl;
    tmp_ipl = src;
    MultiScaleRetinex(&tmp_ipl, weights, sigmas, gain, offset);
    dst = cvarrToMat(&tmp_ipl);
}
/* 函数：MultiScaleRetinexCR 说明：MSRCR算法，MSR算法加上颜色修复。原图像和一系列被滤波的图像转换到对数域，并与带权重的原图像做减运算。*/
void MultiScaleRetinexCR(IplImage *img, vector<double> weights, vector<double> sigmas,int gain, int offset, double restoration_factor, double color_gain)
{
	int i;
	double weight;
    int scales = sigmas.size();
	IplImage *A, *B, *C, *fA, *fB, *fC, *fsA, *fsB, *fsC, *fsD, *fsE, *fsF;

    // 初始化缓存图像
	fA = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
	fB = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
	fC = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
	fsA = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
	fsB = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
	fsC = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
	fsD = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
	fsE = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
	fsF = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);

    // 计算对数图像
	cvConvert( img, fB );
    //计算每个数组元素绝对值的自然对数
	cvLog( fB, fA );
    // 依照权重归一化
	for (i = 0, weight = 0; i < scales; i++)
		weight += weights[i];
    // 计算权重后两图像之差
	if (weight != 1.0) cvScale( fA, fA, weight );
    // 各尺度上进行滤波操作
	for (i = 0; i < scales; i++) {
        //克隆img缓存图像
		A = cvCloneImage( img );
		FastFilter( A, sigmas[i] );
        //cvConvert(src,dst)执行两个操作：将src图像数据类型改变为dst图像数据类型；将src的数据赋值到dst
		cvConvert( A, fB );
		//计算每个数组元素绝对值的自然对数
		cvLog( fB, fC );
        // 释放缓存图像
		cvReleaseImage( &A );
        // 计算权重后两图像之差
		cvScale( fC, fC, weights[i] );
		//矩阵减法运算
		cvSub( fA, fC, fA );
	}

    // 颜色修复
	if (img->nChannels > 1) {
	    //创建缓存图像
		A = cvCreateImage(cvSize(img->width, img->height), img->depth, 1);
		B = cvCreateImage(cvSize(img->width, img->height), img->depth, 1);
		C = cvCreateImage(cvSize(img->width, img->height), img->depth, 1);
        // 将图像分割为若干通道，类型转换为浮点型，并存储通道数据之和
		cvSplit( img, A, B, C , NULL );
		cvConvert( A, fsA );
		cvConvert( B, fsB );
		cvConvert( C, fsC );
        // 释放缓存图像
		cvReleaseImage( &A );
		cvReleaseImage( &B );
		cvReleaseImage( &C );
        // 求和
		cvAdd( fsA, fsB, fsD );
		cvAdd( fsD, fsC, fsD );
        // 带权重矩阵归一化
		cvDiv( fsA, fsD, fsA, restoration_factor);
		cvDiv( fsB, fsD, fsB, restoration_factor);
		cvDiv( fsC, fsD, fsC, restoration_factor);
        //使用线性变换转换数组
		cvConvertScale( fsA, fsA, 1, 1 );
		cvConvertScale( fsB, fsB, 1, 1 );
		cvConvertScale( fsC, fsC, 1, 1 );

        // 带权重矩阵求对数
		cvLog( fsA, fsA );
		cvLog( fsB, fsB );
		cvLog( fsC, fsC );

        // 将Retinex图像切分为三个数组，按照权重和颜色增益重新组合
		cvSplit( fA, fsD, fsE, fsF, NULL );
        //两个矩阵对应元素相乘
		cvMul( fsD, fsA, fsD, color_gain);
		cvMul( fsE, fsB, fsE, color_gain );
		cvMul( fsF, fsC, fsF, color_gain );
        //将单通道图像变成多通道的，相当于cvSplit()的逆运算
		cvMerge( fsD, fsE, fsF, NULL, fA );
	}

	// Restore
    // 恢复图像
	cvConvertScale( fA, img, gain, offset);

    // 释放缓存图像
	cvReleaseImage( &fA );
	cvReleaseImage( &fB );
	cvReleaseImage( &fC );
	cvReleaseImage( &fsA );
	cvReleaseImage( &fsB );
	cvReleaseImage( &fsC );
	cvReleaseImage( &fsD );
	cvReleaseImage( &fsE );
	cvReleaseImage( &fsF );
}
void MultiScaleRetinexCR(Mat src, Mat &dst, vector<double> weights, vector<double> sigmas,int gain, int offset, double restoration_factor, double color_gain)
{
    IplImage tmp_ipl;
    tmp_ipl = src;
    MultiScaleRetinexCR(&tmp_ipl, weights, sigmas, gain, offset, restoration_factor, color_gain);
    dst = cvarrToMat(&tmp_ipl);
}
//JNICALL Java_com_demo_opencvdemo_MainActivity_getEdge
//CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
////判断图片是位图格式有RGB_565 、RGBA_8888
//CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
//          info.format == ANDROID_BITMAP_FORMAT_RGB_565);
//CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
//CV_Assert(pixels);

extern "C" JNIEXPORT void
JNICALL Java_com_demo_nightvision_jniTools_getEdge
        (JNIEnv *env, jobject obj, jobject bitmap) {
    AndroidBitmapInfo info;
    void *pixels;

    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    //判断图片是位图格式有RGB_565 、RGBA_8888
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);

    Mat image(info.height, info.width, CV_8UC4, pixels);

    vector<double> sigema;
    vector<double> weight;
    for(int i = 0; i < 3; i++)
        weight.push_back(1./3);
    sigema.push_back(30);
    sigema.push_back(150);
    sigema.push_back(300);
//    double d[image.rows][image.cols];
    MultiScaleRetinexCR(image,image,weight,sigema,128,128,6,2);

    imwrite("/sdcard/DCIM/AAAA.jpg", image);

}

