package com.demo.nightvision;

import android.Manifest;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;


import static com.demo.nightvision.jniTools.getEdge;

public class MSRSRActivity extends AppCompatActivity implements View.OnClickListener {
    private ImageView imageView;
//    private jniTools jniTools;
    //相册图片的地址
    private String photoAlbumPath;
    //用于读取图片的文件输入流
    FileInputStream fis = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.jniactivity);
        imageView = findViewById(R.id.imageView);
        findViewById(R.id.show).setOnClickListener(this);
        findViewById(R.id.process).setOnClickListener(this);
    }

    @Override
    public void onClick(View v) {
        //相册的onclick事件
        if (v.getId() == R.id.show) {
            //调用系统相册
            Intent intent = new Intent();
            intent.setAction(Intent.ACTION_PICK);
            intent.setType("image/*");
            startActivityForResult(intent, 2);
        }

        //图片处理的onclick事件
        else if(v.getId() == R.id.process) {
            if (photoAlbumPath != ""){
                //从photoAlbumPath读取出所选图片（bitmap类型）
               Bitmap bitmap = BitmapFactory.decodeFile(photoAlbumPath);
                //调用jni，对图片使用算法
                getEdge(bitmap);
                String myJpgPath = "/sdcard/DCIM/AAAA.jpg";
                //使用文件流获取处理后的图片
                try {
                    fis = new FileInputStream(myJpgPath);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
                Bitmap bm = BitmapFactory.decodeStream(fis);

                imageView.setImageBitmap(bm);
               Toast.makeText(MSRSRActivity.this,"图片增强完成",Toast.LENGTH_SHORT).show();
            }else {
               Toast.makeText(MSRSRActivity.this,"请先点击相册选取图片",Toast.LENGTH_SHORT).show();
           }

        }
    }
    //照相机的onclick事件
    private File file;
    private Uri imageUri;
    public void takePhoto(View view){

        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // 指定名字
        String f = System.currentTimeMillis()+".jpg";
        // 指定文件
        file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM), f);
        //封装成Uri
        imageUri = Uri.fromFile(file);
        //保存图片位置
        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);

        startActivityForResult(cameraIntent, 1);
    }



    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        String photoPath;
        if (requestCode == 1 && resultCode == RESULT_OK) {

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                photoPath = String.valueOf(file);
            } else {
                photoPath = imageUri.getEncodedPath();
            }
            Log.d("拍照返回图片路径:", photoPath);
            photoAlbumPath = photoPath;
            Bitmap bm = BitmapFactory.decodeFile(photoPath);
            imageView.setImageBitmap(bm);


        } else if (requestCode == 2 && resultCode == RESULT_OK) {
            photoPath = getPhotoFromPhotoAlbum.getRealPathFromUri(this, data.getData());
            photoAlbumPath = photoPath;
//            Glide.with(MainActivity.this).load(photoPath).into(ivTest);
            Bitmap bm = BitmapFactory.decodeFile(photoPath);
            imageView.setImageBitmap(bm);
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

}