package com.demo.nightvision;

import androidx.appcompat.app.AppCompatActivity;

import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.os.Environment;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {
    //获取视图控件
    private EditText et_username;
    private EditText et_psw;
    private Button bt_login;
    private ImageView im_showPhoto;

    private DatabaseHelper dbHelper;
    private SQLiteDatabase database;
    private EditText editText, editText2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //获取控件
        et_username = (EditText) findViewById(R.id.et_username);
        et_psw = (EditText) findViewById(R.id.et_password);


        //new出数据库，（上下文，版本号）
        dbHelper = new DatabaseHelper(this, 1);
    }

    //注册创建
    public void doInsert(String string) {
        //创建数据库
        database = dbHelper.getReadableDatabase();
        //用ContentValues来放值
        ContentValues values = new ContentValues();
        values.put("name", string);
        //插入数据
        database.insert("person", null, values);
        //Log.e("e", string);
        Toast.makeText(MainActivity.this, "用户创建成功", Toast.LENGTH_SHORT).show();

    }

    //查询用户并跳转
    public int doQuery(String input) {
        database = dbHelper.getReadableDatabase();
        Cursor cursor = database.query(false, "person", null, null, null, null, null, null, null);
        while (cursor.moveToNext()) {
            String name = cursor.getString(0);
            if (name.equals(input)) {
                Intent intent = new Intent(MainActivity.this, MSRSRActivity.class);
                startActivity(intent);
                Toast.makeText(MainActivity.this, "登录成功", Toast.LENGTH_SHORT).show();
                return 1;
            }
        }
        return 0;
    }


    //跳转到处理图像页面jniactivity.xml
    public void turn(View view) {
        Intent intent = new Intent(MainActivity.this, MSRSRActivity.class);
        startActivity(intent);
    }

    //数据库对象
    SQLiteDatabase db;
    //使用数据库


    //注册
    public void logIn(View v) {
        String username = (String) et_username.getText().toString().trim();
        String psw = (String) et_psw.getText().toString().trim();
        if (TextUtils.isEmpty(username) || TextUtils.isEmpty(psw)) {
            Toast.makeText(MainActivity.this, "用户名和密码不能为空", Toast.LENGTH_SHORT).show();
        } else {
            String name_psw = username + "+" + psw;
            doInsert(name_psw);
            et_username.setText("");
            et_psw.setText("");
        }
    }
    //登录
    public void logOn(View v) {
        //获取输入，判段是否为空
        String username = (String) et_username.getText().toString().trim();
        String psw = (String) et_psw.getText().toString().trim();
        if (TextUtils.isEmpty(username) || TextUtils.isEmpty(psw)) {
            Toast.makeText(MainActivity.this, "用户名和密码不能为空", Toast.LENGTH_SHORT).show();
        } else {
            Log.d("MainActivity", "正在登录");
            String name_psw = username + "+" + psw;
            //数据库查询用户名和密码
            if (doQuery(name_psw) == 0) {
                Toast.makeText(MainActivity.this, "登录失败，用户不存在", Toast.LENGTH_SHORT).show();
            }
            ;
        }
    }
}
