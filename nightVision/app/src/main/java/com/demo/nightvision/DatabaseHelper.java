package com.demo.nightvision;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

public class DatabaseHelper extends SQLiteOpenHelper {
    public DatabaseHelper(Context context, int version) {
        super(context, "persons", null, version);
    }
    //当数据库创建时执行**
    @Override
    public void onCreate(SQLiteDatabase db) {
    	//创建表**
        String sql="CREATE TABLE person (name varchar)";
        db.execSQL(sql);
        System.out.println("数据库创建成功");

    }
    //用于数据库升级时执行**
    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
    }
}
