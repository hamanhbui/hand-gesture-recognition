package com.example.cgr;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.os.VibrationEffect;
import android.support.wearable.activity.WearableActivity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.os.Vibrator;

import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
public class MainActivity extends WearableActivity  {

    protected PowerManager.WakeLock mWakeLock;

    public TextView mTextView;
    private DatagramSocket socket;
    private SensorManager sensorManager;
    private Sensor accelerometerSensor;
    private Sensor gyroscopeSensor;
    private Sensor magneticSensor;
    public InetAddress ipMobile;
    private SimpleDateFormat sdf;
    private Vibrator vibrator;
    private float xAcce=0,yAcce=0,zAcce=0,xGyro=0,yGyro=0,zGyro=0,xMagn=0,yMagn=0,zMagn=0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        final PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        this.mWakeLock = pm.newWakeLock(PowerManager.SCREEN_DIM_WAKE_LOCK, "My Tag");
        this.mWakeLock.acquire();

        mTextView = (TextView) findViewById(R.id.text);
        this.vibrator=(Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        // Enables Always-on
        setAmbientEnabled();
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        accelerometerSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroscopeSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        magneticSensor=sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        sensorManager.registerListener(new Accelerometer(), accelerometerSensor, SensorManager.SENSOR_DELAY_FASTEST);
        sensorManager.registerListener(new Gyroscope(),gyroscopeSensor,SensorManager.SENSOR_DELAY_FASTEST);
        sensorManager.registerListener(new Magnetic(),magneticSensor,SensorManager.SENSOR_DELAY_FASTEST);

        mTextView.setText("Waiting for connectivity with mobile device.....");
        ConnectAsyncTask connectAsyncTask=new ConnectAsyncTask(MainActivity.this);
        connectAsyncTask.execute();
        sdf = new SimpleDateFormat("yyyy-MM-dd;HH:mm:ss.SSS");
    }
    public void createSocket(){
        try{
            socket=new DatagramSocket(5556);
            Thread sensorThread=new Thread(new Runnable() {
                @Override
                public void run() {
                    startStreaming();
                }
            });
            sensorThread.start();
            Thread startGestureThread=new Thread(new Runnable() {
                @Override
                public void run() {
                    startGestureStreaming();
                }
            });
            startGestureThread.start();
        }catch(Exception e){
            System.out.println(e.toString());
        }
    }
    public void startGestureStreaming(){
        try{
            ServerSocket serverSocket=new ServerSocket(5557);
            while(true) {
                Socket socket = serverSocket.accept();
                ObjectInputStream ois = new ObjectInputStream(socket.getInputStream());
                String ping = (String) ois.readObject();
                if (ping.equals("Writting!")) {
                    if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.O){
                        vibrator.vibrate(VibrationEffect.createOneShot(500,VibrationEffect.DEFAULT_AMPLITUDE));
                    }else{
                        vibrator.vibrate(500);
                    }
                    ObjectOutputStream oos = new ObjectOutputStream(socket.getOutputStream());
                    oos.writeObject("OK!");
                }
            }
        }catch(Exception e){

        }
    }
    private void startStreaming(){
        try {
            while(true) {
                ByteArrayOutputStream baos=new ByteArrayOutputStream();
                long startTime =   System.currentTimeMillis();
                long currentTime = startTime;
                ObjectOutputStream oos=new ObjectOutputStream(baos);
                ArrayList<Object> listData=new ArrayList<>();
                while(currentTime<startTime+500){
                    currentTime = System.currentTimeMillis();
                    float[] a={xAcce,yAcce,zAcce,xGyro,yGyro,zGyro,xMagn,yMagn,zMagn};
                    String currentDateandTime = sdf.format(new Date());
                    listData.add(currentDateandTime);
                    listData.add(a);
                    long startTime2 =   System.currentTimeMillis();
                    long currentTime2 =startTime2;
                    while(currentTime2<startTime2+40){
                        currentTime2 = System.currentTimeMillis();
                    }
                }
                oos.writeObject(listData);
                oos.flush();
                byte[] buffer= baos.toByteArray();
                DatagramPacket packet=new DatagramPacket(buffer,buffer.length,ipMobile,5556);
                socket.send(packet);
            }
        }catch (Exception e){
            System.out.println(e.toString());
        }
    }

    class Accelerometer implements SensorEventListener{

        @Override
        public void onSensorChanged(SensorEvent event) {
            if(event.sensor.getType()!=Sensor.TYPE_ACCELEROMETER)
                return;
            xAcce = event.values[0];
            yAcce = event.values[1];
            zAcce = event.values[2];
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {

        }
    }

    class Gyroscope implements SensorEventListener{

        @Override
        public void onSensorChanged(SensorEvent event) {
            if(event.sensor.getType()!=Sensor.TYPE_GYROSCOPE)
                return;
            xGyro = event.values[0];
            yGyro = event.values[1];
            zGyro = event.values[2];
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {

        }
    }

    class Magnetic implements SensorEventListener{

        @Override
        public void onSensorChanged(SensorEvent event) {
            if(event.sensor.getType()!=Sensor.TYPE_MAGNETIC_FIELD)
                return;
            xMagn = event.values[0];
            yMagn = event.values[1];
            zMagn = event.values[2];
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {

        }
    }
}
