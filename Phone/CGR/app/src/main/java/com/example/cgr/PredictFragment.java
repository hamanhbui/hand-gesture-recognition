package com.example.cgr;

import android.content.Context;
import android.graphics.Color;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import com.example.cgr.MainActivity;
import com.example.cgr.R;
import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;
import com.google.gson.JsonObject;

import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;


/**
 * A simple {@link Fragment} subclass.
 */
public class PredictFragment extends Fragment {

    private float x = 0, y = 0, z = 0, x1 = 0, y1 = 0, z1 = 0;
    private final int N_SAMPLES = 50;
    private Thread visualizeAcceThread;
    private List<Float> listX = new ArrayList<>();
    private List<Float> listY = new ArrayList<>();
    private List<Float> listZ = new ArrayList<>();
    private List<Float> listX1 = new ArrayList<>();
    private List<Float> listY1 = new ArrayList<>();
    private List<Float> listZ1 = new ArrayList<>();
    private List<Float> input_signal = new ArrayList<>();
    private LineChart mChart;
    private TextView predictTextView;
    private MainActivity mActivity;
    private OpenHabApi mOpenHabApi;
    private int n=50;
    private String mWatchIP;
    private boolean isSending=false;
    private boolean startGesture=false;
    private String previousGesture="";
    public PredictFragment() {
        // Required empty public constructor
    }

    public void setmWatchIP(String mWatchIP){
        this.mWatchIP=mWatchIP;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_predict, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        HttpLoggingInterceptor interceptor = new HttpLoggingInterceptor();
        interceptor.setLevel(HttpLoggingInterceptor.Level.BODY);
        OkHttpClient client = new OkHttpClient.Builder().addInterceptor(interceptor).build();
        //'http://192.168.1.86:8080/rest/' is server address
        mOpenHabApi = new Retrofit.Builder()
                .baseUrl("http://192.168.1.86:8080/rest/")
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build().create(OpenHabApi.class);


        initViews(view);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                startStreaming();
            }
        });
        thread.start();
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        mActivity = (MainActivity) context;
    }

    @Override
    public void onDestroy() {
        visualizeAcceThread.interrupt();
        visualizeAcceThread = null;
        super.onDestroy();
    }
    private void sendsACommandToYeelight(String bodyText) {
        RequestBody body =
                RequestBody.create(MediaType.parse("text/plain"), bodyText);
        //'yeelight_wonder_0x0000000007e3c802_color' is device's ID
        mOpenHabApi.sendsACommandToAnItem("yeelight_wonder_0x0000000007e3c802_color", body).enqueue(new Callback<Void>() {
            @Override
            public void onResponse(Call<Void> call, Response<Void> response) {
//                Toast.makeText(getActivity(), response.toString(), Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onFailure(Call<Void> call, Throwable t) {
//                Toast.makeText(getActivity(), t.toString(), Toast.LENGTH_SHORT).show();
            }
        });
    }

    public void getYeelightState(final int actionType) {
        mOpenHabApi.getASingleItem("yeelight_wonder_0x0000000007e3c802_color")
                .enqueue(new Callback<JsonObject>() {
                    @Override
                    public void onResponse(Call<JsonObject> call,
                                           Response<JsonObject> response) {
                        JsonObject stateResponse = response.body();
                        String state = stateResponse.get("state").getAsString();
                        switch (actionType) {
                            case 0: {
                                String[] splitted = state.split(",", 3);
                                int b = Integer.parseInt(splitted[2]);
                                if (b == 0) {
                                    sendsACommandToYeelight("ON");
                                } else {
                                    sendsACommandToYeelight("OFF");
                                }
                                break;
                            }
                            case 1: {
                                int r = (int) ((Math.random() * ((100 - 1) + 1)) + 1);
                                int g = (int) ((Math.random() * ((100 - 1) + 1)) + 1);
                                int b = (int) ((Math.random() * ((100 - 1) + 1)) + 1);
                                String msg = String.valueOf(r) + "," + String.valueOf(g) + "," + String.valueOf(b);
                                sendsACommandToYeelight(msg);
                                break;
                            }
                            case 2: {
                                int r = (int) ((Math.random() * ((100 - 1) + 1)) + 1);
                                int g = (int) ((Math.random() * ((100 - 1) + 1)) + 1);
                                int b = (int) ((Math.random() * ((100 - 1) + 1)) + 1);
                                String msg = String.valueOf(r) + "," + String.valueOf(g) + "," + String.valueOf(b);
                                sendsACommandToYeelight(msg);
                                break;
                            }
                            case 3: {
                                String[] splitted = state.split(",", 3);
                                int b = Integer.parseInt(splitted[2]);
                                if (b != 0) {
                                    if (n < 100) {
                                        n += 20;
                                        if (n > 100)
                                            n = 100;
                                    }
                                    sendsACommandToYeelight(String.valueOf(n));
                                }
                                break;
                            }

                            case 4: {
                                String[] splitted = state.split(",", 3);
                                int b = Integer.parseInt(splitted[2]);
                                if (b != 0) {
                                    if (n > 1) {
                                        n -= 20;
                                        if (n < 1)
                                            n = 1;
                                    }
                                    sendsACommandToYeelight(String.valueOf(n));
                                }
                                break;
                            }
                        }
                    }

                    @Override
                    public void onFailure(Call<JsonObject> call, Throwable t) {
                    }
                });
    }

    private void initViews(View view) {
        mChart = view.findViewById(R.id.chart1);
        predictTextView = view.findViewById(R.id.textView4);
        acceGraphInit();
    }

    private void startStreaming() {
        try {
            DatagramSocket socket = new DatagramSocket(5556);
            feedMultiple();
            byte[] buffer = new byte[2048];
            while (true) {
                if(isSending==false) {
                    DatagramPacket receivePacket = new DatagramPacket(buffer, buffer.length);
                    socket.receive(receivePacket);
                    ByteArrayInputStream bais = new ByteArrayInputStream(buffer);
                    ObjectInputStream ois = new ObjectInputStream(bais);
                    ArrayList<Object> listData = (ArrayList<Object>) ois.readObject();
                    for (int i = 0; i < listData.size(); i += 2) {
                        float[] a = (float[]) listData.get(i + 1);
                        x = a[0];
                        y = a[1];
                        z = a[2];
                        x1 = a[3];
                        y1 = a[4];
                        z1 = a[5];
                        listX.add(x);
                        listY.add(y);
                        listZ.add(z);
                        listX1.add(x1);
                        listY1.add(y1);
                        listZ1.add(z1);
                        activityPrediction();
                        long startTime2 = System.currentTimeMillis();
                        long currentTime2 = startTime2;
                        while (currentTime2 < startTime2 + 35) {
                            currentTime2 = System.currentTimeMillis();
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    private void activityPrediction() {
        if (listX.size() == N_SAMPLES && listY.size() == N_SAMPLES && listZ.size() == N_SAMPLES &&
                listX1.size() == N_SAMPLES && listY1.size() == N_SAMPLES &&
                listZ1.size() == N_SAMPLES) {
            normalize();
            for (int i = 0; i < listX.size(); ++i) {
                input_signal.add(listX.get(i));
                input_signal.add(listY.get(i));
                input_signal.add(listZ.get(i));
                input_signal.add(listX1.get(i));
                input_signal.add(listY1.get(i));
                input_signal.add(listZ1.get(i));
            }
            final float[] results = mActivity.getActivityInference().getActivityProb(toFloatArray(input_signal));
            final String predictValue=getPredictValue(results);
            setPrediction(predictValue);
            listX.clear();
            listY.clear();
            listZ.clear();
            listX1.clear();
            listY1.clear();
            listZ1.clear();
            input_signal.clear();
        }
    }
    private void setPrediction(final String predictValue){
        if(startGesture){
            if(predictValue.equals("Unknown")){
                startGesture=false;
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + predictValue);
                    }
                });
                previousGesture=predictValue;
            }else if(predictValue.equals("Start_gesture")){
                startGesture=true;
                Thread startGestureThread=new Thread(new Runnable() {
                    @Override
                    public void run() {
                        startGestureStreaming();
                    }
                });
                startGestureThread.start();
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + predictValue);
                    }
                });
                previousGesture=predictValue;
            }else if((previousGesture.equals("Start_move_left")&&!predictValue.equals("Move_left"))||(previousGesture.equals("Start_move_right")&&!predictValue.equals("Move_right"))
                    ||(previousGesture.equals("Start_move_down")&&!predictValue.equals("Move_down"))||(previousGesture.equals("Start_move_up")&&!predictValue.equals("Move_up"))){
                startGesture=false;
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + "Unknown");
                    }
                });
                previousGesture="Unknown";
            }else if(predictValue.equals("Start_move_left")){
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + predictValue);
                    }
                });
                previousGesture = predictValue;
                getYeelightState(2);
            }else if(predictValue.equals("Start_move_right")){
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + predictValue);
                    }
                });
                previousGesture = predictValue;
                getYeelightState(1);
            }else if(predictValue.equals("Start_move_up")){
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + predictValue);
                    }
                });
                previousGesture = predictValue;
                getYeelightState(3);
            }else if(predictValue.equals("Start_move_down")){
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + predictValue);
                    }
                });
                previousGesture = predictValue;
                getYeelightState(4);
            }else if(predictValue.equals("Move_down")){
                if(previousGesture.equals("Start_move_down")) {
                    this.predictTextView.post(new Runnable() {
                        @Override
                        public void run() {
                            predictTextView.setText("Predict: " + predictValue);
                        }
                    });
                    getYeelightState(4);
                }
                else {
                    startGesture=false;
                    this.predictTextView.post(new Runnable() {
                        @Override
                        public void run() {
                            predictTextView.setText("Predict: " + "Unknown");
                        }
                    });
                    previousGesture="Unknown";
                }
            }else if(predictValue.equals("Move_up")){
                if(previousGesture.equals("Start_move_up")) {
                    this.predictTextView.post(new Runnable() {
                        @Override
                        public void run() {
                            predictTextView.setText("Predict: " + predictValue);
                        }
                    });
                    getYeelightState(3);
                }
                else{
                    startGesture=false;
                    this.predictTextView.post(new Runnable() {
                        @Override
                        public void run() {
                            predictTextView.setText("Predict: " + "Unknown");
                        }
                    });
                    previousGesture="Unknown";
                }
            }else if(predictValue.equals("Move_left")){
                if(previousGesture.equals("Start_move_left")) {
                    this.predictTextView.post(new Runnable() {
                        @Override
                        public void run() {
                            predictTextView.setText("Predict: " + predictValue);
                        }
                    });
                    getYeelightState(2);
                }
                else{
                    startGesture=false;
                    this.predictTextView.post(new Runnable() {
                        @Override
                        public void run() {
                            predictTextView.setText("Predict: " + "Unknown");
                        }
                    });
                    previousGesture="Unknown";
                }
            }else if(predictValue.equals("Move_right")){
                if(previousGesture.equals("Start_move_right")) {
                    this.predictTextView.post(new Runnable() {
                        @Override
                        public void run() {
                            predictTextView.setText("Predict: " + predictValue);
                        }
                    });
                    getYeelightState(1);
                }
                else{
                    startGesture=false;
                    this.predictTextView.post(new Runnable() {
                        @Override
                        public void run() {
                            predictTextView.setText("Predict: " + "Unknown");
                        }
                    });
                    previousGesture="Unknown";
                }
            }else if(predictValue.equals("Select")){
                startGesture=false;
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + predictValue);
                    }
                });
                previousGesture=predictValue;
                getYeelightState(0);
            }else if (predictValue.equals("0")||predictValue.equals("1")||predictValue.equals("2")||predictValue.equals("3")||predictValue.equals("4")||predictValue.equals("5")
                    ||predictValue.equals("6")||predictValue.equals("7")||predictValue.equals("8")||predictValue.equals("9")||predictValue.equals("CCWCircle")||predictValue.equals("CWCircle")
                    ||predictValue.equals("Clap")){
                startGesture=false;
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + predictValue);
                    }
                });
                previousGesture=predictValue;
            }
        }else{
            if(predictValue.equals("Start_gesture")){
                startGesture=true;
                Thread startGestureThread=new Thread(new Runnable() {
                    @Override
                    public void run() {
                        startGestureStreaming();
                    }
                });
                startGestureThread.start();
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + predictValue);
                    }
                });
                previousGesture=predictValue;
            }else{
                startGesture=false;
                this.predictTextView.post(new Runnable() {
                    @Override
                    public void run() {
                        predictTextView.setText("Predict: " + "Unknown");
                    }
                });
                previousGesture="Unknown";
            }
        }
    }
    private void startGestureStreaming(){
        try {
            isSending=true;
            listX.clear();
            listY.clear();
            listZ.clear();
            listX1.clear();
            listY1.clear();
            listZ1.clear();
            input_signal.clear();
            Socket socket = new Socket(this.mWatchIP, 5557);
            ObjectOutputStream oos = new ObjectOutputStream(socket.getOutputStream());
            oos.writeObject("Writting!");
            ObjectInputStream ois = new ObjectInputStream(socket.getInputStream());
            String response = (String) ois.readObject();
            if(response.equals("OK!")){
                long startTime =   System.currentTimeMillis();
                long currentTime = startTime;
                while(currentTime<startTime+1000) {
                    currentTime = System.currentTimeMillis();
                }
                isSending=false;
            }
            socket.close();
        }catch(Exception e){
            System.out.println(e.toString());
        }
    }
    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private String getPredictValue(final float results[]) {
        float max = results[0];
        int index = 0;
        for (int i = 0; i < results.length; i++) {
            if(startGesture&&(i==18||i==23))
                continue;
            if (max < results[i]) {
                max = results[i];
                index = i;
            }
        }if (index==0){
            return "0";
        }else if (index==1){
            return "1";
        }else if (index==2){
            return "2";
        }else if (index==3){
            return "3";
        }else if (index == 4){
            return "4";
        }else if (index == 5){
            return "5";
        }else if (index == 6){
            return "6";
        }else if (index == 7){
            return "7";
        }else if (index == 8){
            return "8";
        }else if (index == 9){
            return "9";
        }else if (index == 10){
            return "CCWCircle";
        }else if (index == 11){
            return "CWCircle";
        }else if (index == 12){
            return "Clap";
        }else if (index == 13){
            return "Move_down";
        }else if (index == 14){
            return "Move_left";
        }else if (index == 15){
            return "Move_right";
        }else if (index == 16){
            return "Move_up";
        }else if (index == 17){
            return "Select";
        }else if (index == 18){
            return "Start_gesture";
        }else if (index == 19){
            return "Start_move_down";
        }else if (index == 20){
            return "Start_move_left";
        }else if (index == 21){
            return "Start_move_right";
        }else if (index == 22){
            return "Start_move_up";
        }else {
            return "Unknown";
        }
    }

    private void normalize() {
        //These values are output after normalize training data
        float x_m = 2.7203252689593302f;
        float y_m = -0.6992873427607994f;
        float z_m = 7.292192548647218f;
        float x_s = 5.1338035293393975f;
        float y_s = 3.9269456271302245f;
        float z_s = 4.154542793412262f;
        float x_m1 = -0.01136044864908648f;
        float y_m1 = 0.007768879415007718f;
        float z_m1 = 0.006340345826695809f;
        float x_s1 = 0.9224844617790374f;
        float y_s1 = 1.2469344733466796f;
        float z_s1 = 1.146760450516465f;

        for (int i = 0; i < N_SAMPLES; i++) {
            listX.set(i, ((listX.get(i) - x_m) / x_s));
            listY.set(i, ((listY.get(i) - y_m) / y_s));
            listZ.set(i, ((listZ.get(i) - z_m) / z_s));
            listX1.set(i, ((listX1.get(i) - x_m1) / x_s1));
            listY1.set(i, ((listY1.get(i) - y_m1) / y_s1));
            listZ1.set(i, ((listZ1.get(i) - z_m1) / z_s1));
        }
    }

    private void acceGraphInit(){
        // enable description text
        mChart.getDescription().setEnabled(true);
        mChart.setContentDescription("Accelerometer Legenda");

        // set an alternative background color
        mChart.setBackgroundColor(Color.LTGRAY);

        LineData data = new LineData();
        data.setValueTextColor(Color.WHITE);

        // add empty data
        mChart.setData(data);
        mChart.getDescription().setEnabled(false);

        // get the legend (only possible after setting data)
        Legend l = mChart.getLegend();

        // modify the legend ...
        l.setForm(Legend.LegendForm.LINE);
        l.setTextColor(Color.WHITE);

        XAxis xl = mChart.getXAxis();
        xl.setTextColor(Color.WHITE);
        xl.setDrawGridLines(false);
        xl.setAvoidFirstLastClipping(true);
        xl.setEnabled(false);

        YAxis leftAxis = mChart.getAxisLeft();
        leftAxis.setTextColor(Color.WHITE);
        leftAxis.setAxisMaximum(20f);
        leftAxis.setAxisMinimum(-20f);
        leftAxis.setDrawGridLines(true);

        YAxis rightAxis = mChart.getAxisRight();
        rightAxis.setEnabled(false);
    }

    private void feedMultiple() {
        final Runnable runnable = new Runnable() {
            @Override
            public void run() {
                addEntry();
            }
        };

        visualizeAcceThread = new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    getActivity().runOnUiThread(runnable);
                    try {
                        Thread.sleep(25);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        });

        visualizeAcceThread.start();
    }

    private void addEntry() {
        LineData data = mChart.getData();

        if (data != null) {
            ILineDataSet setX = data.getDataSetByIndex(0);
            ILineDataSet setY = data.getDataSetByIndex(1);
            ILineDataSet setZ = data.getDataSetByIndex(2);
            ILineDataSet setX1 = data.getDataSetByIndex(3);
            ILineDataSet setY1 = data.getDataSetByIndex(4);
            ILineDataSet setZ1 = data.getDataSetByIndex(5);

            if (setX == null || setY == null || setZ == null) {
                setX = createSet("Acce X", Color.BLUE);
                setY = createSet("Acce Y", Color.RED);

                setZ = createSet("Acce Z", Color.GREEN);
                setX1 = createSet("Gyro X", Color.GRAY);
                setY1 = createSet("Gyro Y", Color.BLACK);

                setZ1 = createSet("Gyro Z", Color.YELLOW);
                data.addDataSet(setX);
                data.addDataSet(setY);
                data.addDataSet(setZ);
                data.addDataSet(setX1);
                data.addDataSet(setY1);
                data.addDataSet(setZ1);
            }

            data.addEntry(new Entry(setX.getEntryCount(), x), 0);
            data.addEntry(new Entry(setY.getEntryCount(), y), 1);
            data.addEntry(new Entry(setZ.getEntryCount(), z), 2);
            data.addEntry(new Entry(setX1.getEntryCount(), x1), 3);
            data.addEntry(new Entry(setY1.getEntryCount(), y1), 4);
            data.addEntry(new Entry(setZ1.getEntryCount(), z1), 5);

            data.notifyDataChanged();

            mChart.notifyDataSetChanged();
            mChart.setVisibleXRangeMaximum(120);
            mChart.moveViewToX(data.getEntryCount());
        }
    }

    private LineDataSet createSet(String labelName, int colorValue) {
        LineDataSet set = new LineDataSet(null, labelName);
        set.setColor(colorValue);
        set.setDrawCircles(false);
        set.setDrawValues(false);
        return set;
    }
}
